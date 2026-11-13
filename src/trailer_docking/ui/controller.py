"""Headless logic behind the control UI.

:class:`AppController` owns the editable :class:`Scenario`, one background
training campaign, one background docking test, and the loaded policy bundle.
Training and tests run on the 2D or the 3D simulator, chosen per run.
It imports neither
tkinter nor matplotlib, and it imports the training package only when a run
is actually started, so everything here can be unit-tested without a display
and without the reinforcement-learning dependencies.

The view talks to the controller with plain strings (the text of the entry
fields) and polls it from the GUI thread; background threads never call back
into the view.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, fields, replace
from math import cos, isfinite, sin
from pathlib import Path
from typing import Any

from trailer_docking.core.state import Point2D, VehicleState
from trailer_docking.evaluation.scenarios import (
    BuiltScenario,
    Scenario,
    ScenarioOverrides,
    StartDistribution,
    assembled_start,
)

GROUPS = (
    "Dock",
    "Route",
    "Vehicle",
    "Limits",
    "Physics (3D only)",
    "Start pose",
    "Randomization",
)
BACKENDS = ("2d", "3d")
PHYSICS_GROUP = "Physics (3D only)"
PHYSICS_GROUP_NOTE = "Used by the 3D simulator only"

# Training statuses after which a session no longer occupies the controller.
_TERMINAL_TRAINING = frozenset({"finished", "stopped", "failed"})
_JOIN_TIMEOUT = 2.0


# Marks a test of the model-based controller rather than a policy.
_NMPC = object()


class ControllerError(Exception):
    """Base class for errors the view should show to the user."""


class ParameterError(ControllerError, ValueError):
    """Raised when scenario parameters cannot be parsed, built, or applied."""


class BusyError(ControllerError, RuntimeError):
    """Raised when an action is refused because training or a test is active."""


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """Describes one editable scenario parameter for the view."""

    key: str
    label: str
    group: str
    kind: str
    choices: tuple[str, ...] = ()
    unit: str = ""
    physics_only: bool = False

    def __post_init__(self) -> None:
        if self.group not in GROUPS:
            raise ValueError(f"unknown parameter group: {self.group}")
        if self.kind not in {"float", "choice", "range", "optional_range"}:
            raise ValueError(f"unknown parameter kind: {self.kind}")


PARAMETERS: tuple[ParameterSpec, ...] = (
    ParameterSpec(
        "dock_interior_width", "Bay interior width", "Dock", "float", unit="m"
    ),
    ParameterSpec(
        "dock_interior_depth", "Bay interior depth", "Dock", "float", unit="m"
    ),
    ParameterSpec(
        "planner",
        "Route planner",
        "Route",
        "choice",
        choices=("", "entry", "astar"),
    ),
    ParameterSpec(
        "path_lateral_weight", "Reward: stay on route", "Route", "float", unit="-"
    ),
    ParameterSpec(
        "path_heading_weight", "Reward: align to route", "Route", "float", unit="-"
    ),
    ParameterSpec(
        "trailer_variant",
        "Trailer variant",
        "Vehicle",
        "choice",
        choices=("", "short", "nominal", "long"),
    ),
    ParameterSpec(
        "tractor_wheelbase", "Tractor wheelbase", "Vehicle", "float", unit="m"
    ),
    ParameterSpec(
        "tractor_body_width", "Tractor body width", "Vehicle", "float", unit="m"
    ),
    ParameterSpec(
        "trailer_hitch_to_axle", "Trailer hitch-to-axle", "Vehicle", "float", unit="m"
    ),
    ParameterSpec(
        "trailer_rear_overhang", "Trailer rear overhang", "Vehicle", "float", unit="m"
    ),
    ParameterSpec(
        "trailer_body_width", "Trailer body width", "Vehicle", "float", unit="m"
    ),
    ParameterSpec(
        "steering_angle_max", "Max steering angle", "Limits", "float", unit="rad"
    ),
    ParameterSpec(
        "steering_rate_max", "Max steering rate", "Limits", "float", unit="rad/s"
    ),
    ParameterSpec("reverse_speed", "Reverse speed", "Limits", "float", unit="m/s"),
    ParameterSpec(
        "tractor_chassis_mass",
        "Tractor chassis mass",
        PHYSICS_GROUP,
        "float",
        unit="kg",
        physics_only=True,
    ),
    ParameterSpec(
        "trailer_body_mass",
        "Trailer body mass",
        PHYSICS_GROUP,
        "float",
        unit="kg",
        physics_only=True,
    ),
    ParameterSpec(
        "ground_friction",
        "Ground friction",
        PHYSICS_GROUP,
        "float",
        unit="-",
        physics_only=True,
    ),
    ParameterSpec("standoff", "Standoff from opening", "Start pose", "range", unit="m"),
    ParameterSpec("lateral", "Lateral offset", "Start pose", "range", unit="m"),
    ParameterSpec(
        "trailer_yaw", "Trailer yaw offset", "Start pose", "range", unit="rad"
    ),
    ParameterSpec("articulation", "Articulation", "Start pose", "range", unit="rad"),
    ParameterSpec(
        "random_trailer_hitch_to_axle",
        "Trailer length per episode",
        "Randomization",
        "optional_range",
        unit="m",
    ),
    ParameterSpec(
        "observation_noise_std",
        "Observation noise (std)",
        "Randomization",
        "float",
        unit="-",
    ),
)

_OVERRIDE_KEYS = frozenset(item.name for item in fields(ScenarioOverrides))
_START_KEYS = frozenset(item.name for item in fields(StartDistribution))
_PLANNER_KEY = "planner"
_RANDOM_TRAILER_KEY = "random_trailer_hitch_to_axle"
_NOISE_KEY = "observation_noise_std"
_SPEC_BY_KEY = {spec.key: spec for spec in PARAMETERS}


def _format_number(value: float) -> str:
    return repr(float(value))


def _parse_number(spec: ParameterSpec, text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise ParameterError(f"{spec.label}: '{text}' is not a number") from None
    if not isfinite(value):
        raise ParameterError(f"{spec.label}: value must be finite")
    return value


def _parse_value(spec: ParameterSpec, text: str) -> Any:
    """Convert one field's text; ``None`` means "use the base value"."""

    text = text.strip()
    if not text:
        return None
    if spec.kind == "choice":
        if text not in spec.choices:
            options = ", ".join(choice or "(base)" for choice in spec.choices)
            raise ParameterError(f"{spec.label}: '{text}' is not one of {options}")
        return text
    if spec.kind in ("range", "optional_range"):
        parts = [part.strip() for part in text.split(",")]
        if len(parts) != 2 or not all(parts):
            raise ParameterError(
                f"{spec.label}: enter a range as 'minimum, maximum', got '{text}'"
            )
        return (_parse_number(spec, parts[0]), _parse_number(spec, parts[1]))
    return _parse_number(spec, text)


def _default_session_factory(
    scenario: Scenario, config: Any, *, output_root: Path, live_3d: bool = False
) -> Any:
    from trailer_docking.training.campaign import TrainingCampaign

    return TrainingCampaign(scenario, config, output_root=output_root, live_3d=live_3d)


def _default_test_runner(scenario: Scenario, policy: Any, **options: Any) -> Any:
    from trailer_docking.training.evaluate import run_docking_test

    return run_docking_test(scenario, policy, **options)


def _default_policy_loader(path: Path) -> Any:
    from trailer_docking.training.evaluate import load_policy

    return load_policy(path)


DEFAULT_PLANNER = "entry"


def _default_baseline_policy() -> Any:
    from trailer_docking.training.evaluate import straight_reverse_policy

    return straight_reverse_policy


class AppController:
    """Scenario editing plus one training session and one docking test.

    ``session_factory(scenario, config, output_root=...)`` must return an
    object with the ``TrainingCampaign`` interface (it also receives
    ``live_3d=True`` when a 3D display was requested), ``test_runner`` has the
    signature of ``run_docking_test``, and ``policy_loader(path)`` returns a
    policy.  All three default to the real training package, imported only
    when first used.  ``baseline_policy`` replaces the untrained
    straight-reverse policy (also imported lazily when omitted), and
    ``route_planner(vehicle_geometry, dock_geometry, start_state)`` replaces
    the scenario's registered planner; it must return an object with
    ``.points``.
    """

    def __init__(
        self,
        scenario: Scenario | None = None,
        *,
        output_root: str | Path = ".",
        session_factory: Callable[..., Any] | None = None,
        test_runner: Callable[..., Any] | None = None,
        policy_loader: Callable[[Path], Any] | None = None,
        baseline_policy: Any = None,
        route_planner: Callable[..., Any] | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._scenario = scenario if scenario is not None else Scenario()
        # A scenario that cannot be built is a start-up error, not a UI state.
        self._built = self._scenario.build()
        self._output_root = Path(output_root)
        self._session_factory = session_factory or _default_session_factory
        self._test_runner = test_runner or _default_test_runner
        self._policy_loader = policy_loader or _default_policy_loader
        self._baseline_policy = baseline_policy
        self._route_planner = route_planner

        self._session: Any = None
        self._test_thread: threading.Thread | None = None
        self._test_stop = threading.Event()
        self._test_status = "idle"
        self._test_results: tuple[Any, ...] = ()
        self._test_error: str | None = None
        self._test_policy = ""
        self._test_state: VehicleState | None = None
        self._test_frame: Any = None
        self._test_stop_requested = False
        # Policy bundle loaded from disk, if any, and where it came from.
        self._bundle: Any = None
        # Which activity most recently owned the live view.
        self._spectator_source: str | None = None
        # Planned route of the episode currently shown, and what it was
        # planned from, so a new episode can be recognised in the feed.
        self._route: tuple[Point2D, ...] | None = None
        self._route_state: VehicleState | None = None
        self._route_marker: Any = None

    # ------------------------------------------------------------ scenario

    @property
    def scenario(self) -> Scenario:
        with self._lock:
            return self._scenario

    @property
    def built(self) -> BuiltScenario:
        """Validated configuration of the last successfully applied scenario."""

        with self._lock:
            return self._built

    @property
    def output_root(self) -> Path:
        return self._output_root

    def parameter_values(self) -> dict[str, str]:
        """Return every parameter as the string shown in its entry field."""

        scenario = self.scenario
        values: dict[str, str] = {}
        for spec in PARAMETERS:
            if spec.key == _PLANNER_KEY:
                values[spec.key] = scenario.planner
                continue
            if spec.key == _RANDOM_TRAILER_KEY:
                bounds = scenario.randomization.trailer_hitch_to_axle
                values[spec.key] = (
                    ""
                    if bounds is None
                    else f"{_format_number(bounds[0])}, {_format_number(bounds[1])}"
                )
                continue
            if spec.key == _NOISE_KEY:
                noise = scenario.randomization.observation_noise_std
                values[spec.key] = "" if noise == 0.0 else _format_number(noise)
                continue
            if spec.kind == "range":
                low, high = getattr(scenario.start, spec.key)
                values[spec.key] = f"{_format_number(low)}, {_format_number(high)}"
                continue
            value = getattr(scenario.overrides, spec.key)
            if value is None:
                values[spec.key] = ""
            elif spec.kind == "choice":
                values[spec.key] = str(value)
            else:
                values[spec.key] = _format_number(value)
        return values

    def apply_parameters(self, values: Mapping[str, str]) -> None:
        """Parse field strings, validate them, and replace the scenario.

        A blank override keeps the base specification, a blank start range
        restores the default range, and a blank randomization field turns that
        randomization off.  Keys that are absent keep their current
        value.  On any problem :class:`ParameterError` is raised and the
        current scenario is left unchanged.
        """

        unknown = sorted(set(values) - set(_SPEC_BY_KEY))
        if unknown:
            raise ParameterError(f"unknown parameters: {', '.join(unknown)}")
        with self._lock:
            self._require_idle("change parameters", ParameterError)
            overrides: dict[str, Any] = {}
            start: dict[str, Any] = {}
            randomization: dict[str, Any] = {}
            planner = self._scenario.planner
            defaults = StartDistribution()
            for key, text in values.items():
                spec = _SPEC_BY_KEY[key]
                parsed = _parse_value(spec, str(text))
                if key == _PLANNER_KEY:
                    planner = parsed or DEFAULT_PLANNER
                elif key == _RANDOM_TRAILER_KEY:
                    randomization["trailer_hitch_to_axle"] = parsed
                elif key == _NOISE_KEY:
                    randomization[key] = 0.0 if parsed is None else parsed
                elif key in _START_KEYS:
                    start[key] = getattr(defaults, key) if parsed is None else parsed
                else:
                    overrides[key] = parsed
            try:
                candidate = replace(
                    self._scenario,
                    overrides=replace(self._scenario.overrides, **overrides),
                    start=replace(self._scenario.start, **start),
                    randomization=replace(
                        self._scenario.randomization, **randomization
                    ),
                    planner=planner,
                )
                built = candidate.build()
            except ValueError as error:
                raise ParameterError(str(error)) from error
            self._scenario, self._built = candidate, built
            self._forget_spectator()
            self._forget_test()

    def save_scenario(self, path: str | Path) -> Path:
        """Write the current scenario as YAML."""

        try:
            return self.scenario.save(path)
        except OSError as error:
            raise ControllerError(f"cannot save scenario: {error}") from error

    def load_scenario(self, path: str | Path) -> None:
        """Replace the scenario with one read from *path* after validating it."""

        with self._lock:
            self._require_idle("load a scenario", ParameterError)
            try:
                candidate = Scenario.load(path)
                built = candidate.build()
            except ValueError as error:
                raise ParameterError(str(error)) from error
            self._scenario, self._built = candidate, built
            self._forget_spectator()
            self._forget_test()

    def preview_state(self) -> VehicleState:
        """Return the centre of the start distribution, for an idle view."""

        with self._lock:
            start, built = self._scenario.start, self._built
        bay = built.dock.geometry
        geometry = built.vehicle.geometry

        def middle(bounds: tuple[float, float]) -> float:
            return (bounds[0] + bounds[1]) / 2.0

        yaw = bay.target_trailer_yaw
        along = bay.interior_depth + middle(start.standoff)
        lateral = middle(start.lateral)
        point = Point2D(
            along * cos(yaw) - lateral * sin(yaw),
            along * sin(yaw) + lateral * cos(yaw),
        )
        tractor_pose, trailer_yaw = assembled_start(
            geometry,
            point,
            yaw + middle(start.trailer_yaw),
            middle(start.articulation),
        )
        return geometry.assembled_state(tractor_pose, trailer_yaw)

    # ------------------------------------------------------------ training

    def make_training_config(self, **changes: Any) -> Any:
        """Return a ``TrainingConfig`` with *changes* applied to the defaults."""

        try:
            from trailer_docking.training.config import TrainingConfig

            return TrainingConfig(**changes)
        except ImportError as error:
            raise ControllerError(f"training is unavailable: {error}") from error
        except (TypeError, ValueError) as error:
            raise ParameterError(f"invalid training settings: {error}") from error

    @property
    def training_progress(self) -> Any:
        """The session's ``TrainingProgress``, or ``None`` before any training."""

        session = self._current_session()
        if session is None:
            return None
        try:
            return session.progress
        except Exception:
            return None

    @property
    def training_active(self) -> bool:
        session = self._current_session()
        return session is not None and self._session_active(session)

    @property
    def episode_history(self) -> tuple[Any, ...]:
        session = self._current_session()
        if session is None:
            return ()
        try:
            return tuple(session.episode_history)
        except Exception:
            return ()

    def start_training(self, config: Any = None, *, show_3d: bool = False) -> None:
        """Start a background training campaign on the current scenario.

        The campaign repeats the run ``config.runs`` times on
        ``config.backend``, tests each policy, and exports the best one as a
        bundle.  ``show_3d`` renders the 3D episodes for :attr:`spectator_frame`;
        it slows 3D training and has no effect on 2D runs.
        """

        with self._lock:
            self._require_idle("start training", BusyError)
            options: dict[str, Any] = {"output_root": self._output_root}
            if show_3d:
                options["live_3d"] = True
            try:
                session = self._session_factory(self._scenario, config, **options)
                session.start()
            except ImportError as error:
                raise ControllerError(f"training is unavailable: {error}") from error
            except ControllerError:
                raise
            except Exception as error:
                raise ControllerError(f"cannot start training: {error}") from error
            self._session = session
            self._forget_spectator()
            self._spectator_source = "training"

    def stop_training(self) -> None:
        """Ask the training session to stop gracefully; never blocks."""

        session = self._current_session()
        if session is None:
            return
        try:
            session.stop()
        except Exception:
            # Stopping is best effort; the status panel shows the real state.
            return

    # ---------------------------------------------------------------- test

    @property
    def test_status(self) -> str:
        with self._lock:
            return self._test_status

    @property
    def test_results(self) -> tuple[Any, ...]:
        with self._lock:
            return self._test_results

    @property
    def test_error(self) -> str | None:
        with self._lock:
            return self._test_error

    @property
    def test_policy(self) -> str:
        """Human-readable description of the policy used by the last test."""

        with self._lock:
            return self._test_policy

    @property
    def latest_checkpoint(self) -> Path | None:
        """Latest checkpoint of the last training session, if it wrote one."""

        progress = self.training_progress
        path = getattr(progress, "checkpoint_path", None)
        return Path(path) if path else None

    def start_test(
        self,
        *,
        policy_path: str | Path | None = None,
        episodes: int = 1,
        seed: int = 0,
        baseline: bool = False,
        backend: str = "2d",
        use_bundle: bool = False,
        method: str = "ppo",
        show_3d: bool = False,
    ) -> None:
        """Run a docking test in a daemon thread on a freshly built simulator.

        With ``policy_path`` omitted the latest training checkpoint is used
        when there is one and the untrained straight-reverse policy
        otherwise.  ``baseline=True`` forces the untrained policy and
        ``use_bundle=True`` runs the policy of the loaded bundle.
        ``method="nmpc"`` tests the model-based controller instead of a
        policy.  ``backend`` is ``"2d"`` or ``"3d"``; ``show_3d`` renders 3D
        episodes for :attr:`spectator_frame`.
        """

        if backend not in BACKENDS:
            raise ParameterError("simulator must be 2d or 3d")
        if method not in ("ppo", "nmpc"):
            raise ParameterError("method must be ppo or nmpc")
        if sum((baseline, policy_path is not None, use_bundle)) > 1:
            raise ParameterError(
                "choose one of a policy file, the loaded bundle, or the baseline"
            )

        if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1:
            raise ParameterError("episodes must be a positive whole number")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ParameterError("seed must be a whole number")
        with self._lock:
            self._require_idle("start a test", BusyError)
            source: Any
            if method == "nmpc":
                source = _NMPC
            elif baseline:
                source = None
            elif use_bundle:
                if self._bundle is None:
                    raise ParameterError("load a policy bundle first")
                source = self._bundle
            elif policy_path is not None:
                source = Path(policy_path)
            else:
                source = self.latest_checkpoint
            stop = threading.Event()
            self._test_stop = stop
            self._test_status = "running"
            self._test_results = ()
            self._test_error = None
            self._forget_spectator()
            self._test_stop_requested = False
            if source is None:
                description = "untrained straight-reverse baseline"
            elif source is _NMPC:
                description = "NMPC controller"
            elif isinstance(source, Path):
                description = str(source)
            else:
                description = f"bundle {source.description()}"
            self._test_policy = f"{description} [{backend}]"
            self._spectator_source = "test"
            thread = threading.Thread(
                target=self._run_test,
                args=(
                    self._scenario,
                    source,
                    episodes,
                    seed,
                    stop,
                    backend,
                    show_3d and backend == "3d",
                ),
                name="docking-test",
                daemon=True,
            )
            self._test_thread = thread
            thread.start()

    def stop_test(self) -> None:
        """Ask the running test to stop at its next step; never blocks."""

        with self._lock:
            if self._test_status == "running":
                self._test_stop_requested = True
            self._test_stop.set()

    def _run_test(
        self,
        scenario: Scenario,
        source: Any,
        episodes: int,
        seed: int,
        stop: threading.Event,
        backend: str = "2d",
        show_3d: bool = False,
    ) -> None:
        def on_state(state: VehicleState) -> None:
            with self._lock:
                if self._test_stop is stop:
                    self._test_state = state
                    self._observe(state)

        def on_frame(frame: Any) -> None:
            with self._lock:
                if self._test_stop is stop:
                    self._test_frame = frame

        status, message, results = "finished", None, ()
        try:
            options: dict[str, Any] = {}
            policy: Any = None
            if source is _NMPC:
                from trailer_docking.training.methods import nmpc_controller_factory

                options["controller_factory"] = nmpc_controller_factory()
            elif isinstance(source, Path):
                policy = self._policy_loader(source)
            elif source is not None:
                policy = source.load_policy()
                options["max_episode_steps"] = source.max_episode_steps
            elif self._baseline_policy is not None:
                policy = self._baseline_policy
            else:
                policy = _default_baseline_policy()
            # Only non-default choices are passed, so a replacement runner
            # needs to understand them only when they are used.
            if backend != "2d":
                options["backend"] = backend
            if show_3d:
                options["on_frame"] = on_frame
            results = tuple(
                self._test_runner(
                    scenario,
                    policy,
                    episodes=episodes,
                    seed=seed,
                    on_state=on_state,
                    should_stop=stop.is_set,
                    **options,
                )
            )
        except BaseException as error:  # nothing may escape a background thread
            status = "failed"
            message = f"{type(error).__name__}: {error}"
        with self._lock:
            if self._test_stop is stop:
                self._test_status = status
                self._test_error = message
                self._test_results = results

    # -------------------------------------------------------------- result

    @property
    def latest_bundle(self) -> Path | None:
        """Bundle exported by the last training campaign, if it wrote one."""

        path = getattr(self.training_progress, "bundle_path", None)
        return Path(path) if path else None

    @property
    def loaded_bundle(self) -> Any:
        """The policy bundle loaded with :meth:`load_result`, if any."""

        with self._lock:
            return self._bundle

    def export_result(self, path: str | Path) -> Path:
        """Copy the last campaign's policy bundle to *path*."""

        import shutil

        source = self.latest_bundle
        if source is None:
            raise ControllerError("there is no finished training result to export yet")
        target = Path(path)
        if target.exists():
            raise ControllerError(f"refusing to overwrite {target}")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        except OSError as error:
            raise ControllerError(f"cannot export the result: {error}") from error
        return target

    def load_result(self, path: str | Path) -> Any:
        """Load a policy bundle and adopt the scenario it was trained on.

        The bundle's policy can then be tested with ``use_bundle=True`` in a
        fresh 2D or 3D simulator.  Returns the loaded bundle.
        """

        with self._lock:
            self._require_idle("load a result", ParameterError)
            try:
                from trailer_docking.training.bundle import load_bundle

                bundle = load_bundle(path)
                built = bundle.scenario.build()
            except (OSError, ValueError) as error:
                raise ParameterError(str(error)) from error
            self._bundle = bundle
            self._scenario, self._built = bundle.scenario, built
            self._forget_spectator()
            self._forget_test()
        return bundle

    def preview_frame(self, *, show_route: bool = True) -> Any:
        """Render the 3D scene at the centre of the start distribution.

        Must be called from the thread that owns the display.  Raises
        :class:`ControllerError` when MuJoCo or a renderer is unavailable.
        """

        state = self.preview_state()
        built = self.built
        try:
            from trailer_docking.visualization.render3d import render_scene

            return render_scene(
                built.vehicle,
                built.dock,
                state.tractor_pose,
                state.trailer_pose.yaw,
                standoff=max(self.scenario.start.standoff),
                route=self.preview_route() if show_route else None,
            )
        except Exception as error:
            raise ControllerError(f"3D preview unavailable: {error}") from error

    @property
    def spectator_frame(self) -> Any:
        """Latest rendered 3D image of the active (or most recent) activity."""

        with self._lock:
            source = self._spectator_source
            session = self._session
            if source == "test":
                return self._test_frame
        if source != "training" or session is None:
            return None
        try:
            return getattr(session, "spectator_frame", None)
        except Exception:
            return None

    # ------------------------------------------------------------- shared

    @property
    def spectator_state(self) -> VehicleState | None:
        """Latest vehicle state of the active (or most recent) activity."""

        with self._lock:
            source = self._spectator_source
            session = self._session
            if source == "test":
                # Test states were already observed as they arrived.
                return self._test_state
        if source != "training" or session is None:
            return None
        try:
            state: VehicleState | None = session.spectator_state
            episodes = session.progress.episodes
            # The session reports the environment's own reference path, which
            # is exact; replanning from a polled state is only a fallback.
            exact_route = getattr(session, "spectator_route", None)
        except Exception:
            return None
        with self._lock:
            if self._session is session and self._spectator_source == "training":
                if exact_route is not None:
                    self._route = tuple(exact_route)
                    self._route_state, self._route_marker = state, episodes
                else:
                    self._observe(state, episodes)
        return state

    @property
    def spectator_route(self) -> tuple[Point2D, ...] | None:
        """Planned docking-point route of the episode being shown, if any.

        The route is replanned whenever the spectator feed starts a new
        episode.  A test delivers every state, so its episodes are detected
        exactly; training is only polled, so the episode counter is used as
        well as a drop in simulation time.
        """

        self.spectator_state  # noqa: B018 - polls the training feed
        with self._lock:
            return self._route

    def _observe(self, state: VehicleState | None, marker: Any = None) -> None:
        """Replan the route if *state* begins a new episode (lock held)."""

        last = self._route_state
        if state is None or (state is last and marker == self._route_marker):
            return
        new_episode = (
            last is None
            or state.simulation_time < last.simulation_time
            or marker != self._route_marker
        )
        self._route_state, self._route_marker = state, marker
        if not new_episode:
            return
        try:
            self._route = self._plan(state)
        except Exception:
            # A planner failure only hides the route; the feed carries on.
            self._route = None

    def _plan(self, state: VehicleState) -> tuple[Point2D, ...]:
        """Route from *state* with the scenario's planner (lock held)."""

        geometry = self._built.vehicle.geometry
        dock = self._built.dock.geometry
        if self._route_planner is not None:
            return tuple(self._route_planner(geometry, dock, state).points)
        from trailer_docking.paths.planners import plan_reference

        return tuple(
            plan_reference(self._scenario.planner, geometry, dock, state).points
        )

    def preview_route(self) -> tuple[Point2D, ...]:
        """Route the scenario's planner draws from the start-pose preview."""

        state = self.preview_state()
        with self._lock:
            try:
                return self._plan(state)
            except Exception:
                return ()

    def _forget_test(self) -> None:
        """Drop the last test's results; they describe another scenario
        once the scenario is replaced (lock held)."""

        self._test_status = "idle"
        self._test_results = ()
        self._test_error = None
        self._test_policy = ""
        self._test_stop_requested = False

    def _forget_spectator(self) -> None:
        """Drop the shown state and its route (lock held)."""

        self._spectator_source = None
        self._test_state = None
        self._test_frame = None
        self._route = None
        self._route_state = None
        self._route_marker = None

    @property
    def busy(self) -> bool:
        """True while training or a docking test occupies the controller."""

        return self.training_active or self.test_status == "running"

    def training_status_lines(self) -> tuple[str, ...]:
        progress = self.training_progress
        if progress is None:
            return ("Training: idle",)
        lines = [f"Training: {progress.status}"]
        runs = getattr(progress, "runs", 1)
        if runs > 1:
            lines.append(f"Run: {getattr(progress, 'run', 1)} of {runs}")
        phase = getattr(progress, "phase", "")
        if phase:
            lines.append(f"Now: {phase}")
        lines += [
            f"Timesteps: {progress.timesteps:,} / {progress.total_timesteps:,}",
            f"Episodes: {progress.episodes}",
        ]
        rate = progress.recent_success_rate
        lines.append(
            "Recent success rate: "
            + ("n/a" if rate is None else f"{rate * 100.0:.1f} %")
        )
        mean_return = progress.recent_mean_return
        lines.append(
            "Recent mean return: "
            + ("n/a" if mean_return is None else f"{mean_return:.2f}")
        )
        lines.append(f"Last outcome: {progress.last_outcome or 'n/a'}")
        lines.append(f"Experiment: {progress.experiment_id or 'n/a'}")
        lines.append(f"Checkpoint: {progress.checkpoint_path or 'none yet'}")
        lines.extend(self._run_summary_lines())
        bundle = getattr(progress, "bundle_path", None)
        if bundle:
            lines.append(f"Result bundle: {bundle}")
        if progress.error:
            lines.append(f"Error: {progress.error}")
        return tuple(lines)

    def _run_summary_lines(self) -> list[str]:
        """One line per finished campaign run with its test success rates."""

        session = self._current_session()
        try:
            runs = tuple(getattr(session, "run_summaries", ()))
        except Exception:
            return []
        lines = []
        for run in runs:
            tests = ", ".join(
                f"{item['backend']} {item['success_rate'] * 100.0:.0f} %"
                f" of {item['episodes']}"
                for item in run.get("evaluation", ())
            )
            lines.append(
                f"Run {run['run']} (seed {run['seed']}): {tests or 'not tested'}"
            )
        return lines

    def test_result_lines(self) -> tuple[str, ...]:
        """One line per finished test episode."""

        return tuple(
            f"{index}: {result.outcome}"
            f"  steps={result.steps}  reward={result.total_reward:.1f}"
            f"  err=({result.final_longitudinal_error:+.3f},"
            f" {result.final_lateral_error:+.3f}) m"
            f"  yaw={result.final_yaw_error:+.3f} rad"
            for index, result in enumerate(self.test_results, start=1)
        )

    def test_status_lines(self) -> tuple[str, ...]:
        with self._lock:
            status = self._test_status
            error = self._test_error
            policy = self._test_policy
            results = self._test_results
            stopped = self._test_stop_requested
        if status == "idle":
            return ("Test: idle",)
        label = "stopped by user" if status == "finished" and stopped else status
        lines = [f"Test: {label}", f"Policy: {policy}"]
        if error:
            lines.append(f"Error: {error}")
        if results:
            successes = sum(1 for result in results if result.success)
            lines.append(f"Successes: {successes} / {len(results)}")
        lines.extend(f"Episode {line}" for line in self.test_result_lines())
        return tuple(lines)

    def status_lines(self) -> tuple[str, ...]:
        """Short human-readable lines for the status panel."""

        return self.training_status_lines() + self.test_status_lines()

    def shutdown(self) -> None:
        """Stop training and any test, then wait briefly for both."""

        self.stop_training()
        self.stop_test()
        session = self._current_session()
        if session is not None:
            # Shutdown must never raise; the worker threads are daemons.
            with suppress(Exception):
                session.join(timeout=_JOIN_TIMEOUT)
        with self._lock:
            thread = self._test_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=_JOIN_TIMEOUT)

    # ------------------------------------------------------------ helpers

    def _current_session(self) -> Any:
        with self._lock:
            return self._session

    @staticmethod
    def _session_active(session: Any) -> bool:
        try:
            return session.progress.status not in _TERMINAL_TRAINING
        except Exception:
            return False

    def _require_idle(self, action: str, error_type: type[ControllerError]) -> None:
        """Raise *error_type* if training or a test is running (lock held)."""

        if self._session is not None and self._session_active(self._session):
            raise error_type(f"cannot {action} while training is running")
        if self._test_status == "running":
            raise error_type(f"cannot {action} while a docking test is running")
