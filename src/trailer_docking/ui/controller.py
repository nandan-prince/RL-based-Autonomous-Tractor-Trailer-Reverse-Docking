"""Headless logic behind the control UI.

:class:`AppController` owns the editable :class:`Scenario`, one background
training session, and one background docking test.  It imports neither
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

GROUPS = ("Dock", "Vehicle", "Physics (3D only)", "Start pose")
PHYSICS_GROUP = "Physics (3D only)"
PHYSICS_GROUP_NOTE = "3D physics only \N{EM DASH} no effect in the 2D model"

# Training statuses after which a session no longer occupies the controller.
_TERMINAL_TRAINING = frozenset({"finished", "stopped", "failed"})
_JOIN_TIMEOUT = 2.0


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
        if self.kind not in {"float", "choice", "range"}:
            raise ValueError(f"unknown parameter kind: {self.kind}")


PARAMETERS: tuple[ParameterSpec, ...] = (
    ParameterSpec(
        "dock_interior_width", "Bay interior width", "Dock", "float", unit="m"
    ),
    ParameterSpec(
        "dock_interior_depth", "Bay interior depth", "Dock", "float", unit="m"
    ),
    ParameterSpec(
        "trailer_variant",
        "Trailer variant",
        "Vehicle",
        "choice",
        choices=("", "short", "nominal", "long"),
    ),
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
)

_OVERRIDE_KEYS = frozenset(item.name for item in fields(ScenarioOverrides))
_START_KEYS = frozenset(item.name for item in fields(StartDistribution))
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
    if spec.kind == "range":
        parts = [part.strip() for part in text.split(",")]
        if len(parts) != 2 or not all(parts):
            raise ParameterError(
                f"{spec.label}: enter a range as 'minimum, maximum', got '{text}'"
            )
        return (_parse_number(spec, parts[0]), _parse_number(spec, parts[1]))
    return _parse_number(spec, text)


def _default_session_factory(
    scenario: Scenario, config: Any, *, output_root: Path
) -> Any:
    from trailer_docking.training.session import TrainingSession

    return TrainingSession(scenario, config, output_root=output_root)


def _default_test_runner(scenario: Scenario, policy: Any, **options: Any) -> Any:
    from trailer_docking.training.evaluate import run_docking_test

    return run_docking_test(scenario, policy, **options)


def _default_policy_loader(path: Path) -> Any:
    from trailer_docking.training.evaluate import load_policy

    return load_policy(path)


def _default_route_planner(vehicle: Any, dock: Any, start: VehicleState) -> Any:
    from trailer_docking.paths.generators import plan_docking_route

    return plan_docking_route(vehicle, dock, start)


def _default_baseline_policy() -> Any:
    from trailer_docking.training.evaluate import straight_reverse_policy

    return straight_reverse_policy


class AppController:
    """Scenario editing plus one training session and one docking test.

    ``session_factory(scenario, config, output_root=...)`` must return an
    object with the ``TrainingSession`` interface, ``test_runner`` has the
    signature of ``run_docking_test``, and ``policy_loader(path)`` returns a
    policy.  All three default to the real training package, imported only
    when first used.  ``baseline_policy`` replaces the untrained
    straight-reverse policy (also imported lazily when omitted), and
    ``route_planner(vehicle_geometry, dock_geometry, start_state)`` replaces
    ``plan_docking_route``; it must return an object with ``.points``.
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
        self._route_planner = route_planner or _default_route_planner

        self._session: Any = None
        self._test_thread: threading.Thread | None = None
        self._test_stop = threading.Event()
        self._test_status = "idle"
        self._test_results: tuple[Any, ...] = ()
        self._test_error: str | None = None
        self._test_policy = ""
        self._test_state: VehicleState | None = None
        self._test_stop_requested = False
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

        A blank override keeps the base specification and a blank range
        restores the default range.  Keys that are absent keep their current
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
            defaults = StartDistribution()
            for key, text in values.items():
                spec = _SPEC_BY_KEY[key]
                parsed = _parse_value(spec, str(text))
                if key in _START_KEYS:
                    start[key] = getattr(defaults, key) if parsed is None else parsed
                else:
                    overrides[key] = parsed
            try:
                candidate = replace(
                    self._scenario,
                    overrides=replace(self._scenario.overrides, **overrides),
                    start=replace(self._scenario.start, **start),
                )
                built = candidate.build()
            except ValueError as error:
                raise ParameterError(str(error)) from error
            self._scenario, self._built = candidate, built
            self._forget_spectator()

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

    def start_training(self, config: Any = None) -> None:
        """Start a background training session on the current scenario."""

        with self._lock:
            self._require_idle("start training", BusyError)
            try:
                session = self._session_factory(
                    self._scenario, config, output_root=self._output_root
                )
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
    ) -> None:
        """Run a docking test in a daemon thread.

        With ``policy_path`` omitted the latest training checkpoint is used
        when there is one and the untrained straight-reverse policy
        otherwise.  ``baseline=True`` forces the untrained policy.
        """

        if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1:
            raise ParameterError("episodes must be a positive whole number")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ParameterError("seed must be a whole number")
        if baseline and policy_path is not None:
            raise ParameterError("choose either a policy file or the baseline")
        with self._lock:
            self._require_idle("start a test", BusyError)
            source: Path | None
            if baseline:
                source = None
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
            self._test_policy = (
                "untrained straight-reverse baseline" if source is None else str(source)
            )
            self._spectator_source = "test"
            thread = threading.Thread(
                target=self._run_test,
                args=(self._scenario, source, episodes, seed, stop),
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
        source: Path | None,
        episodes: int,
        seed: int,
        stop: threading.Event,
    ) -> None:
        def on_state(state: VehicleState) -> None:
            with self._lock:
                if self._test_stop is stop:
                    self._test_state = state
                    self._observe(state)

        status, message, results = "finished", None, ()
        try:
            if source is not None:
                policy = self._policy_loader(source)
            elif self._baseline_policy is not None:
                policy = self._baseline_policy
            else:
                policy = _default_baseline_policy()
            results = tuple(
                self._test_runner(
                    scenario,
                    policy,
                    episodes=episodes,
                    seed=seed,
                    on_state=on_state,
                    should_stop=stop.is_set,
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
            state = session.spectator_state
            episodes = session.progress.episodes
        except Exception:
            return None
        with self._lock:
            if self._session is session and self._spectator_source == "training":
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
            route = self._route_planner(
                self._built.vehicle.geometry, self._built.dock.geometry, state
            )
            self._route = tuple(route.points)
        except Exception:
            # A planner failure only hides the route; the feed carries on.
            self._route = None

    def _forget_spectator(self) -> None:
        """Drop the shown state and its route (lock held)."""

        self._spectator_source = None
        self._test_state = None
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
        lines = [
            f"Training: {progress.status}",
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
        if progress.error:
            lines.append(f"Error: {progress.error}")
        return tuple(lines)

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
