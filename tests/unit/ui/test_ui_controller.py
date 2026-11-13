"""Headless tests of the control-UI logic, using fakes for training."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from trailer_docking.core.state import Point2D, Pose2D, VehicleState
from trailer_docking.evaluation.scenarios import (
    PHYSICS_ONLY_FIELDS,
    Scenario,
    ScenarioOverrides,
    StartDistribution,
)
from trailer_docking.ui.controller import (
    GROUPS,
    PARAMETERS,
    AppController,
    BusyError,
    ControllerError,
    ParameterError,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _repository_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    # Scenario base-specification paths are relative to the repository root.
    monkeypatch.chdir(REPOSITORY_ROOT)


def wait_until(condition: Callable[[], bool], timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition was not met in time")
        time.sleep(0.005)


def make_state(simulation_time: float = 0.0, x: float = 8.0) -> VehicleState:
    return VehicleState(
        tractor_pose=Pose2D(x, 0.0, 0.0),
        trailer_pose=Pose2D(x - 2.0, 0.0, 0.0),
        hitch_position=Point2D(x - 0.5, 0.0),
        steering_angle=0.0,
        steering_rate=0.0,
        longitudinal_speed=0.0,
        tractor_yaw_rate=0.0,
        trailer_yaw_rate=0.0,
        simulation_time=simulation_time,
    )


@dataclass(frozen=True)
class FakeProgress:
    status: str = "idle"
    timesteps: int = 0
    total_timesteps: int = 1000
    episodes: int = 0
    recent_success_rate: float | None = None
    recent_mean_return: float | None = None
    last_outcome: str | None = None
    error: str | None = None
    experiment_id: str | None = None
    checkpoint_path: Path | None = None


class FakeSession:
    """Stands in for ``TrainingSession`` without threads or learning."""

    def __init__(self, scenario: Scenario, config: Any, output_root: Path) -> None:
        self.scenario = scenario
        self.config = config
        self.output_root = output_root
        self.progress = FakeProgress()
        self.spectator_state: VehicleState | None = None
        self.episode_history: tuple[Any, ...] = ()
        self.stop_calls = 0
        self.join_calls = 0

    def start(self) -> None:
        if self.progress.status != "idle":
            raise RuntimeError("already started")
        self.progress = replace(self.progress, status="running")

    def stop(self) -> None:
        self.stop_calls += 1
        if self.progress.status == "running":
            self.progress = replace(self.progress, status="stopped")

    def join(self, timeout: float | None = None) -> None:
        self.join_calls += 1


class SessionFactory:
    def __init__(self) -> None:
        self.sessions: list[FakeSession] = []

    def __call__(
        self, scenario: Scenario, config: Any, *, output_root: Path
    ) -> FakeSession:
        session = FakeSession(scenario, config, output_root)
        self.sessions.append(session)
        return session


def make_result(outcome: str = "success", success: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        outcome=outcome,
        success=success,
        steps=42,
        total_reward=12.5,
        final_longitudinal_error=0.01,
        final_lateral_error=-0.02,
        final_yaw_error=0.003,
        max_abs_articulation=0.2,
    )


class FakeRunner:
    """A ``run_docking_test`` double that can block until released."""

    def __init__(self, *, block: bool = False, fail: bool = False) -> None:
        self.block = block
        self.fail = fail
        self.release = threading.Event()
        self.calls: list[dict[str, Any]] = []
        self.states = [make_state(0.0), make_state(0.1)]

    def __call__(
        self,
        scenario: Scenario,
        policy: Any,
        *,
        episodes: int = 1,
        seed: int = 0,
        on_state: Callable[[VehicleState], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        **_: Any,
    ) -> list[SimpleNamespace]:
        self.calls.append(
            {"scenario": scenario, "policy": policy, "episodes": episodes, "seed": seed}
        )
        assert on_state is not None and should_stop is not None
        for state in self.states:
            on_state(state)
        if self.fail:
            raise RuntimeError("simulator exploded")
        if self.block:
            deadline = time.monotonic() + 5.0
            while not (should_stop() or self.release.is_set()):
                assert time.monotonic() < deadline
                time.sleep(0.005)
            if should_stop():
                return [make_result("stopped", False)]
        return [make_result() for _ in range(episodes)]


class FakePlanner:
    def __init__(self) -> None:
        self.starts: list[VehicleState] = []
        self.fail = False

    def __call__(self, vehicle: Any, dock: Any, start: VehicleState) -> Any:
        self.starts.append(start)
        if self.fail:
            raise RuntimeError("no route")
        origin = start.trailer_pose.position
        return SimpleNamespace(points=(origin, dock.target_position), length=1.0)


def make_controller(**options: Any) -> AppController:
    options.setdefault("session_factory", SessionFactory())
    options.setdefault("test_runner", FakeRunner())
    options.setdefault("policy_loader", lambda path: ("loaded", Path(path)))
    options.setdefault("baseline_policy", "baseline")
    options.setdefault("route_planner", FakePlanner())
    return AppController(**options)


# ------------------------------------------------------------ parameters


def test_parameters_cover_every_scenario_field() -> None:
    keys = [spec.key for spec in PARAMETERS]
    override_keys = {item.name for item in fields(ScenarioOverrides)}
    start_keys = {item.name for item in fields(StartDistribution)}
    assert len(keys) == len(set(keys))
    random_keys = {"random_trailer_hitch_to_axle", "observation_noise_std"}
    extra = {"planner", "obstacles"}
    assert set(keys) == override_keys | start_keys | random_keys | extra
    for spec in PARAMETERS:
        assert spec.group in GROUPS
        assert spec.physics_only == (spec.key in PHYSICS_ONLY_FIELDS)
        assert (spec.group == "Physics (3D only)") == spec.physics_only
        assert (spec.kind == "range") == (spec.key in start_keys)
        assert (spec.group == "Start pose") == (spec.key in start_keys)
        assert (spec.group == "Randomization") == (spec.key in random_keys)
    variant = next(spec for spec in PARAMETERS if spec.key == "trailer_variant")
    assert variant.kind == "choice"
    assert variant.choices == ("", "short", "nominal", "long")


def test_parameter_values_round_trip() -> None:
    controller = make_controller()
    values = controller.parameter_values()
    assert values["dock_interior_width"] == ""
    assert values["trailer_variant"] == ""
    assert values["standoff"] == "6.0, 9.0"
    assert values["lateral"] == "-0.4, 0.4"
    before = controller.scenario
    controller.apply_parameters(values)
    assert controller.scenario == before
    assert controller.parameter_values() == values


def test_apply_updates_built_scenario_and_blank_restores_base() -> None:
    controller = make_controller()
    base = controller.built
    controller.apply_parameters(
        {
            "dock_interior_width": "2.0",
            "trailer_variant": "short",
            "tractor_chassis_mass": "900",
            "ground_friction": "0.6",
            "standoff": " 2 , 6.5 ",
        }
    )
    built = controller.built
    assert built.dock.geometry.interior_width == 2.0
    assert built.vehicle.trailer.variant == "short"
    assert built.vehicle.mujoco.tractor_chassis_mass == 900.0
    assert built.vehicle.mujoco.contact_friction[0] == 0.6
    assert controller.scenario.start.standoff == (2.0, 6.5)
    # Keys that were not passed keep their value.
    assert controller.scenario.overrides.trailer_body_mass is None
    values = controller.parameter_values()
    assert values["dock_interior_width"] == "2.0"
    assert values["standoff"] == "2.0, 6.5"
    controller.apply_parameters(values)
    assert controller.built == built

    controller.apply_parameters(dict.fromkeys(values, ""))
    assert controller.scenario.overrides == ScenarioOverrides()
    assert controller.scenario.start == StartDistribution()
    assert controller.built == base


@pytest.mark.parametrize(
    "values",
    [
        {"dock_interior_width": "abc"},
        {"dock_interior_width": "-1.5"},
        {"dock_interior_width": "0.01"},
        {"dock_interior_depth": "nan"},
        {"trailer_variant": "enormous"},
        {"tractor_chassis_mass": "-5"},
        {"ground_friction": "0"},
        {"standoff": "1"},
        {"standoff": "2, 1"},
        {"standoff": "-1, 1"},
        {"lateral": "1, 2, 3"},
        {"lateral": "a, b"},
        {"not_a_parameter": "1"},
    ],
)
def test_invalid_parameters_leave_scenario_unchanged(values: dict[str, str]) -> None:
    controller = make_controller()
    controller.apply_parameters({"dock_interior_width": "2.0"})
    scenario, built = controller.scenario, controller.built
    with pytest.raises(ParameterError):
        controller.apply_parameters({"trailer_variant": "long", **values})
    assert controller.scenario is scenario
    assert controller.built is built


def test_a_dock_the_trailer_cannot_enter_is_rejected() -> None:
    # Docking success needs the whole trailer inside the bay, so a bay
    # narrower than the trailer could never be solved.
    controller = make_controller()
    trailer = controller.built.vehicle.trailer
    corners = trailer.footprint(Pose2D(0.0, 0.0, 0.0))
    trailer_width = max(p.y for p in corners) - min(p.y for p in corners)
    before = controller.scenario
    with pytest.raises(ParameterError, match="cannot dock completely inside"):
        controller.apply_parameters({"dock_interior_width": repr(trailer_width * 0.8)})
    assert controller.scenario == before
    controller.apply_parameters({"dock_interior_width": repr(trailer_width * 1.2)})
    assert controller.built.dock.geometry.interior_width == pytest.approx(
        trailer_width * 1.2
    )


def test_error_types() -> None:
    assert issubclass(ParameterError, ValueError)
    assert issubclass(ParameterError, ControllerError)
    assert issubclass(BusyError, RuntimeError)


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    controller = make_controller()
    controller.apply_parameters({"dock_interior_width": "2.25", "lateral": "-1, 1"})
    saved = controller.save_scenario(tmp_path / "nested" / "scenario.yaml")
    assert saved.is_file()

    other = make_controller()
    other.load_scenario(saved)
    assert other.scenario == controller.scenario
    assert other.built.dock.geometry.interior_width == 2.25
    assert other.parameter_values() == controller.parameter_values()


def test_load_rejects_bad_files_and_keeps_scenario(tmp_path: Path) -> None:
    controller = make_controller()
    scenario = controller.scenario
    with pytest.raises(ParameterError):
        controller.load_scenario(tmp_path / "missing.yaml")
    for name, override in (
        ("negative", "dock_interior_width: -1.0"),
        ("wrong_type", "trailer_body_mass: heavy"),
        ("too_narrow", "dock_interior_width: 0.01"),
    ):
        bad = tmp_path / f"{name}.yaml"
        bad.write_text(f"overrides: {{{override}}}", encoding="utf-8")
        with pytest.raises(ParameterError):
            controller.load_scenario(bad)
    assert controller.scenario is scenario


def test_preview_state_sits_outside_the_bay() -> None:
    controller = make_controller()
    state = controller.preview_state()
    bay = controller.built.dock.geometry
    rear = controller.built.vehicle.trailer.rear_face(state.trailer_pose)
    assert rear.x == pytest.approx(bay.interior_depth + 7.5)
    assert rear.y == pytest.approx(0.0, abs=1e-9)


# -------------------------------------------------------------- training


def test_training_lifecycle_and_refusals() -> None:
    factory = SessionFactory()
    controller = make_controller(session_factory=factory, output_root="runs")
    assert controller.training_progress is None
    assert not controller.busy
    assert controller.status_lines()[0] == "Training: idle"

    controller.start_training("config")
    (session,) = factory.sessions
    assert session.config == "config"
    assert session.scenario is controller.scenario
    assert session.output_root == Path("runs")
    assert controller.busy and controller.training_active
    assert controller.training_progress.status == "running"

    with pytest.raises(BusyError):
        controller.start_training()
    with pytest.raises(BusyError):
        controller.start_test()
    with pytest.raises(ParameterError):
        controller.apply_parameters({"dock_interior_width": "2.0"})
    with pytest.raises(ParameterError):
        controller.load_scenario("anything.yaml")
    assert len(factory.sessions) == 1
    assert controller.scenario.overrides.dock_interior_width is None

    session.spectator_state = make_state(1.0)
    assert controller.spectator_state is session.spectator_state

    controller.stop_training()
    assert session.stop_calls == 1
    assert not controller.busy
    assert controller.training_progress.status == "stopped"
    # A finished session no longer blocks editing or a new run.
    controller.apply_parameters({"dock_interior_width": "2.0"})
    controller.start_training()
    assert len(factory.sessions) == 2
    assert factory.sessions[1].scenario.overrides.dock_interior_width == 2.0


def test_stopping_session_still_counts_as_busy() -> None:
    factory = SessionFactory()
    controller = make_controller(session_factory=factory)
    controller.start_training()
    factory.sessions[0].progress = FakeProgress(status="stopping")
    assert controller.busy
    factory.sessions[0].progress = FakeProgress(status="failed", error="boom")
    assert not controller.busy
    assert "Error: boom" in controller.status_lines()


def test_start_training_failure_is_reported() -> None:
    def broken_factory(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("no torch")

    controller = make_controller(session_factory=broken_factory)
    with pytest.raises(ControllerError, match="no torch"):
        controller.start_training()
    assert not controller.busy
    assert controller.training_progress is None


def test_status_lines_describe_training() -> None:
    factory = SessionFactory()
    controller = make_controller(session_factory=factory)
    controller.start_training()
    factory.sessions[0].progress = FakeProgress(
        status="running",
        timesteps=2048,
        total_timesteps=10_000,
        episodes=7,
        recent_success_rate=0.25,
        recent_mean_return=-3.5,
        last_outcome="collision",
        experiment_id="2026-11-11_ppo-docking",
        checkpoint_path=Path("checkpoints/latest.zip"),
    )
    text = "\n".join(controller.status_lines())
    assert "Training: running" in text
    assert "Timesteps: 2,048 / 10,000" in text
    assert "Episodes: 7" in text
    assert "Recent success rate: 25.0 %" in text
    assert "Last outcome: collision" in text
    assert "Experiment: 2026-11-11_ppo-docking" in text
    assert "latest.zip" in text
    assert "Test: idle" in text


# ------------------------------------------------------------------ test


def test_docking_test_fills_results_and_spectator() -> None:
    runner = FakeRunner()
    controller = make_controller(test_runner=runner)
    assert controller.test_status == "idle"
    assert controller.spectator_state is None

    controller.start_test(episodes=3, seed=5)
    wait_until(lambda: controller.test_status == "finished")
    assert not controller.busy
    assert len(controller.test_results) == 3
    assert controller.test_error is None
    assert controller.spectator_state is runner.states[-1]
    (call,) = runner.calls
    assert call["episodes"] == 3 and call["seed"] == 5
    assert call["scenario"] is controller.scenario
    assert call["policy"] == "baseline"

    text = "\n".join(controller.status_lines())
    assert "Test: finished" in text
    assert "Successes: 3 / 3" in text
    assert "Episode 1: success" in text
    assert "Episode 3: success" in text
    assert len(controller.test_result_lines()) == 3


def test_docking_test_failure_is_captured() -> None:
    controller = make_controller(test_runner=FakeRunner(fail=True))
    controller.start_test()
    wait_until(lambda: controller.test_status == "failed")
    assert "simulator exploded" in (controller.test_error or "")
    assert controller.test_results == ()
    assert not controller.busy
    assert any("simulator exploded" in line for line in controller.status_lines())
    # The controller is usable again afterwards.
    controller.apply_parameters({"dock_interior_width": "2.0"})


def test_policy_loader_failure_fails_the_test(tmp_path: Path) -> None:
    def loader(path: Path) -> Any:
        raise FileNotFoundError(f"no policy at {path}")

    controller = make_controller(policy_loader=loader)
    controller.start_test(policy_path=str(tmp_path / "missing.zip"))
    wait_until(lambda: controller.test_status == "failed")
    assert "no policy at" in (controller.test_error or "")


def test_stop_test_ends_a_running_test() -> None:
    runner = FakeRunner(block=True)
    factory = SessionFactory()
    controller = make_controller(test_runner=runner, session_factory=factory)
    controller.start_test()
    wait_until(lambda: controller.spectator_state is not None)
    assert controller.busy and controller.test_status == "running"
    with pytest.raises(BusyError):
        controller.start_test()
    with pytest.raises(BusyError):
        controller.start_training()
    with pytest.raises(ParameterError):
        controller.apply_parameters({"dock_interior_width": "2.0"})
    assert factory.sessions == []

    controller.stop_test()
    wait_until(lambda: controller.test_status == "finished")
    assert [result.outcome for result in controller.test_results] == ["stopped"]
    assert "Test: stopped by user" in controller.status_lines()
    assert not controller.busy


def test_policy_selection(tmp_path: Path) -> None:
    factory = SessionFactory()
    runner = FakeRunner()
    controller = make_controller(session_factory=factory, test_runner=runner)

    def run(**options: Any) -> Any:
        controller.start_test(**options)
        wait_until(lambda: controller.test_status != "running")
        assert controller.test_status == "finished"
        return runner.calls[-1]["policy"]

    # No training yet: the untrained baseline.
    assert controller.latest_checkpoint is None
    assert run() == "baseline"
    assert "baseline" in controller.test_policy

    # A session without a checkpoint still means the baseline.
    controller.start_training()
    session = factory.sessions[0]
    session.progress = FakeProgress(status="finished")
    assert run() == "baseline"

    # With a checkpoint, the latest checkpoint is loaded.
    checkpoint = tmp_path / "latest.zip"
    session.progress = FakeProgress(status="finished", checkpoint_path=checkpoint)
    assert controller.latest_checkpoint == checkpoint
    assert run() == ("loaded", checkpoint)
    assert str(checkpoint) in controller.test_policy

    # An explicit path wins, and the baseline can be forced.
    explicit = tmp_path / "other.zip"
    assert run(policy_path=str(explicit)) == ("loaded", explicit)
    assert run(baseline=True) == "baseline"
    with pytest.raises(ParameterError):
        controller.start_test(policy_path=str(explicit), baseline=True)


@pytest.mark.parametrize("options", [{"episodes": 0}, {"episodes": 1.5}, {"seed": "a"}])
def test_start_test_validates_arguments(options: dict[str, Any]) -> None:
    runner = FakeRunner()
    controller = make_controller(test_runner=runner)
    with pytest.raises(ParameterError):
        controller.start_test(**options)
    assert controller.test_status == "idle"
    assert runner.calls == []


# ----------------------------------------------------------------- route


def test_route_follows_test_episodes() -> None:
    planner = FakePlanner()
    runner = FakeRunner()
    first, second, third = make_state(0.0), make_state(0.1), make_state(0.0, x=9.0)
    runner.states = [first, second, third]
    controller = make_controller(test_runner=runner, route_planner=planner)
    assert controller.spectator_route is None

    controller.start_test()
    wait_until(lambda: controller.test_status == "finished")
    # Planned at the first state and again when simulation time dropped.
    assert planner.starts == [first, third]
    route = controller.spectator_route
    assert route is not None
    assert route[0] == third.trailer_pose.position
    assert route[-1] == controller.built.dock.geometry.target_position
    assert controller.spectator_route is route

    controller.apply_parameters({"dock_interior_width": "2.0"})
    assert controller.spectator_route is None
    assert controller.spectator_state is None


def test_route_follows_polled_training_feed() -> None:
    planner = FakePlanner()
    factory = SessionFactory()
    controller = make_controller(session_factory=factory, route_planner=planner)
    controller.start_training()
    session = factory.sessions[0]
    assert controller.spectator_route is None

    session.spectator_state = make_state(0.5)
    first = controller.spectator_route
    assert first is not None and len(planner.starts) == 1
    # Polling the same state again does not replan.
    assert controller.spectator_state is session.spectator_state
    assert controller.spectator_route is first

    session.spectator_state = make_state(0.6)
    assert controller.spectator_route is first
    session.spectator_state = make_state(0.1, x=9.5)
    second = controller.spectator_route
    assert second is not first and len(planner.starts) == 2
    # A finished episode missed between polls still triggers a replan.
    session.progress = replace(session.progress, episodes=3)
    session.spectator_state = make_state(4.0)
    assert controller.spectator_route is not second
    assert len(planner.starts) == 3


def test_failing_planner_hides_route_but_keeps_the_feed() -> None:
    planner = FakePlanner()
    planner.fail = True
    runner = FakeRunner()
    controller = make_controller(test_runner=runner, route_planner=planner)
    controller.start_test()
    wait_until(lambda: controller.test_status == "finished")
    assert planner.starts
    assert controller.spectator_route is None
    assert controller.spectator_state is runner.states[-1]
    assert controller.test_error is None

    # A later working plan replaces an earlier one that failed.
    planner.fail = False
    controller.start_test()
    wait_until(lambda: controller.test_status == "finished")
    assert controller.spectator_route is not None


# -------------------------------------------------------------- shutdown


def test_shutdown_is_idempotent_and_stops_everything() -> None:
    factory = SessionFactory()
    runner = FakeRunner(block=True)
    controller = make_controller(session_factory=factory, test_runner=runner)
    controller.shutdown()  # nothing running

    controller.start_test()
    wait_until(lambda: controller.test_status == "running")
    controller.shutdown()
    assert controller.test_status == "finished"

    controller.start_training()
    controller.shutdown()
    controller.shutdown()
    session = factory.sessions[0]
    assert session.stop_calls >= 1 and session.join_calls >= 1
    assert not controller.busy


def test_controller_import_is_headless() -> None:
    code = (
        "import sys; import trailer_docking.ui, trailer_docking.ui.controller;"
        "bad = [n for n in ('tkinter', 'matplotlib', 'trailer_docking.training',"
        " 'mujoco') if n in sys.modules]; assert not bad, bad"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPOSITORY_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPOSITORY_ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
