"""Control-UI logic for simulator choice, campaigns, bundles, and 3D display."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from trailer_docking.evaluation.scenarios import Scenario, ScenarioOverrides
from trailer_docking.training.bundle import export_bundle
from trailer_docking.training.config import TrainingConfig
from trailer_docking.ui.controller import (
    AppController,
    ControllerError,
    ParameterError,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _repository_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(REPOSITORY_ROOT)


def wait_until(condition: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition was not met in time")
        time.sleep(0.005)


class Runner:
    """Records how ``run_docking_test`` would have been called."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, scenario: Scenario, policy: Any, **options: Any) -> list[Any]:
        self.calls.append({"scenario": scenario, "policy": policy, **options})
        if "on_frame" in options:
            options["on_frame"](np.full((2, 2, 3), 7, np.uint8))
        return [
            SimpleNamespace(
                outcome="success",
                success=True,
                steps=3,
                total_reward=1.0,
                final_longitudinal_error=0.0,
                final_lateral_error=0.0,
                final_yaw_error=0.0,
                max_abs_articulation=0.0,
            )
        ]


class Campaign:
    """A campaign double that is finished as soon as it starts."""

    def __init__(self, scenario: Scenario, config: Any, **options: Any) -> None:
        self.config = config
        self.options = options
        self.spectator_state = None
        self.spectator_frame = np.zeros((2, 2, 3), np.uint8)
        self.episode_history: tuple[Any, ...] = ()
        self.bundle: Path | None = None
        self.status = "idle"
        self.run_summaries = (
            {
                "run": 1,
                "seed": 4,
                "evaluation": [
                    {"backend": "2d", "success_rate": 0.5, "episodes": 2},
                    {"backend": "3d", "success_rate": 0.0, "episodes": 2},
                ],
            },
            {"run": 2, "seed": 5, "evaluation": []},
        )

    def start(self) -> None:
        self.status = "finished"

    def stop(self) -> None:
        return None

    def join(self, timeout: float | None = None) -> None:
        return None

    @property
    def progress(self) -> Any:
        return SimpleNamespace(
            status=self.status,
            timesteps=10,
            total_timesteps=20,
            episodes=1,
            recent_success_rate=None,
            recent_mean_return=None,
            last_outcome=None,
            error=None,
            experiment_id="campaign-1",
            checkpoint_path=None,
            run=2,
            runs=2,
            phase="testing run 2 of 2 in 3d (2 episodes)",
            bundle_path=self.bundle,
        )


def make(tmp_path: Path, **options: Any) -> tuple[AppController, Runner, list[Any]]:
    runner = Runner()
    campaigns: list[Any] = []

    def factory(scenario: Scenario, config: Any, **kwargs: Any) -> Campaign:
        campaign = Campaign(scenario, config, **kwargs)
        campaigns.append(campaign)
        return campaign

    options.setdefault("session_factory", factory)
    options.setdefault("test_runner", runner)
    options.setdefault("baseline_policy", "baseline")
    return AppController(output_root=tmp_path, **options), runner, campaigns


def run_test(controller: AppController, **options: Any) -> None:
    controller.start_test(**options)
    wait_until(lambda: controller.test_status != "running")


# ------------------------------------------------------------- parameters


def test_truck_dimensions_limits_and_randomization_are_editable(
    tmp_path: Path,
) -> None:
    controller, _, _ = make(tmp_path)
    values = controller.parameter_values()
    assert values["tractor_wheelbase"] == "" and values["reverse_speed"] == ""
    assert values["random_trailer_hitch_to_axle"] == ""
    assert values["observation_noise_std"] == ""

    controller.apply_parameters(
        {
            "tractor_wheelbase": "2.3",
            "trailer_rear_overhang": "0.8",
            "trailer_body_width": "1.1",
            "steering_rate_max": "0.4",
            "reverse_speed": "0.15",
            "tractor_chassis_mass": "950",
            "random_trailer_hitch_to_axle": "3.0, 4.2",
            "observation_noise_std": "0.02",
        }
    )
    scenario = controller.scenario
    assert scenario.overrides.tractor_wheelbase == 2.3
    assert scenario.randomization.trailer_hitch_to_axle == (3.0, 4.2)
    assert scenario.randomization.observation_noise_std == 0.02
    built = controller.built
    assert built.vehicle.trailer.variant == "custom"
    assert built.vehicle.limits.steering_rate_abs_max == 0.4
    assert built.vehicle.command.nominal_target_speed == -0.15
    assert built.vehicle.mujoco.tractor_chassis_mass == 950.0
    shown = controller.parameter_values()
    assert shown["random_trailer_hitch_to_axle"] == "3.0, 4.2"
    assert shown["observation_noise_std"] == "0.02"
    assert shown["trailer_rear_overhang"] == "0.8"
    # The preview uses the edited truck.
    state = controller.preview_state()
    assert built.vehicle.trailer.rear_overhang == 0.8
    assert state.hitch_position.x - state.trailer_pose.x == pytest.approx(4.0)
    # A fixed trailer length cannot be combined with a randomized one.
    with pytest.raises(ParameterError, match="not both"):
        controller.apply_parameters({"trailer_hitch_to_axle": "3.4"})

    # Blank fields turn the randomization off and restore the base values.
    controller.apply_parameters(
        {
            "random_trailer_hitch_to_axle": "",
            "observation_noise_std": " ",
            "tractor_wheelbase": "",
        }
    )
    assert controller.scenario.randomization.trailer_hitch_to_axle is None
    assert controller.scenario.randomization.observation_noise_std == 0.0
    assert controller.scenario.overrides.tractor_wheelbase is None
    assert controller.scenario.overrides.trailer_rear_overhang == 0.8
    controller.apply_parameters({"trailer_hitch_to_axle": "3.4"})
    assert controller.built.vehicle.trailer.hitch_to_axle == 3.4


@pytest.mark.parametrize(
    "values",
    [
        {"tractor_wheelbase": "-2"},
        {"trailer_hitch_to_axle": "abc"},
        {"random_trailer_hitch_to_axle": "5, 3"},
        {"random_trailer_hitch_to_axle": "4"},
        {"observation_noise_std": "-0.1"},
        {"steering_angle_max": "1.6"},
    ],
)
def test_bad_parameters_leave_the_scenario_unchanged(
    tmp_path: Path, values: dict[str, str]
) -> None:
    controller, _, _ = make(tmp_path)
    before = controller.scenario
    with pytest.raises(ParameterError):
        controller.apply_parameters(values)
    assert controller.scenario == before


def test_planner_and_path_reward_weights_are_scenario_settings(
    tmp_path: Path,
) -> None:
    controller, _, _ = make(tmp_path)
    assert controller.parameter_values()["planner"] == "entry"
    controller.apply_parameters({"lateral": "1.2, 1.2", "standoff": "5, 5"})
    straight = controller.preview_route()
    controller.apply_parameters(
        {"planner": "astar", "path_lateral_weight": "4", "path_heading_weight": "1.5"}
    )
    assert controller.scenario.planner == "astar"
    assert controller.parameter_values()["planner"] == "astar"
    assert controller.built.task.reward_lateral_error == 4.0
    assert controller.built.task.reward_heading_error == 1.5
    searched = controller.preview_route()
    # Swapping the planner changes the route that is drawn and tracked.
    assert searched and straight and searched != straight
    assert searched[0] == straight[0] and searched[-1] == straight[-1]
    # A blank choice restores the default planner and the task's own weights.
    controller.apply_parameters({"planner": "", "path_lateral_weight": ""})
    assert controller.scenario.planner == "entry"
    assert controller.built.task.reward_lateral_error == 0.5
    with pytest.raises(ParameterError):
        controller.apply_parameters({"planner": "rrt"})
    with pytest.raises(ParameterError):
        controller.apply_parameters({"path_lateral_weight": "-1"})


def test_3d_preview_draws_the_route_only_when_asked(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    controller, _, _ = make(tmp_path)
    try:
        shown = controller.preview_frame(show_route=True)
    except ControllerError as error:  # no OpenGL on this machine
        pytest.skip(str(error))
    hidden = controller.preview_frame(show_route=False)
    assert not np.array_equal(shown, hidden)


# ---------------------------------------------------------------- training


def test_training_starts_a_campaign_with_the_chosen_settings(tmp_path: Path) -> None:
    controller, _, campaigns = make(tmp_path)
    config = controller.make_training_config(
        runs=3,
        backend="3d",
        method="ppo",
        evaluation_episodes=5,
        evaluation_backends=("2d", "3d"),
    )
    assert isinstance(config, TrainingConfig)
    controller.start_training(config)
    assert campaigns[0].config is config
    assert campaigns[0].options == {"output_root": tmp_path}
    controller.start_training(config, show_3d=True)
    assert campaigns[1].options == {"output_root": tmp_path, "live_3d": True}
    assert controller.spectator_frame is campaigns[1].spectator_frame

    lines = controller.training_status_lines()
    assert "Run: 2 of 2" in lines
    assert "Now: testing run 2 of 2 in 3d (2 episodes)" in lines
    assert "Run 1 (seed 4): 2d 50 % of 2, 3d 0 % of 2" in lines
    assert "Run 2 (seed 5): not tested" in lines
    for changes in (
        {"backend": "4d"},
        {"runs": 0},
        {"method": "lqr"},
        # NMPC is only tested, so a run with no test episodes would do nothing.
        {"method": "nmpc", "evaluation_episodes": 0},
    ):
        with pytest.raises(ParameterError):
            controller.make_training_config(**changes)


def test_default_factory_builds_a_real_campaign(tmp_path: Path) -> None:
    controller = AppController(output_root=tmp_path)
    controller.start_training(
        controller.make_training_config(method="nmpc", evaluation_episodes=1)
    )
    wait_until(lambda: not controller.training_active)
    progress = controller.training_progress
    # NMPC plugs in through one hook; until it exists the run says so.
    assert progress.status == "failed"
    assert "NMPC is not implemented yet" in progress.error
    assert any("NMPC is not implemented" in line for line in controller.status_lines())


# ------------------------------------------------------------------- tests


def test_tests_run_on_the_chosen_simulator(tmp_path: Path) -> None:
    controller, runner, _ = make(tmp_path)
    run_test(controller, baseline=True)
    assert "backend" not in runner.calls[0] and "on_frame" not in runner.calls[0]
    assert controller.test_policy == "untrained straight-reverse baseline [2d]"
    assert controller.spectator_frame is None

    run_test(controller, baseline=True, backend="3d", show_3d=True)
    assert runner.calls[1]["backend"] == "3d"
    assert controller.test_policy.endswith("[3d]")
    frame = controller.spectator_frame
    assert frame is not None and int(frame[0, 0, 0]) == 7

    # Rendering is only meaningful for the 3D simulator.
    run_test(controller, baseline=True, backend="2d", show_3d=True)
    assert "on_frame" not in runner.calls[2]
    assert controller.spectator_frame is None  # cleared for the new test

    for options in (
        {"backend": "4d"},
        {"method": "lqr"},
        {"baseline": True, "use_bundle": True},
        {"policy_path": "x.zip", "use_bundle": True},
        {"use_bundle": True},
    ):
        with pytest.raises(ParameterError):
            controller.start_test(**options)


def test_nmpc_test_uses_the_controller_hook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller, runner, _ = make(tmp_path)
    run_test(controller, method="nmpc")
    assert controller.test_status == "failed"
    assert "NMPC is not implemented yet" in str(controller.test_error)
    assert runner.calls == []

    import trailer_docking.controllers.nmpc as module

    def hook(env: Any) -> Any:
        return lambda observation: [0.0]

    monkeypatch.setattr(module, "make_docking_controller", hook, raising=False)
    run_test(controller, method="nmpc", backend="3d")
    assert controller.test_status == "finished"
    assert runner.calls[0]["policy"] is None
    assert runner.calls[0]["controller_factory"] is hook
    assert controller.test_policy == "NMPC controller [3d]"


# ----------------------------------------------------------------- bundles


def _bundle(tmp_path: Path, content: bytes) -> Path:
    checkpoint = tmp_path / "policy.zip"
    checkpoint.write_bytes(content)
    scenario = Scenario(
        name="trained", overrides=ScenarioOverrides(trailer_hitch_to_axle=3.6)
    )
    return export_bundle(
        checkpoint,
        scenario,
        tmp_path / "trained.dockpolicy.zip",
        training=TrainingConfig(max_episode_steps=123, backend="3d"),
        source={"campaign_id": "campaign-9"},
    )


def test_export_and_load_result(
    tmp_path: Path, monkeypatch: Any, policy_checkpoint: bytes
) -> None:
    controller, runner, campaigns = make(tmp_path)
    with pytest.raises(ControllerError, match="no finished training result"):
        controller.export_result(tmp_path / "out.zip")
    assert controller.latest_bundle is None and controller.loaded_bundle is None

    source = _bundle(tmp_path, policy_checkpoint)
    controller.start_training(controller.make_training_config())
    campaigns[0].bundle = source
    assert controller.latest_bundle == source
    assert f"Result bundle: {source}" in controller.training_status_lines()
    exported = controller.export_result(tmp_path / "share" / "result.dockpolicy.zip")
    assert exported.read_bytes() == source.read_bytes()
    with pytest.raises(ControllerError, match="refusing to overwrite"):
        controller.export_result(exported)

    # Loading adopts the scenario the policy was trained on, and results of
    # a test on the previous scenario no longer apply.
    fresh, runner, _ = make(tmp_path)
    run_test(fresh, baseline=True)
    assert fresh.test_status == "finished" and fresh.test_results
    runner.calls.clear()
    bundle = fresh.load_result(exported)
    assert fresh.test_status == "idle" and fresh.test_results == ()
    assert fresh.test_status_lines() == ("Test: idle",)
    assert fresh.loaded_bundle is bundle
    assert fresh.scenario.name == "trained"
    assert fresh.parameter_values()["trailer_hitch_to_axle"] == "3.6"
    assert fresh.built.vehicle.trailer.hitch_to_axle == 3.6

    monkeypatch.setattr(type(bundle), "load_policy", lambda self: "bundled policy")
    run_test(fresh, use_bundle=True, backend="3d")
    call = runner.calls[0]
    assert call["policy"] == "bundled policy"
    assert call["backend"] == "3d"
    assert call["max_episode_steps"] == 123
    assert call["scenario"].name == "trained"
    assert "campaign-9" in fresh.test_policy and "trained in 3d" in fresh.test_policy

    for bad in (tmp_path / "missing.zip", tmp_path / "policy.zip"):
        with pytest.raises(ParameterError):
            fresh.load_result(bad)
    assert fresh.loaded_bundle is bundle  # a failed load keeps the old one


def test_loading_is_refused_while_busy(
    tmp_path: Path, policy_checkpoint: bytes
) -> None:
    release = threading.Event()

    def blocking(scenario: Scenario, policy: Any, **options: Any) -> list[Any]:
        release.wait(5)
        return []

    controller, _, _ = make(tmp_path, test_runner=blocking)
    controller.start_test(baseline=True)
    try:
        with pytest.raises(ParameterError, match="while a docking test"):
            controller.load_result(_bundle(tmp_path, policy_checkpoint))
    finally:
        release.set()
    wait_until(lambda: controller.test_status != "running")


# ---------------------------------------------------------------- 3D view


def test_3d_preview_renders_the_edited_scene(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    controller, _, _ = make(tmp_path)
    try:
        plain = controller.preview_frame()
    except ControllerError as error:  # no OpenGL on this machine
        pytest.skip(str(error))
    assert plain.shape == (480, 640, 3) and plain.dtype == np.uint8
    controller.apply_parameters({"trailer_hitch_to_axle": "2.0", "standoff": "1, 1"})
    assert not np.array_equal(controller.preview_frame(), plain)


def test_3d_preview_failure_is_a_controller_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import trailer_docking.visualization.render3d as render3d

    def broken(*arguments: Any) -> Any:
        raise RuntimeError("no OpenGL context")

    monkeypatch.setattr(render3d, "render_scene", broken)
    controller, _, _ = make(tmp_path)
    with pytest.raises(ControllerError, match="3D preview unavailable"):
        controller.preview_frame()


def test_top_down_view_shows_and_hides_a_route() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from trailer_docking.core.state import Point2D
    from trailer_docking.visualization.topdown import TopDownView

    built = Scenario().build()
    view = TopDownView(built.vehicle, built.dock)
    view.set_route((Point2D(0.25, 0.0), Point2D(6.75, 0.0), Point2D(9.0, 0.4)))
    assert list(view._route.get_xdata()) == [0.25, 6.75, 9.0]
    assert list(view._route.get_ydata()) == [0.0, 0.0, 0.4]
    view.set_route(None)
    assert list(view._route.get_xdata()) == []
    plt.close(view.figure)
