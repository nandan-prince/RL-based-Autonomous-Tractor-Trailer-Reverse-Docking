"""End-to-end straight-path LQR and basic NMPC acceptance tests."""

from math import pi

import numpy as np
import pytest

from trailer_docking.controllers import (
    BasicNMPCController,
    TrajectoryLQRController,
)
from trailer_docking.controllers.base import ControllerContext
from trailer_docking.core.state import Point2D
from trailer_docking.environments import DockingTaskSettings, TrailerDockingEnv
from trailer_docking.environments.task import aligned_pose_for_docking_point
from trailer_docking.evaluation.runner import run_episode
from trailer_docking.paths import straight_reference


def _scenario(env: TrailerDockingEnv) -> dict[str, object]:
    start_x = env.dock.geometry.interior_depth + env.settings.start_standoff
    heading = 3.0 * pi / 180.0
    path = straight_reference(
        env.dock.geometry.target_position,
        Point2D(start_x, 0.0),
    )
    pose = aligned_pose_for_docking_point(
        env.vehicle,
        Point2D(start_x, 0.35),
        heading,
    )
    return {
        "tractor_pose": pose,
        "trailer_yaw": heading,
        "reference_path": path,
    }


def test_lqr_stabilizes_the_linear_model_and_docks() -> None:
    env = TrailerDockingEnv(action_mode="fixed_reverse")
    controller = TrajectoryLQRController(env.vehicle)
    system, control = controller.linear_model
    closed_loop = system - control @ controller.gain
    assert max(abs(np.linalg.eigvals(closed_loop))) < 1.0
    result = run_episode(env, controller, reset_options=_scenario(env))
    assert result.success
    assert result.termination_reason == "success"
    assert not result.collision and not result.jackknifed
    assert abs(result.final_lateral_error) < 0.01
    env.close()


def test_nmpc_produces_valid_action_and_docks() -> None:
    pytest.importorskip("casadi")
    env = TrailerDockingEnv(action_mode="fixed_reverse")
    options = _scenario(env)
    observation, _ = env.reset(options=options)
    controller = BasicNMPCController(env.vehicle)
    context = ControllerContext(
        observation,
        env.state,
        env.path,
        env.tracking,
        env.vehicle,
        0,
    )
    action = controller.act(context)
    assert env.action_space.contains(action)
    assert controller.last_solve_succeeded
    controller.reset()
    result = run_episode(env, controller, reset_options=options)
    assert result.success
    assert result.termination_reason == "success"
    assert result.solve_failures == 0
    assert not result.collision and not result.jackknifed
    env.close()


def test_runner_labels_controller_and_emits_rgb_frames() -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    env = TrailerDockingEnv(
        action_mode="fixed_reverse",
        render_mode="rgb_array",
        task_settings=DockingTaskSettings(max_episode_steps=1),
    )
    controller = TrajectoryLQRController(env.vehicle)
    frames: list[np.ndarray] = []
    result = run_episode(env, controller, frame_callback=frames.append)
    assert result.truncated
    assert len(frames) == 2
    assert all(frame.ndim == 3 and frame.shape[2] == 3 for frame in frames)
    assert env.controller_label == controller.display_name
    assert controller.display_name in env.figure.axes[0].get_title()
    env.close()
