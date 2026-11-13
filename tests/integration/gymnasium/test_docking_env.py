"""Gymnasium API, episode-event, action, rendering, and stability tests."""

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from trailer_docking.core.state import Point2D, Pose2D
from trailer_docking.environments import (  # registers TrailerDocking-v0
    DockingTaskSettings,
    TrailerDockingEnv,
)
from trailer_docking.environments.task import aligned_pose_for_docking_point


def test_gymnasium_checker_and_registration() -> None:
    env = TrailerDockingEnv()
    check_env(env, skip_render_check=True)
    env.close()
    registered = gym.make("TrailerDocking-v0")
    assert isinstance(registered.unwrapped, TrailerDockingEnv)
    registered.close()


def test_fixed_reverse_action_advances_one_controller_period() -> None:
    env = TrailerDockingEnv()
    observation, info = env.reset(seed=1)
    start_x = env.state.tractor_pose.x
    assert env.observation_space.contains(observation)
    for _ in range(5):
        observation, reward, terminated, truncated, info = env.step(
            np.array([0.0], dtype=np.float32)
        )
    assert env.state.simulation_time == pytest.approx(0.5)
    assert env.state.tractor_pose.x < start_x
    assert env.state.longitudinal_speed == pytest.approx(-0.2)
    assert reward > 0.0
    assert not terminated and not truncated
    assert sum(info["reward_terms"].values()) == pytest.approx(reward)
    env.close()


def test_teleop_mode_controls_forward_reverse_and_steering() -> None:
    env = TrailerDockingEnv(action_mode="teleop")
    env.reset(seed=2)
    start_x = env.state.tractor_pose.x
    for _ in range(8):
        env.step(np.array([1.0, 1.0], dtype=np.float32))
    assert env.state.tractor_pose.x > start_x
    assert env.state.steering_angle > 0.0
    env.reset(seed=2)
    for _ in range(8):
        env.step(np.array([-1.0, -1.0], dtype=np.float32))
    assert env.state.tractor_pose.x < start_x
    assert env.state.steering_angle < 0.0
    env.close()


def test_seeded_randomized_reset_is_reproducible() -> None:
    first = TrailerDockingEnv()
    second = TrailerDockingEnv()
    first_observation, _ = first.reset(seed=47, options={"randomize": True})
    second_observation, _ = second.reset(seed=47, options={"randomize": True})
    assert first.state.to_dict() == second.state.to_dict()
    np.testing.assert_array_equal(first_observation, second_observation)
    assert first.path == second.path
    first.close()
    second.close()


def test_success_collision_jackknife_and_timeout_events(
    vehicle_config, dock_config
) -> None:
    stop = np.array([0.0, 0.0], dtype=np.float32)
    env = TrailerDockingEnv(
        vehicle=vehicle_config, dock=dock_config, action_mode="teleop"
    )
    success_pose = aligned_pose_for_docking_point(
        vehicle_config, dock_config.geometry.target_position
    )
    env.reset(
        options={"tractor_pose": success_pose, "trailer_yaw": 0.0}
    )
    _, reward, terminated, truncated, info = env.step(stop)
    assert terminated and not truncated and info["termination_reason"] == "success"
    assert reward > 90.0
    with pytest.raises(RuntimeError):
        env.step(stop)

    wall_pose = aligned_pose_for_docking_point(vehicle_config, Point2D(0.0, 0.0))
    env.reset(options={"tractor_pose": wall_pose, "trailer_yaw": 0.0})
    _, reward, terminated, _, info = env.step(stop)
    assert terminated and info["termination_reason"] == "collision"
    assert reward < -90.0

    env.reset(
        options={
            "tractor_pose": Pose2D(
                15.0, 0.0, vehicle_config.limits.articulation_hard_abs_max
            ),
            "trailer_yaw": 0.0,
        }
    )
    _, _, terminated, _, info = env.step(stop)
    assert terminated and info["termination_reason"] == "jackknife"
    env.close()

    timeout = TrailerDockingEnv(
        action_mode="teleop", task_settings=DockingTaskSettings(max_episode_steps=1)
    )
    timeout.reset()
    _, _, terminated, truncated, info = timeout.step(stop)
    assert not terminated and truncated and info["termination_reason"] == "time_limit"
    timeout.close()


def test_invalid_actions_are_rejected_without_advancing() -> None:
    env = TrailerDockingEnv()
    env.reset()
    before = env.state.to_dict()
    for action in (
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([2.0], dtype=np.float32),
        np.array([np.nan], dtype=np.float32),
    ):
        with pytest.raises(ValueError):
            env.step(action)
        assert env.state.to_dict() == before
    env.close()


@pytest.mark.parametrize("variant", ["short", "nominal", "long"])
def test_all_trailer_variants_support_rollouts(variant: str) -> None:
    env = TrailerDockingEnv(trailer_variant=variant)
    observation, _ = env.reset(seed=5)
    for _ in range(50):
        observation, _, terminated, truncated, _ = env.step(
            np.array([0.25], dtype=np.float32)
        )
        assert env.observation_space.contains(observation)
        if terminated or truncated:
            break
    assert env.state.simulation_time > 0.0
    env.close()


def test_long_random_rollouts_remain_finite_and_space_compliant() -> None:
    env = TrailerDockingEnv(
        task_settings=DockingTaskSettings(max_episode_steps=200)
    )
    observation, _ = env.reset(seed=123, options={"randomize": True})
    random = np.random.default_rng(123)
    steps = 0
    for _ in range(1_000):
        action = random.uniform(-1.0, 1.0, size=(1,)).astype(np.float32)
        observation, reward, terminated, truncated, _ = env.step(action)
        assert env.observation_space.contains(observation)
        assert np.isfinite(reward)
        steps += 1
        if terminated or truncated:
            observation, _ = env.reset(options={"randomize": True})
    assert steps == 1_000
    env.close()


def test_default_straight_reverse_reaches_docking_success() -> None:
    env = TrailerDockingEnv(action_mode="teleop")
    env.reset(seed=0)
    reverse = np.array([0.0, -1.0], dtype=np.float32)
    info = {}
    for _ in range(env.settings.max_episode_steps):
        _, _, terminated, truncated, info = env.step(reverse)
        if terminated or truncated:
            break
    assert terminated and not truncated
    assert info["termination_reason"] == "success"
    assert not info["collision"]
    env.close()


def test_rgb_array_render_returns_image() -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    env = TrailerDockingEnv(render_mode="rgb_array")
    env.reset()
    image = env.render()
    assert image is not None
    assert image.dtype == np.uint8
    assert image.ndim == 3 and image.shape[2] == 3
    env.close()
