"""Shared controller action and context contract tests."""

import numpy as np
import pytest

from trailer_docking.controllers.base import (
    ControllerContext,
    ControllerError,
    normalized_steering_rate,
    validate_controller_action,
)
from trailer_docking.controllers.ppo import PPOController
from trailer_docking.environments import TrailerDockingEnv


def test_normalized_steering_rate_clips_and_uses_float32(vehicle_config) -> None:
    maximum = vehicle_config.limits.steering_rate_abs_max
    positive = normalized_steering_rate(2.0 * maximum, vehicle_config)
    negative = normalized_steering_rate(-0.5 * maximum, vehicle_config)
    np.testing.assert_array_equal(positive, np.array([1.0], dtype=np.float32))
    np.testing.assert_array_equal(negative, np.array([-0.5], dtype=np.float32))


def test_action_validation_and_context(vehicle_config) -> None:
    env = TrailerDockingEnv(vehicle=vehicle_config)
    observation, _ = env.reset()
    context = ControllerContext(
        observation,
        env.state,
        env.path,
        env.tracking,
        env.vehicle,
        0,
    )
    assert context.step_index == 0
    validated = validate_controller_action(np.array([0.25], dtype=np.float32))
    assert validated.dtype == np.float32
    with pytest.raises(ControllerError):
        validate_controller_action(np.array([0.0, 0.0], dtype=np.float32))
    with pytest.raises(ControllerError):
        validate_controller_action(np.array([np.nan], dtype=np.float32))
    with pytest.raises(ValueError, match="step_index"):
        ControllerContext(
            observation,
            env.state,
            env.path,
            env.tracking,
            env.vehicle,
            -1,
        )
    env.close()


def test_ppo_adapter_uses_only_the_observation(vehicle_config) -> None:
    env = TrailerDockingEnv(vehicle=vehicle_config)
    observation, _ = env.reset()
    seen: list[np.ndarray] = []

    def policy(value: np.ndarray) -> np.ndarray:
        seen.append(value)
        return np.array([0.4], dtype=np.float32)

    controller = PPOController(policy)
    context = ControllerContext(
        observation,
        env.state,
        env.path,
        env.tracking,
        env.vehicle,
        0,
    )
    action = controller.act(context)
    np.testing.assert_array_equal(action, np.array([0.4], dtype=np.float32))
    assert seen == [observation]
    assert controller.name == "ppo-policy"
    env.close()
