"""Gymnasium docking environment over the shared simulator protocol."""

from __future__ import annotations

import pytest

from trailer_docking.core.config import DockConfiguration, VehicleConfiguration
from trailer_docking.core.state import Pose2D, VehicleCommand, VehicleState

gymnasium = pytest.importorskip("gymnasium")
np = pytest.importorskip("numpy")
pytest.importorskip("mujoco")

from gymnasium.utils.env_checker import check_env  # noqa: E402

from trailer_docking.environments.docking_env import (  # noqa: E402
    OUTCOME_DOCKED,
    OUTCOME_JACKKNIFE,
    OUTCOME_TIMEOUT,
    DockingEnv,
)
from trailer_docking.evaluation.scenarios import (  # noqa: E402
    Scenario,
    StartDistribution,
)
from trailer_docking.paths.reference import ReferencePath  # noqa: E402
from trailer_docking.simulators.base import Simulator  # noqa: E402
from trailer_docking.simulators.mujoco.adapter import MujocoSimulator  # noqa: E402

ALIGNED = Scenario(start=StartDistribution((4, 4), (0, 0), (0, 0), (0, 0)))
ZERO = np.zeros(1, dtype=np.float32)


def test_mujoco_backend_satisfies_the_simulator_protocol(
    vehicle_config: VehicleConfiguration, dock_config: DockConfiguration
) -> None:
    assert isinstance(MujocoSimulator(vehicle_config, dock_config), Simulator)
    assert not isinstance(object(), Simulator)


def test_environment_passes_the_gymnasium_checker() -> None:
    env = DockingEnv()
    check_env(env, skip_render_check=True)
    assert env.observation_space.shape == (env.layout.size,)
    assert env.action_space.shape == (1,)


def test_reset_plans_a_route_and_is_reproducible() -> None:
    env = DockingEnv()
    first, info = env.reset(seed=5)
    path = env.reference_path
    assert isinstance(path, ReferencePath)
    assert info["path_progress"] == pytest.approx(0.0, abs=1e-6)
    assert info["path_remaining"] == pytest.approx(path.length, abs=1e-6)
    # The route begins at the trailer and ends at the dock.
    state = env.simulator.state
    assert path.poses[0].to_dict() == pytest.approx(
        state.trailer_pose.to_dict(), abs=1e-6
    )
    again, _ = env.reset(seed=5)
    assert np.array_equal(first, again)
    other, _ = env.reset(seed=6)
    assert not np.array_equal(first, other)
    assert env.reference_path is not path


def test_step_before_reset_is_an_error() -> None:
    with pytest.raises(RuntimeError):
        DockingEnv().step(ZERO)
    with pytest.raises(ValueError):
        DockingEnv(max_episode_steps=0)


def test_straight_reverse_follows_the_route_into_the_dock() -> None:
    seen: list[VehicleState] = []
    env = DockingEnv(ALIGNED, on_state=seen.append)
    env.reset(seed=0)
    assert env.reference_path is not None
    assert env.reference_path.max_abs_curvature == pytest.approx(0.0, abs=1e-9)
    total, progress = 0.0, []
    for _ in range(900):
        _, reward, terminated, truncated, info = env.step(ZERO)
        total += reward
        progress.append(info["path_progress"])
        assert abs(info["path_lateral_error"]) < 0.02
        if terminated or truncated:
            break
    assert terminated and not truncated
    assert info["outcome"] == OUTCOME_DOCKED and info["is_success"]
    assert abs(info["longitudinal_error"]) <= 0.10
    assert progress == sorted(progress)
    assert total > 10.0
    assert len(seen) == len(progress) + 1


def test_full_lock_ends_in_a_jackknife_with_a_penalty() -> None:
    env = DockingEnv()
    env.reset(seed=3)
    rewards = []
    for _ in range(900):
        _, reward, terminated, truncated, info = env.step(np.ones(1, dtype=np.float32))
        rewards.append(reward)
        if terminated or truncated:
            break
    assert terminated and info["outcome"] == OUTCOME_JACKKNIFE
    assert not info["is_success"]
    assert rewards[-1] < -5.0


def test_timeout_truncates_and_actions_are_clipped() -> None:
    env = DockingEnv(ALIGNED, max_episode_steps=5)
    env.reset(seed=0)
    for index in range(5):
        _, _, terminated, truncated, info = env.step(np.array([50.0]))
        assert not terminated
        assert truncated == (index == 4)
    assert info["outcome"] == OUTCOME_TIMEOUT
    limit = env.vehicle.limits.steering_rate_abs_max
    assert env.simulator.applied_command.steering_rate <= limit + 1e-9


class _RecordingSimulator:
    """Minimal protocol implementation: a kinematic stand-in for a backend."""

    def __init__(self, vehicle: VehicleConfiguration) -> None:
        self._geometry = vehicle.geometry
        self._pose = Pose2D(0.0, 0.0, 0.0)
        self._trailer_yaw = 0.0
        self._time = 0.0
        self.commands: list[VehicleCommand] = []
        self.controller_period = vehicle.timing.controller_period

    @property
    def state(self) -> VehicleState:
        return self._geometry.assembled_state(
            self._pose, self._trailer_yaw, simulation_time=self._time
        )

    @property
    def applied_command(self) -> VehicleCommand:
        return self.commands[-1] if self.commands else VehicleCommand(0.0, 0.0)

    def reset(self, tractor_pose=None, trailer_yaw=None, *, steering_angle=0.0):
        self._pose = tractor_pose or Pose2D(0.0, 0.0, 0.0)
        self._trailer_yaw = trailer_yaw or 0.0
        self._time = 0.0
        self.commands.clear()
        return self.state

    def step(self, command: VehicleCommand) -> VehicleState:
        self.commands.append(command)
        distance = command.target_speed * self.controller_period
        self._pose = Pose2D(self._pose.x + distance, self._pose.y, 0.0)
        self._time += self.controller_period
        return self.state


def test_environment_is_backend_independent() -> None:
    created: list[_RecordingSimulator] = []

    def factory(vehicle, dock):
        created.append(_RecordingSimulator(vehicle))
        return created[-1]

    env = DockingEnv(ALIGNED, simulator_factory=factory)
    assert isinstance(created[0], Simulator)
    env.reset(seed=0)
    for _ in range(900):
        _, _, terminated, truncated, info = env.step(np.array([0.5], dtype=np.float32))
        if terminated or truncated:
            break
    assert info["outcome"] == OUTCOME_DOCKED
    command = created[0].commands[0]
    assert command.target_speed == env.vehicle.command.nominal_target_speed
    assert command.steering_rate == pytest.approx(
        0.5 * env.vehicle.limits.steering_rate_abs_max
    )
