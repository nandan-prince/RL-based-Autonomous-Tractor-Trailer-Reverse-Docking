"""Shared command-sequence parity between the 2D and MuJoCo backends."""

import pytest

from trailer_docking.core.config import VehicleConfiguration
from trailer_docking.core.state import VehicleCommand
from trailer_docking.simulators.base import SimulatorBackend
from trailer_docking.simulators.kinematic import KinematicSimulator

mujoco = pytest.importorskip("mujoco")

from trailer_docking.simulators.mujoco.adapter import MujocoSimulator  # noqa: E402


def test_both_backends_satisfy_the_common_protocol(
    vehicle_config: VehicleConfiguration,
) -> None:
    assert isinstance(KinematicSimulator(vehicle_config), SimulatorBackend)
    assert isinstance(MujocoSimulator(vehicle_config), SimulatorBackend)


@pytest.mark.parametrize("speed", [0.2, -0.2])
def test_scripted_low_speed_trajectory_parity(
    vehicle_config: VehicleConfiguration, speed: float
) -> None:
    kinematic = KinematicSimulator(vehicle_config)
    physics = MujocoSimulator(vehicle_config)
    physics.reset(settle_time=0.5)
    commands = (
        [VehicleCommand(0.3, speed)] * 8
        + [VehicleCommand(0.0, speed)] * 60
        + [VehicleCommand(-0.3, speed)] * 16
        + [VehicleCommand(0.0, speed)] * 66
    )
    kinematic_state = kinematic.state
    physics_state = physics.state
    for command in commands:
        kinematic_state = kinematic.step(command)
        physics_state = physics.step(command)

    assert physics_state.tractor_pose.x == pytest.approx(
        kinematic_state.tractor_pose.x, abs=0.05
    )
    assert physics_state.tractor_pose.y == pytest.approx(
        kinematic_state.tractor_pose.y, abs=0.05
    )
    assert physics_state.tractor_pose.yaw == pytest.approx(
        kinematic_state.tractor_pose.yaw, abs=0.02
    )
    assert physics_state.trailer_pose.yaw == pytest.approx(
        kinematic_state.trailer_pose.yaw, abs=0.02
    )
    assert physics_state.simulation_time == pytest.approx(
        kinematic_state.simulation_time
    )
