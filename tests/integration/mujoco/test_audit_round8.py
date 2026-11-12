"""MuJoCo tests closing coverage gaps found by the eighth audit."""

from __future__ import annotations

from math import sin

import pytest

from trailer_docking.core.config import VehicleConfiguration
from trailer_docking.core.state import Pose2D, VehicleCommand

mujoco = pytest.importorskip("mujoco")

from trailer_docking.simulators.mujoco.adapter import (  # noqa: E402
    MujocoSimulator,
)


def test_reset_clears_the_previous_episodes_command_state(
    vehicle_config: VehicleConfiguration,
) -> None:
    simulator = MujocoSimulator(vehicle_config)
    simulator.reset()
    for _ in range(30):
        simulator.step(VehicleCommand(0.3, -0.2))
    assert simulator.steering_target > 0.1 and simulator.speed_target < -0.1

    state = simulator.reset()
    assert simulator.steering_target == 0.0
    assert simulator.speed_target == 0.0
    assert simulator.applied_command == VehicleCommand(0.0, 0.0)
    assert not simulator.data.ctrl.any()
    assert state.simulation_time == 0.0

    # A reset with a steering angle starts the target there, not at zero.
    simulator.reset(steering_angle=0.2)
    assert simulator.steering_target == pytest.approx(0.2)
    assert simulator.data.ctrl[:2].all()

    # After reset the simulator behaves exactly like a new one.
    simulator.reset()
    fresh = MujocoSimulator(vehicle_config)
    fresh.reset()
    for _ in range(40):
        command = VehicleCommand(-0.2, -0.2)
        assert simulator.step(command).to_dict() == fresh.step(command).to_dict()


@pytest.mark.parametrize("articulation", [0.3, -0.3])
def test_trailer_yaw_rate_matches_the_kinematic_equation(
    vehicle_config: VehicleConfiguration, articulation: float
) -> None:
    simulator = MujocoSimulator(vehicle_config)
    # Tractor straight, trailer swung to one side, driving forward.
    simulator.reset(Pose2D(0.0, 0.0, 0.0), -articulation, settle_time=0.3)
    previous = simulator.state
    for _ in range(25):
        state = simulator.step(VehicleCommand(0.0, 0.2))
        measured_trailer_rate = (
            state.trailer_pose.yaw - previous.trailer_pose.yaw
        ) / simulator.controller_period
        previous = state

    # Specification section 10: psi_r_dot = v sin(psi_t - psi_r) / L_r.
    expected = (
        state.longitudinal_speed
        * sin(state.articulation_angle)
        / vehicle_config.trailer.hitch_to_axle
    )
    assert abs(expected) > 0.005
    assert state.trailer_yaw_rate == pytest.approx(expected, rel=0.1)
    # The reported rate has the sign and size of the trailer's actual motion,
    # and is distinct from the tractor's (which is going straight).
    assert state.trailer_yaw_rate == pytest.approx(measured_trailer_rate, rel=0.1)
    assert abs(state.tractor_yaw_rate) < abs(state.trailer_yaw_rate) / 3.0
    assert state.articulation_rate == pytest.approx(
        state.tractor_yaw_rate - state.trailer_yaw_rate
    )
    # Driving forward straightens the trailer: articulation decays toward zero.
    assert state.articulation_rate * articulation < 0.0
    # Speed is measured along the tractor, not the trailer.
    assert state.longitudinal_speed == pytest.approx(0.2, abs=0.01)


def test_rear_differential_slows_the_inner_wheel(
    vehicle_config: VehicleConfiguration,
) -> None:
    simulator = MujocoSimulator(vehicle_config)
    simulator.reset(steering_angle=0.3)
    simulator.step(VehicleCommand(0.0, 0.2))
    left = simulator.data.actuator("rear_left_wheel_velocity").ctrl[0]
    right = simulator.data.actuator("rear_right_wheel_velocity").ctrl[0]
    # Positive steering turns left, so the left rear wheel is the inner wheel.
    assert 0.0 < left < right
    # Their mean is the rear-axle midpoint speed over the wheel radius.
    assert (left + right) / 2.0 == pytest.approx(
        simulator.speed_target / vehicle_config.wheels.radius
    )
