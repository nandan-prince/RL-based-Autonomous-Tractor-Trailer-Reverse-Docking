"""State management, plant-limit, and behavior tests for the 2D backend."""

from dataclasses import replace
from math import pi

import pytest

from trailer_docking.core.config import VehicleConfiguration
from trailer_docking.core.state import Pose2D, VehicleCommand, VehicleState
from trailer_docking.safety.supervisor import state_limit_violations
from trailer_docking.simulators.base import SimulatorBackend
from trailer_docking.simulators.kinematic import KinematicSimulator

STOP = VehicleCommand(0.0, 0.0)


def _run(
    simulator: KinematicSimulator, command: VehicleCommand, steps: int
) -> VehicleState:
    state = simulator.state
    for _ in range(steps):
        state = simulator.step(command)
    return state


def test_backend_satisfies_common_protocol(
    vehicle_config: VehicleConfiguration,
) -> None:
    simulator = KinematicSimulator(vehicle_config)
    assert isinstance(simulator, SimulatorBackend)
    assert simulator.controller_period == pytest.approx(0.1)
    with pytest.raises(ValueError):
        KinematicSimulator(object())  # type: ignore[arg-type]


def test_reset_reconstructs_pose_and_clears_previous_command(
    vehicle_config: VehicleConfiguration,
) -> None:
    simulator = KinematicSimulator(vehicle_config)
    _run(simulator, VehicleCommand(0.3, -0.2), 10)
    pose = Pose2D(3.0, -2.0, 0.7)
    state = simulator.reset(pose, -0.2, steering_angle=-0.25)
    expected = vehicle_config.geometry.assembled_state(
        pose, -0.2, steering_angle=-0.25
    )
    assert state.to_dict() == expected.to_dict()
    assert simulator.applied_command == STOP
    assert state.simulation_time == 0.0


def test_rejected_reset_leaves_existing_episode_untouched(
    vehicle_config: VehicleConfiguration,
) -> None:
    simulator = KinematicSimulator(vehicle_config)
    _run(simulator, VehicleCommand(0.2, -0.2), 12)
    before = simulator.state.to_dict()
    for arguments in (
        {"tractor_pose": (1.0, 2.0, 0.0)},
        {"tractor_pose": ()},
        {"trailer_yaw": float("nan")},
        {"steering_angle": 0.6},
        {"steering_angle": None},
        {"steering_angle": True},
    ):
        with pytest.raises(ValueError):
            simulator.reset(**arguments)  # type: ignore[arg-type]
        assert simulator.state.to_dict() == before


def test_rejected_step_leaves_existing_episode_untouched(
    vehicle_config: VehicleConfiguration,
) -> None:
    simulator = KinematicSimulator(vehicle_config)
    before = simulator.state.to_dict()
    with pytest.raises(ValueError):
        simulator.step((0.0, -0.2))  # type: ignore[arg-type]
    assert simulator.state.to_dict() == before


def test_stationary_and_straight_motion_are_exact(
    vehicle_config: VehicleConfiguration,
) -> None:
    simulator = KinematicSimulator(vehicle_config)
    stationary = _run(simulator, STOP, 10)
    assert stationary.tractor_pose == Pose2D(0.0, 0.0, 0.0)
    assert stationary.trailer_pose.yaw == 0.0
    assert stationary.longitudinal_speed == 0.0
    simulator.reset()
    forward = _run(simulator, VehicleCommand(0.0, 0.2), 10)
    assert forward.tractor_pose.x == pytest.approx(0.15)
    assert forward.tractor_pose.y == 0.0
    assert forward.trailer_pose.y == 0.0
    assert forward.longitudinal_speed == pytest.approx(0.2)
    assert forward.articulation_angle == 0.0
    simulator.reset()
    reverse = _run(simulator, VehicleCommand(0.0, -0.2), 10)
    assert reverse.tractor_pose.x == pytest.approx(-0.15)
    assert reverse.longitudinal_speed == pytest.approx(-0.2)
    assert reverse.articulation_angle == 0.0


def test_controller_period_and_acceleration_ramp(
    vehicle_config: VehicleConfiguration,
) -> None:
    simulator = KinematicSimulator(vehicle_config)
    expected_speeds = (0.04, 0.08, 0.12, 0.16, 0.20)
    for index, expected in enumerate(expected_speeds, start=1):
        state = simulator.step(VehicleCommand(0.0, 0.2))
        assert state.longitudinal_speed == pytest.approx(expected)
        assert state.simulation_time == pytest.approx(0.1 * index)
        assert simulator.applied_command.target_speed == pytest.approx(expected)
    assert state.tractor_pose.x == pytest.approx(0.05)


def test_constant_steering_has_expected_signs_and_rigid_geometry(
    vehicle_config: VehicleConfiguration,
) -> None:
    simulator = KinematicSimulator(vehicle_config)
    for _ in range(80):
        state = simulator.step(VehicleCommand(0.3, 0.2))
        assert max(vehicle_config.geometry.constraint_residuals(state).values()) < 1e-12
    assert state.steering_angle == pytest.approx(0.5)
    assert state.tractor_pose.yaw > 0.0
    assert state.tractor_pose.y > 0.0
    assert state.articulation_angle > 0.0
    assert state_limit_violations(state, vehicle_config.limits) == ()


def test_reverse_steering_grows_articulation_with_expected_sign(
    vehicle_config: VehicleConfiguration,
) -> None:
    simulator = KinematicSimulator(vehicle_config)
    state = _run(simulator, VehicleCommand(0.3, -0.2), 30)
    assert state.steering_angle > 0.0
    assert state.tractor_pose.yaw < 0.0
    assert state.articulation_angle < 0.0
    assert abs(state.articulation_angle) > 0.05


def test_angle_wrapping_is_preserved_during_propagation(
    vehicle_config: VehicleConfiguration,
) -> None:
    simulator = KinematicSimulator(vehicle_config)
    simulator.reset(Pose2D(0.0, 0.0, pi - 0.01), pi - 0.01, steering_angle=0.4)
    state = _run(simulator, VehicleCommand(0.0, 0.2), 10)
    assert -pi <= state.tractor_pose.yaw < pi
    assert -pi <= state.trailer_pose.yaw < pi
    assert state.tractor_pose.x < 0.0


def test_all_command_limits_are_auditable(
    vehicle_config: VehicleConfiguration,
) -> None:
    simulator = KinematicSimulator(vehicle_config)
    state = simulator.step(VehicleCommand(5.0, 5.0))
    assert simulator.applied_command.steering_rate == pytest.approx(0.3)
    assert simulator.applied_command.target_speed == pytest.approx(0.04)
    assert simulator.command_assessment.reasons == (
        "steering_rate_limit",
        "speed_limit",
        "acceleration_limit",
    )
    assert state_limit_violations(state, vehicle_config.limits) == ()
    _run(simulator, VehicleCommand(5.0, 0.0), 30)
    state = simulator.step(VehicleCommand(5.0, 0.0))
    assert state.steering_angle == pytest.approx(0.5)
    assert simulator.applied_command.steering_rate == pytest.approx(0.0)
    assert "steering_angle_limit" in simulator.command_assessment.reasons


@pytest.mark.parametrize("variant", ["short", "nominal", "long"])
def test_trailer_variants_propagate_without_constraint_drift(
    vehicle_config: VehicleConfiguration, variant: str
) -> None:
    vehicle = vehicle_config.with_trailer_variant(variant)
    simulator = KinematicSimulator(vehicle)
    state = _run(simulator, VehicleCommand(-0.2, -0.2), 40)
    assert max(vehicle.geometry.constraint_residuals(state).values()) < 1e-12
    assert state.trailer_pose != state.tractor_pose


def test_nonzero_command_delay_is_rejected(
    vehicle_config: VehicleConfiguration,
) -> None:
    timing = replace(vehicle_config.timing, nominal_command_delay=0.1)
    with pytest.raises(NotImplementedError):
        KinematicSimulator(replace(vehicle_config, timing=timing))


def test_rollout_is_bitwise_repeatable(vehicle_config: VehicleConfiguration) -> None:
    commands = [
        VehicleCommand(0.2 if index < 20 else -0.1, -0.2)
        for index in range(60)
    ]

    def rollout() -> dict[str, object]:
        simulator = KinematicSimulator(vehicle_config)
        state = simulator.state
        for command in commands:
            state = simulator.step(command)
        return state.to_dict()

    assert rollout() == rollout()


def test_smaller_timestep_converges(vehicle_config: VehicleConfiguration) -> None:
    commands = [VehicleCommand(0.3, -0.2)] * 10 + [
        VehicleCommand(0.0, -0.2)
    ] * 80

    def rollout(step: float) -> VehicleState:
        timing = replace(vehicle_config.timing, kinematic_integration_step=step)
        simulator = KinematicSimulator(replace(vehicle_config, timing=timing))
        state = simulator.state
        for command in commands:
            state = simulator.step(command)
        return state

    coarse, medium, fine = rollout(0.05), rollout(0.025), rollout(0.0125)
    coarse_error = abs(coarse.trailer_pose.yaw - fine.trailer_pose.yaw)
    medium_error = abs(medium.trailer_pose.yaw - fine.trailer_pose.yaw)
    assert medium_error < coarse_error
    assert coarse_error < 1e-3
