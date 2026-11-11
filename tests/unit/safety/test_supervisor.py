import pytest

from trailer_docking.core.config import VehicleConfiguration
from trailer_docking.core.state import VehicleCommand
from trailer_docking.safety.supervisor import (
    state_limit_violations,
    supervise_command,
)


def test_valid_command_passes_without_intervention(
    vehicle_config: VehicleConfiguration,
) -> None:
    state = vehicle_config.geometry.neutral_state()
    command = VehicleCommand(steering_rate=0.1, target_speed=0.0)
    assessment = supervise_command(
        command, state, vehicle_config.limits, vehicle_config.timing
    )
    assert assessment.applied == command
    assert not assessment.intervened
    assert assessment.reasons == ()


def test_supervisor_records_rate_speed_and_acceleration_limits(
    vehicle_config: VehicleConfiguration,
) -> None:
    state = vehicle_config.geometry.neutral_state()
    command = VehicleCommand(steering_rate=2.0, target_speed=-2.0)
    assessment = supervise_command(
        command, state, vehicle_config.limits, vehicle_config.timing
    )
    assert assessment.requested == command
    assert assessment.applied.steering_rate == pytest.approx(0.3)
    assert assessment.applied.target_speed == pytest.approx(-0.04)
    assert assessment.reasons == (
        "steering_rate_limit",
        "speed_limit",
        "acceleration_limit",
    )


def test_supervisor_prevents_crossing_steering_angle_limit(
    vehicle_config: VehicleConfiguration,
) -> None:
    state = vehicle_config.geometry.assembled_state(
        vehicle_config.geometry.neutral_state().tractor_pose,
        0.0,
        steering_angle=0.49,
    )
    assessment = supervise_command(
        VehicleCommand(0.3, 0.0),
        state,
        vehicle_config.limits,
        vehicle_config.timing,
    )
    assert assessment.applied.steering_rate == pytest.approx(0.1)
    assert "steering_angle_limit" in assessment.reasons


def test_state_limit_violations_are_stable(
    vehicle_config: VehicleConfiguration,
) -> None:
    state = vehicle_config.geometry.assembled_state(
        vehicle_config.geometry.neutral_state().tractor_pose,
        -1.0,
        steering_angle=0.6,
        steering_rate=0.4,
        longitudinal_speed=-0.3,
    )
    assert state_limit_violations(state, vehicle_config.limits) == (
        "steering_angle_limit",
        "steering_rate_limit",
        "reverse_speed_limit",
        "jackknife_limit",
    )
