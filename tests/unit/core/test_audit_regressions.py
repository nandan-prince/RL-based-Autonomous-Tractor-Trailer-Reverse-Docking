"""Regression tests for edge-case defects found by repository audits."""

import math
from pathlib import Path

import pytest

from trailer_docking.core.config import (
    ConfigurationError,
    SimulationTiming,
    VehicleConfiguration,
    load_vehicle_configuration,
)
from trailer_docking.core.geometry import DockGeometry, GeometryValidationError
from trailer_docking.core.state import (
    Point2D,
    Pose2D,
    StateValidationError,
    VehicleCommand,
    VehicleState,
    wrap_angle,
)
from trailer_docking.safety.supervisor import supervise_command


@pytest.mark.parametrize(
    "angle",
    [
        math.nextafter(-math.pi, -math.inf),
        math.nextafter(math.pi, math.inf),
        -3.0 * math.pi - 4e-16,
        math.pi,
        -math.pi,
        1e300,
    ],
)
def test_wrap_angle_never_returns_positive_pi(angle: float) -> None:
    wrapped = wrap_angle(angle)
    assert -math.pi <= wrapped < math.pi


def test_serialized_articulation_must_be_numeric() -> None:
    state = VehicleState(
        Pose2D(0.0, 0.0, 0.2), Pose2D(-4.0, 0.0, 0.0), Point2D(0.0, 0.0), *([0.0] * 6)
    )
    data = state.to_dict()
    data["articulation_angle"] = "0.2"
    with pytest.raises(StateValidationError):
        VehicleState.from_dict(data)


def test_integration_step_cannot_exceed_controller_period() -> None:
    with pytest.raises(ConfigurationError):
        SimulationTiming(0.1, 1e12, 0.002, 0.0)
    with pytest.raises(ConfigurationError):
        SimulationTiming(0.1, 0.02, 1e12, 0.0)


def test_duplicate_yaml_keys_are_rejected(
    tmp_path: Path, vehicle_config: VehicleConfiguration
) -> None:
    text = vehicle_config.source_path.read_text(encoding="utf-8")
    assert "  width: 0.12\n" in text
    path = tmp_path / "duplicate.yaml"
    path.write_text(
        text.replace("  width: 0.12\n", "  width: 0.12\n  width: 0.50\n"),
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="duplicate key"):
        load_vehicle_configuration(path)


def test_dock_rejects_non_finite_margin_and_untyped_target() -> None:
    arguments = dict(
        specification_id="dock",
        interior_width=1.5,
        interior_depth=6.0,
        wall_thickness=0.1,
        wall_height=1.0,
        collision_margin=0.03,
        target_position=Point2D(0.25, 0.0),
        target_trailer_yaw=0.0,
        longitudinal_tolerance=0.1,
        lateral_tolerance=0.1,
        trailer_yaw_tolerance=0.05,
        require_no_collision=True,
        require_articulation_below_hard_limit=True,
        require_complete_trailer_inside_bay=True,
    )
    DockGeometry(**arguments)  # type: ignore[arg-type]
    with pytest.raises(GeometryValidationError):
        DockGeometry(**{**arguments, "collision_margin": math.nan})  # type: ignore[arg-type]
    with pytest.raises(GeometryValidationError):
        DockGeometry(**{**arguments, "target_position": (0.25, 0.0)})  # type: ignore[arg-type]


@pytest.mark.parametrize("angle", [0.6, -0.6])
def test_supervisor_respects_rate_limit_when_angle_is_out_of_bounds(
    vehicle_config: VehicleConfiguration, angle: float
) -> None:
    geometry = vehicle_config.geometry
    state = geometry.assembled_state(Pose2D(0.0, 0.0, 0.0), 0.0, steering_angle=angle)
    for requested in (-5.0, 0.0, 5.0):
        assessment = supervise_command(
            VehicleCommand(requested, 0.0),
            state,
            vehicle_config.limits,
            vehicle_config.timing,
        )
        rate = assessment.applied.steering_rate
        assert abs(rate) <= vehicle_config.limits.steering_rate_abs_max
        # The applied rate always moves the angle back toward its bound.
        assert rate * angle < 0.0
        assert "steering_angle_limit" in assessment.reasons
