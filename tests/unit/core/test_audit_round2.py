"""Regression tests for defects found by the second repository audit."""

from dataclasses import replace
from pathlib import Path

import pytest

from trailer_docking.core.config import (
    ConfigurationError,
    VehicleConfiguration,
    load_vehicle_configuration,
)
from trailer_docking.core.geometry import (
    Bounds1D,
    GeometryValidationError,
    TrailerGeometry,
)
from trailer_docking.core.state import (
    Point2D,
    Pose2D,
    StateValidationError,
    VehicleCommand,
    VehicleState,
    wrap_angle,
)
from trailer_docking.safety.supervisor import supervise_command
from trailer_docking.simulators.mujoco.builder import _n


@pytest.mark.parametrize("angle", [0.1, -0.1, 1e-17, 3.0, -3.0, 0.7853981633974483])
def test_wrap_angle_is_the_identity_inside_the_interval(angle: float) -> None:
    assert wrap_angle(angle) == angle
    assert Pose2D(0.0, 0.0, angle).yaw == angle


def test_huge_integers_raise_validation_errors() -> None:
    huge = 10**400
    with pytest.raises(StateValidationError):
        Point2D(huge, 0.0)
    with pytest.raises(StateValidationError):
        wrap_angle(huge)
    with pytest.raises(StateValidationError):
        VehicleCommand(huge, 0.0)


def test_huge_yaml_integer_is_a_configuration_error(
    tmp_path: Path, vehicle_config: VehicleConfiguration
) -> None:
    text = vehicle_config.source_path.read_text(encoding="utf-8")
    assert "  wheelbase: 1.90\n" in text
    path = tmp_path / "huge.yaml"
    path.write_text(
        text.replace("  wheelbase: 1.90\n", f"  wheelbase: 1{'0' * 400}\n"),
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError):
        load_vehicle_configuration(path)


@pytest.mark.parametrize("data", [None, 5, [("x", 1.0)], "text"])
def test_from_dict_rejects_non_mappings(data: object) -> None:
    for loader in (
        Point2D.from_dict,
        Pose2D.from_dict,
        VehicleCommand.from_dict,
        VehicleState.from_dict,
    ):
        with pytest.raises(StateValidationError):
            loader(data)  # type: ignore[arg-type]


def test_nested_state_fields_must_be_mappings(
    vehicle_config: VehicleConfiguration,
) -> None:
    data = vehicle_config.geometry.neutral_state().to_dict()
    data["tractor_pose"] = None
    with pytest.raises(StateValidationError):
        VehicleState.from_dict(data)


def test_geometry_rejects_wrong_types(vehicle_config: VehicleConfiguration) -> None:
    with pytest.raises(GeometryValidationError):
        Bounds1D("a", 1.0)  # type: ignore[arg-type]
    with pytest.raises(GeometryValidationError):
        replace(vehicle_config.trailer, variant=1)  # type: ignore[arg-type]
    with pytest.raises(GeometryValidationError):
        replace(vehicle_config.trailer, body_width="1.0")  # type: ignore[arg-type]
    assert isinstance(vehicle_config.trailer, TrailerGeometry)


def test_unreachable_steering_limit_is_rejected(
    vehicle_config: VehicleConfiguration,
) -> None:
    # Nominal singularity: atan(1.9 / 0.4) = 1.363 rad.
    replace(
        vehicle_config,
        limits=replace(vehicle_config.limits, steering_angle_abs_max=1.3),
    )
    for limit in (1.37, 1.5, 1.6):
        with pytest.raises(ConfigurationError, match="Ackermann"):
            replace(
                vehicle_config,
                limits=replace(vehicle_config.limits, steering_angle_abs_max=limit),
            )


@pytest.mark.parametrize(
    ("measured", "requested", "expected"),
    [(0.3, 0.2, 0.2), (0.3, 5.0, 0.2), (-0.3, -0.2, -0.2), (-0.3, -5.0, -0.2)],
)
def test_supervised_speed_never_leaves_the_speed_limits(
    vehicle_config: VehicleConfiguration,
    measured: float,
    requested: float,
    expected: float,
) -> None:
    state = vehicle_config.geometry.assembled_state(
        Pose2D(0.0, 0.0, 0.0), 0.0, longitudinal_speed=measured
    )
    assessment = supervise_command(
        VehicleCommand(0.0, requested),
        state,
        vehicle_config.limits,
        vehicle_config.timing,
    )
    assert assessment.applied.target_speed == pytest.approx(expected)


@pytest.mark.parametrize("value", [0.1 / 30.0, 1.0 / 3.0, 0.002, 2.5, -0.553, 1e-12])
def test_mjcf_numbers_round_trip_exactly(value: float) -> None:
    assert float(_n(value)) == value
    assert _n(0.0) == "0"
    assert _n(-0.0) == "0"
    assert _n(0.30) == "0.3"
