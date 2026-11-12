"""Core regression tests from repository audits that need no simulator."""

from dataclasses import replace
from pathlib import Path

import pytest

from trailer_docking.core.config import (
    ConfigurationError,
    VehicleConfiguration,
    load_vehicle_configuration,
)
from trailer_docking.core.geometry import GeometryValidationError, VehicleGeometry
from trailer_docking.safety.supervisor import (
    MeasurementTolerance,
    state_limit_violations,
)
from trailer_docking.validation import vehicle_spec


def test_every_trailer_variant_is_validated_at_load(
    tmp_path: Path, vehicle_config: VehicleConfiguration
) -> None:
    text = vehicle_config.source_path.read_text(encoding="utf-8")
    original = (
        "  short:\n    hitch_to_axle: 2.00\n    front_overhang: 0.50\n"
        "    rear_overhang: 1.00\n"
    )
    assert original in text
    changed = original.replace("rear_overhang: 1.00", "rear_overhang: 0.20")
    path = tmp_path / "bad_variant.yaml"
    path.write_text(text.replace(original, changed), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="short"):
        load_vehicle_configuration(path)


def test_yaml_merge_keys_are_supported_and_duplicates_still_rejected(
    tmp_path: Path, vehicle_config: VehicleConfiguration
) -> None:
    text = vehicle_config.source_path.read_text(encoding="utf-8")
    original = (
        "  short:\n    hitch_to_axle: 2.00\n    front_overhang: 0.50\n"
        "    rear_overhang: 1.00\n"
        "  nominal:\n    hitch_to_axle: 4.00\n    front_overhang: 0.50\n"
        "    rear_overhang: 1.00\n"
    )
    assert original in text
    merged = (
        "  short: &base\n    hitch_to_axle: 2.00\n    front_overhang: 0.50\n"
        "    rear_overhang: 1.00\n"
        "  nominal:\n    <<: *base\n    hitch_to_axle: 4.00\n"
    )
    path = tmp_path / "merge.yaml"
    path.write_text(text.replace(original, merged), encoding="utf-8")
    loaded = load_vehicle_configuration(path)
    assert loaded.trailer_variant("nominal").hitch_to_axle == 4.0
    assert loaded.trailer_variant("nominal").rear_overhang == 1.0

    duplicate = tmp_path / "merge_duplicate.yaml"
    duplicate.write_text(
        text.replace(original, merged + "    hitch_to_axle: 4.50\n"), encoding="utf-8"
    )
    with pytest.raises(ConfigurationError, match="duplicate key"):
        load_vehicle_configuration(duplicate)


def test_measurement_tolerance_never_relaxes_the_jackknife_threshold(
    vehicle_config: VehicleConfiguration,
) -> None:
    state = vehicle_config.geometry.assembled_state(
        vehicle_config.geometry.neutral_state().tractor_pose,
        -vehicle_config.limits.articulation_hard_abs_max,
        steering_angle=0.502,
    )
    loose = MeasurementTolerance(steering_angle=0.005, steering_rate=1.0, speed=1.0)
    assert state_limit_violations(state, vehicle_config.limits, loose) == (
        "jackknife_limit",
    )
    assert "steering_angle_limit" in state_limit_violations(
        state, vehicle_config.limits
    )
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            MeasurementTolerance(speed=bad)


# YAML 1.1 only reads exponent notation as a float when it has a decimal point.
@pytest.mark.parametrize("step", ["1.0e-320", "1.0e-9", "1.0e-300", "0.0000001"])
@pytest.mark.parametrize("field", ["mujoco_physics_step", "kinematic_integration_step"])
def test_absurd_integration_steps_are_configuration_errors(
    tmp_path: Path, vehicle_config: VehicleConfiguration, field: str, step: str
) -> None:
    text = vehicle_config.source_path.read_text(encoding="utf-8")
    original = {
        "mujoco_physics_step": "  mujoco_physics_step: 0.002\n",
        "kinematic_integration_step": "  kinematic_integration_step: 0.02\n",
    }[field]
    assert original in text
    path = tmp_path / "tiny_step.yaml"
    path.write_text(text.replace(original, f"  {field}: {step}\n"), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="too small"):
        load_vehicle_configuration(path)
    assert vehicle_spec.main(["--vehicle", str(path)]) == 2


def test_wheels_must_lie_inside_the_collision_footprints(
    vehicle_config: VehicleConfiguration,
) -> None:
    geometry = vehicle_config.geometry
    cases = (
        {"trailer": replace(geometry.trailer, axle_track=1.6)},
        {"tractor": replace(geometry.tractor, front_track=0.95)},
        {"tractor": replace(geometry.tractor, rear_track=0.95)},
        {"wheels": replace(geometry.wheels, width=0.3)},
        {"wheels": replace(geometry.wheels, radius=0.45)},
    )
    for change in cases:
        with pytest.raises(GeometryValidationError, match="footprints"):
            replace(geometry, **change)  # type: ignore[arg-type]
    # Exactly flush wheels are allowed.
    flush = replace(geometry, wheels=replace(geometry.wheels, width=0.2))
    assert isinstance(flush, VehicleGeometry)
