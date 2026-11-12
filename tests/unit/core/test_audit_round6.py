"""Regression tests for defects found by the sixth repository audit."""

from dataclasses import replace
from fractions import Fraction

import pytest

from trailer_docking.core.config import VehicleConfiguration
from trailer_docking.core.geometry import (
    Bounds1D,
    GeometryValidationError,
    WheelGeometry,
)
from trailer_docking.safety.supervisor import MeasurementTolerance


def test_geometry_stores_real_numbers_as_floats(
    vehicle_config: VehicleConfiguration,
) -> None:
    trailer = replace(vehicle_config.trailer, axle_track=Fraction(4, 5))
    assert type(trailer.axle_track) is float
    assert trailer.axle_track == 0.8
    changed = replace(vehicle_config, trailer=trailer)
    # Fingerprinting serializes the geometry, so it needs plain floats.
    assert changed.effective_fingerprint == vehicle_config.effective_fingerprint
    assert changed.manifest_summary()["trailer_variant"] == "nominal"
    wheels = WheelGeometry(radius=1, width=Fraction(1, 8))
    assert (type(wheels.radius), type(wheels.width)) == (float, float)
    bounds = Bounds1D(0, 1)
    assert (type(bounds.minimum), type(bounds.maximum)) == (float, float)


def test_geometry_rejects_non_numbers_with_validation_errors(
    vehicle_config: VehicleConfiguration,
) -> None:
    tractor, trailer = vehicle_config.tractor, vehicle_config.trailer
    for build in (
        lambda: replace(tractor, body_bottom_height="0.3"),
        lambda: replace(tractor, hitch_longitudinal=None),
        lambda: replace(tractor, hitch_lateral=True),
        lambda: replace(trailer, body_bottom_height="0.75"),
        lambda: replace(trailer, body_width=True),
        lambda: WheelGeometry(Fraction(10**400), 0.1),
        lambda: WheelGeometry(float("nan"), 0.1),
    ):
        with pytest.raises(GeometryValidationError):
            build()


def test_measurement_tolerance_accepts_reals_and_rejects_booleans() -> None:
    tolerance = MeasurementTolerance(speed=Fraction(1, 100))
    assert type(tolerance.speed) is float and tolerance.speed == 0.01
    with pytest.raises(ValueError):
        MeasurementTolerance(speed=True)
    with pytest.raises(ValueError):
        MeasurementTolerance(steering_rate="0.1")  # type: ignore[arg-type]
