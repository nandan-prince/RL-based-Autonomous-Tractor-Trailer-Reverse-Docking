"""Tests closing validation and coverage gaps found by the eighth audit."""

from pathlib import Path

import pytest

from trailer_docking.core.config import (
    ConfigurationError,
    VehicleConfiguration,
    load_vehicle_configuration,
)
from trailer_docking.core.state import Pose2D
from trailer_docking.safety.supervisor import state_limit_violations

# Realistic single-line mistakes in the vehicle YAML, each caught by a
# different validation guard.  ``match`` pins the guard that must fire.
YAML_MISTAKES = (
    (
        "negative limit copied from the nominal target",
        "  reverse_speed_abs_max: 0.20\n",
        "  reverse_speed_abs_max: -0.20\n",
        "must be finite and positive",
    ),
    (
        "changed sign convention",
        "  positive_steering: left\n",
        "  positive_steering: right\n",
        "positive_steering",
    ),
    (
        "tractor footprint no longer matches the overhangs",
        "    x: [-0.40, 2.50]\n",
        "    x: [-0.40, 2.80]\n",
        "tractor collision footprint x",
    ),
    (
        "tractor footprint no longer matches the body width",
        "  body_width: 1.00\n  front_track: 0.80\n",
        "  body_width: 1.10\n  front_track: 0.80\n",
        "tractor collision footprint y",
    ),
    (
        "trailer footprint no longer matches the axle distance",
        "    x: [-1.00, 4.50]\n",
        "    x: [-1.00, 4.90]\n",
        "trailer collision footprint x",
    ),
    (
        "nominal speed outside the speed limits",
        "  nominal_target_speed: -0.20\n",
        "  nominal_target_speed: -0.50\n",
        "nominal target speed exceeds",
    ),
    (
        "gravity written without its sign",
        "  gravity: [0.0, 0.0, -9.81]\n",
        "  gravity: [0.0, 0.0, 9.81]\n",
        "gravity must point down",
    ),
    (
        "frictionless ground",
        "  contact_friction: [1.0, 0.005, 0.0001]\n",
        "  contact_friction: [0.0, 0.005, 0.0001]\n",
        "sliding friction must be positive",
    ),
)


@pytest.mark.parametrize(
    ("original", "changed", "message"),
    [mistake[1:] for mistake in YAML_MISTAKES],
    ids=[mistake[0] for mistake in YAML_MISTAKES],
)
def test_realistic_yaml_mistakes_are_rejected_by_the_intended_guard(
    tmp_path: Path,
    vehicle_config: VehicleConfiguration,
    original: str,
    changed: str,
    message: str,
) -> None:
    text = vehicle_config.source_path.read_text(encoding="utf-8")
    assert text.count(original) == 1, original
    path = tmp_path / "mistake.yaml"
    path.write_text(text.replace(original, changed), encoding="utf-8")
    with pytest.raises(ConfigurationError, match=message):
        load_vehicle_configuration(path)


def test_selected_trailer_must_match_its_named_variant(
    tmp_path: Path, vehicle_config: VehicleConfiguration
) -> None:
    text = vehicle_config.source_path.read_text(encoding="utf-8")
    # Lengthen the selected trailer consistently (axle distance and footprint)
    # without updating the "nominal" variant it claims to be.
    for original, changed in (
        ("\n  hitch_to_axle: 4.00\n", "\n  hitch_to_axle: 4.20\n"),
        ("    x: [-1.00, 4.50]\n", "    x: [-1.00, 4.70]\n"),
    ):
        assert text.count(original) == 1, original
        text = text.replace(original, changed)
    path = tmp_path / "mismatch.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigurationError, match="does not match variant"):
        load_vehicle_configuration(path)


def test_effective_fingerprint_distinguishes_trailer_variants(
    vehicle_config: VehicleConfiguration,
) -> None:
    fingerprints = {
        name: vehicle_config.with_trailer_variant(name).effective_fingerprint
        for name in ("short", "nominal", "long")
    }
    assert len(set(fingerprints.values())) == 3
    assert fingerprints["nominal"] == vehicle_config.effective_fingerprint
    # The source file is the same, so only the effective fingerprint differs.
    assert (
        vehicle_config.with_trailer_variant("short").source_fingerprint
        == vehicle_config.source_fingerprint
    )


def test_each_speed_limit_violation_is_reported_separately(
    vehicle_config: VehicleConfiguration,
) -> None:
    geometry, limits = vehicle_config.geometry, vehicle_config.limits
    forward = geometry.assembled_state(
        Pose2D(0.0, 0.0, 0.0), 0.0, longitudinal_speed=limits.forward_speed_max + 0.1
    )
    reverse = geometry.assembled_state(
        Pose2D(0.0, 0.0, 0.0),
        0.0,
        longitudinal_speed=-limits.reverse_speed_abs_max - 0.1,
    )
    assert state_limit_violations(forward, limits) == ("forward_speed_limit",)
    assert state_limit_violations(reverse, limits) == ("reverse_speed_limit",)
    at_limit = geometry.assembled_state(
        Pose2D(0.0, 0.0, 0.0), 0.0, longitudinal_speed=limits.forward_speed_max
    )
    assert state_limit_violations(at_limit, limits) == ()
