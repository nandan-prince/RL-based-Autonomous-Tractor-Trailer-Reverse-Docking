"""Regression tests for defects found by the fifth repository audit."""

from __future__ import annotations

from pathlib import Path

import pytest

from trailer_docking.core.config import (
    ConfigurationError,
    VehicleConfiguration,
    load_vehicle_configuration,
)
from trailer_docking.core.state import Pose2D, VehicleCommand
from trailer_docking.safety.supervisor import state_limit_violations

mujoco = pytest.importorskip("mujoco")
np = pytest.importorskip("numpy")

from trailer_docking.simulators.mujoco.adapter import (  # noqa: E402
    MUJOCO_MEASUREMENT_TOLERANCE,
    MujocoSimulator,
)


@pytest.mark.parametrize("variant", ["short", "nominal", "long"])
@pytest.mark.parametrize("period", [1, 8, 20, 40])
def test_published_tolerance_covers_lock_to_lock_sweeps(
    vehicle_config: VehicleConfiguration, variant: str, period: int
) -> None:
    vehicle = vehicle_config.with_trailer_variant(variant)
    simulator = MujocoSimulator(vehicle)
    simulator.reset(settle_time=0.5)
    for index in range(120):
        rate = 0.3 if (index // period) % 2 == 0 else -0.3
        speed = -0.2 if (index // 37) % 2 == 0 else 0.2
        state = simulator.step(VehicleCommand(rate, speed))
        violations = state_limit_violations(
            state, vehicle.limits, MUJOCO_MEASUREMENT_TOLERANCE
        )
        assert [name for name in violations if name != "jackknife_limit"] == []


def test_numpy_scalars_are_accepted_as_real_numbers(
    vehicle_config: VehicleConfiguration,
) -> None:
    command = VehicleCommand(np.float32(0.1), np.float32(0.0))
    assert isinstance(command.steering_rate, float)
    assert command.steering_rate == pytest.approx(0.1)
    assert VehicleCommand(np.int64(0), np.float64(-0.2)).target_speed == -0.2
    assert Pose2D(np.float32(1.0), 0, np.float32(0.5)).yaw == pytest.approx(0.5)
    action = np.array([0.5], dtype=np.float32)
    simulator = MujocoSimulator(vehicle_config)
    state = simulator.step(VehicleCommand(action[0] * 0.3, -0.2))
    assert state.simulation_time == pytest.approx(0.1)
    with pytest.raises(ValueError):
        VehicleCommand(np.bool_(True), 0.0)  # type: ignore[arg-type]


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
