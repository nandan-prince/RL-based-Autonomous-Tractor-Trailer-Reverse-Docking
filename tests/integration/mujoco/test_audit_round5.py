"""Regression tests for defects found by the fifth repository audit."""

from __future__ import annotations

from pathlib import Path

import pytest

from trailer_docking.core.config import (
    VehicleConfiguration,
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


def test_reset_accepts_numpy_scalars(vehicle_config: VehicleConfiguration) -> None:
    simulator = MujocoSimulator(vehicle_config)
    state = simulator.reset(steering_angle=np.float32(0.1), settle_time=np.float32(0.0))
    assert state.steering_angle == pytest.approx(0.1, abs=1e-6)
    with pytest.raises(ValueError):
        simulator.reset(steering_angle=True)


def test_viewer_reports_unsupported_specifications_cleanly(
    tmp_path: Path, vehicle_config: VehicleConfiguration
) -> None:
    from trailer_docking.visualization import viewer

    text = vehicle_config.source_path.read_text(encoding="utf-8")
    assert "  nominal_command_delay: 0.00\n" in text
    delayed = tmp_path / "delayed.yaml"
    delayed.write_text(
        text.replace("nominal_command_delay: 0.00", "nominal_command_delay: 0.10"),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="cannot build the MuJoCo scene"):
        viewer.main(["--vehicle", str(delayed), "--no-dock", "--mode", "2d"])
