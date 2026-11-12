"""Regression tests for defects found by the third repository audit."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from trailer_docking.core.config import DockConfiguration, VehicleConfiguration
from trailer_docking.core.state import Pose2D, VehicleCommand
from trailer_docking.safety.supervisor import supervise_command
from trailer_docking.simulators.mujoco import builder
from trailer_docking.validation import vehicle_spec

mujoco = pytest.importorskip("mujoco")

from trailer_docking.simulators.mujoco.adapter import (  # noqa: E402
    MujocoSimulator,
)


def test_low_dock_walls_still_stop_and_report_the_trailer(
    vehicle_config: VehicleConfiguration, dock_config: DockConfiguration
) -> None:
    # A wall lower than the trailer floor must not let the trailer pass over.
    assert vehicle_config.trailer.body_bottom_height > 0.5
    low = replace(dock_config, geometry=replace(dock_config.geometry, wall_height=0.5))
    simulator = MujocoSimulator(vehicle_config, low)

    simulator.reset(Pose2D(8.0, 0.6, 0.0), 0.0)
    assert "trailer_body" in simulator.wall_contacts()

    simulator.reset(Pose2D(5.3, 0.0, 0.0), 0.0, settle_time=0.3)
    assert simulator.wall_contacts() == ()
    for _ in range(60):
        state = simulator.step(VehicleCommand(0.0, -0.2))
    assert simulator.wall_contacts() == ("trailer_body",)
    rear = vehicle_config.trailer.rear_face(state.trailer_pose)
    assert rear.x == pytest.approx(0.0, abs=0.02)


def test_rejected_reset_leaves_the_episode_untouched(
    vehicle_config: VehicleConfiguration,
) -> None:
    simulator = MujocoSimulator(vehicle_config)
    simulator.reset()
    for _ in range(20):
        simulator.step(VehicleCommand(0.3, -0.2))
    before = simulator.state.to_dict()
    targets = (simulator.steering_target, simulator.speed_target)
    bad_calls = (
        {"tractor_pose": (1.0, 2.0, 3.0)},
        {"settle_time": float("nan")},
        {"settle_time": float("inf")},
        {"settle_time": "1"},
        {"tractor_pose": Pose2D(20.0, 0.0, 0.0), "trailer_yaw": 10**400},
        {"tractor_pose": Pose2D(20.0, 0.0, 0.0), "trailer_yaw": float("nan")},
        {"steering_angle": None},
    )
    for arguments in bad_calls:
        with pytest.raises(ValueError):
            simulator.reset(**arguments)  # type: ignore[arg-type]
        assert simulator.state.to_dict() == before
        assert (simulator.steering_target, simulator.speed_target) == targets


@pytest.mark.parametrize(
    ("steering_angle", "speed", "command"),
    [
        (0.0, 0.3, VehicleCommand(0.0, 0.2)),
        (0.6, 0.0, VehicleCommand(-0.3, 0.0)),
        (-0.6, -0.3, VehicleCommand(0.3, -0.2)),
        (0.0, 0.0, VehicleCommand(0.1, 0.02)),
    ],
)
def test_no_intervention_is_recorded_when_the_command_is_unchanged(
    vehicle_config: VehicleConfiguration,
    steering_angle: float,
    speed: float,
    command: VehicleCommand,
) -> None:
    state = vehicle_config.geometry.assembled_state(
        Pose2D(0.0, 0.0, 0.0),
        0.0,
        steering_angle=steering_angle,
        longitudinal_speed=speed,
    )
    assessment = supervise_command(
        command, state, vehicle_config.limits, vehicle_config.timing
    )
    assert assessment.applied == command
    assert assessment.reasons == ()
    assert not assessment.intervened


def test_mjcf_comparison_tolerates_last_bit_noise_only() -> None:
    base = '<mujoco><geom name="a" size="0.45557737670447235 1"/></mujoco>'
    noisy = '<mujoco><geom name="a" size="0.4555773767044724 1"/></mujoco>'
    assert builder.mjcf_equivalent(base, noisy)
    assert not builder.mjcf_equivalent(
        base, '<mujoco><geom name="a" size="0.4556 1"/></mujoco>'
    )
    assert not builder.mjcf_equivalent(
        base, '<mujoco><geom name="b" size="0.45557737670447235 1"/></mujoco>'
    )
    assert not builder.mjcf_equivalent(
        base, '<mujoco><geom name="a" size="0.45557737670447235"/></mujoco>'
    )
    assert not builder.mjcf_equivalent(base, "<mujoco><geom")
    assert not builder.mjcf_equivalent(base, "<mujoco/>")


def test_command_line_tools_report_missing_configuration(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = str(tmp_path / "missing.yaml")
    assert vehicle_spec.main(["--vehicle", missing]) == 2
    assert "error:" in capsys.readouterr().out
    assert builder.main(["--vehicle", missing]) == 2
    assert "error:" in capsys.readouterr().out
