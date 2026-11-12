"""Headless checks of the keyboard teleoperation logic and top-down drawing."""

from pathlib import Path

import pytest

from trailer_docking.core.config import DockConfiguration, VehicleConfiguration
from trailer_docking.core.state import Pose2D
from trailer_docking.visualization.viewer import (
    KEY_BACKSPACE,
    KEY_DOWN,
    KEY_LEFT,
    KEY_RIGHT,
    KEY_SPACE,
    KEY_UP,
    STEERING_STEP,
    Teleop,
    main,
)


def test_teleop_maps_keys_to_bounded_commands(
    vehicle_config: VehicleConfiguration,
) -> None:
    limits = vehicle_config.limits
    teleop = Teleop(limits, vehicle_config.timing.controller_period)
    assert teleop.command(0.0).to_dict() == {"steering_rate": 0.0, "target_speed": 0.0}

    teleop.press(KEY_DOWN)
    assert teleop.command(0.0).target_speed == -limits.reverse_speed_abs_max
    teleop.press(KEY_UP)
    assert teleop.command(0.0).target_speed == limits.forward_speed_max
    teleop.press(KEY_SPACE)
    assert teleop.command(0.0).target_speed == 0.0

    # Left is positive steering; the rate is limited and stops at the target.
    teleop.press(KEY_LEFT)
    assert teleop.desired_steering == pytest.approx(STEERING_STEP)
    assert teleop.command(0.0).steering_rate == limits.steering_rate_abs_max
    assert teleop.command(STEERING_STEP).steering_rate == pytest.approx(0.0)
    assert teleop.command(STEERING_STEP - 0.01).steering_rate == pytest.approx(0.1)
    for _ in range(100):
        teleop.press(KEY_RIGHT)
    assert teleop.desired_steering == -limits.steering_angle_abs_max

    teleop.press(-1)  # unmapped keys are ignored
    assert not teleop.reset_requested
    teleop.press(KEY_BACKSPACE)
    assert teleop.reset_requested
    teleop.clear()
    assert (teleop.desired_speed, teleop.desired_steering) == (0.0, 0.0)
    assert not teleop.reset_requested


def test_viewer_rejects_non_positive_speed() -> None:
    with pytest.raises(SystemExit):
        main(["--speed", "0"])


def test_topdown_view_draws_shared_geometry(
    tmp_path: Path,
    vehicle_config: VehicleConfiguration,
    dock_config: DockConfiguration,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from trailer_docking.visualization.topdown import TopDownView

    view = TopDownView(vehicle_config, dock_config)
    geometry = vehicle_config.geometry
    state = geometry.assembled_state(Pose2D(15.0, 0.5, 0.2), -0.1, steering_angle=0.3)
    view.update(state, note="note line")
    view.update(state)

    tractor = [float(value) for point in view._tractor.get_xy()[:4] for value in point]
    expected = [
        value
        for corner in geometry.tractor.footprint(state.tractor_pose)
        for value in (corner.x, corner.y)
    ]
    assert tractor == pytest.approx(expected)
    docking = geometry.trailer.rear_face(state.trailer_pose)
    assert view._docking_point.get_xdata()[0] == pytest.approx(docking.x)
    assert len(view._trail_points) == 2
    assert "dock error" in view._readout.get_text()
    view.clear_trail()
    assert view._trail_points == []

    output = tmp_path / "topdown.png"
    view.draw().savefig(output)
    assert output.stat().st_size > 0

    yard = TopDownView(vehicle_config)
    yard.update(geometry.neutral_state())
    assert "dock error" not in yard._readout.get_text()
