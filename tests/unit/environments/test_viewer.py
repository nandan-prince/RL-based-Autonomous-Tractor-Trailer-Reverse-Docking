"""Keyboard-to-action mapping tests for the interactive 2D viewer."""

import numpy as np

from trailer_docking.environments.viewer import KeyboardControl, main


def test_keyboard_control_maps_speed_steering_stop_and_reset(vehicle_config) -> None:
    control = KeyboardControl(vehicle_config.limits, 0.1)
    control.press("down")
    control.press("left")
    action = control.action(0.0)
    np.testing.assert_array_equal(action, np.array([1.0, -1.0], dtype=np.float32))
    control.press("space")
    assert control.action(0.05)[1] == 0.0
    control.press("right")
    assert control.desired_steering == 0.0
    control.press("r")
    assert control.reset_requested
    control.clear()
    assert not control.reset_requested
    assert control.desired_speed == 0.0


def test_headless_viewer_smoke(monkeypatch) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    assert main(["--duration", "0"]) == 0
