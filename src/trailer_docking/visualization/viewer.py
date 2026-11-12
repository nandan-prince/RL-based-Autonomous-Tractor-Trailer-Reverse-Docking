"""Interactive local viewers for the generated dock and vehicle.

Run from the repository root::

    python -m trailer_docking.visualization.viewer --mode both

``--mode 3d`` opens the MuJoCo window, ``--mode 2d`` the top-down plot, and
``--mode both`` shows the two side by side, driven by the same simulator and
the same shared state.  Click a window to give it keyboard focus, then:

    Up / Down        drive forward / reverse at the speed limit
    Left / Right     steer left / right in small steps
    Space            stop (zero speed, hold steering)
    Backspace        reset to the start pose

``--demo`` replaces the keyboard with a scripted straight reverse into the bay.

The MuJoCo backend is the state source in every mode, because the project's 2D
kinematic simulator is not implemented yet; the top-down plot only consumes the
shared ``VehicleState`` and will work unchanged with that backend.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from trailer_docking.core.config import (
    VehicleLimits,
    load_dock_configuration,
    load_vehicle_configuration,
)
from trailer_docking.core.state import VehicleCommand, VehicleState

# GLFW key codes delivered by the MuJoCo viewer.
KEY_SPACE, KEY_BACKSPACE = 32, 259
KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP = 262, 263, 264, 265
MATPLOTLIB_KEYS = {
    "up": KEY_UP,
    "down": KEY_DOWN,
    "left": KEY_LEFT,
    "right": KEY_RIGHT,
    " ": KEY_SPACE,
    "space": KEY_SPACE,
    "backspace": KEY_BACKSPACE,
}
STEERING_STEP = 0.05


def _clip(value: float, minimum: float, maximum: float) -> float:
    return min(max(value, minimum), maximum)


@dataclass
class Teleop:
    """Keyboard intent: a desired speed and a desired virtual steering angle."""

    limits: VehicleLimits
    controller_period: float
    desired_speed: float = 0.0
    desired_steering: float = 0.0
    reset_requested: bool = False

    def press(self, key: int) -> None:
        limit = self.limits.steering_angle_abs_max
        if key == KEY_UP:
            self.desired_speed = self.limits.forward_speed_max
        elif key == KEY_DOWN:
            self.desired_speed = -self.limits.reverse_speed_abs_max
        elif key == KEY_SPACE:
            self.desired_speed = 0.0
        elif key == KEY_LEFT:
            self.desired_steering = _clip(
                self.desired_steering + STEERING_STEP, -limit, limit
            )
        elif key == KEY_RIGHT:
            self.desired_steering = _clip(
                self.desired_steering - STEERING_STEP, -limit, limit
            )
        elif key == KEY_BACKSPACE:
            self.reset_requested = True

    def clear(self) -> None:
        self.desired_speed = 0.0
        self.desired_steering = 0.0
        self.reset_requested = False

    def command(self, steering_target: float) -> VehicleCommand:
        """Steering rate that reaches the desired angle, within the rate limit."""

        rate_limit = self.limits.steering_rate_abs_max
        rate = (self.desired_steering - steering_target) / self.controller_period
        return VehicleCommand(_clip(rate, -rate_limit, rate_limit), self.desired_speed)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=("3d", "2d", "both"), default="both")
    parser.add_argument(
        "--vehicle",
        type=Path,
        default=Path("configs/vehicles/compact_semitrailer_v1.yaml"),
    )
    parser.add_argument(
        "--dock", type=Path, default=Path("configs/environments/nominal_dock_v1.yaml")
    )
    parser.add_argument(
        "--no-dock", action="store_true", help="show the vehicle on an empty yard"
    )
    parser.add_argument("--variant", choices=("short", "nominal", "long"))
    parser.add_argument(
        "--demo", action="store_true", help="scripted straight reverse into the bay"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="playback speed relative to real time (default 1)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="close automatically after this many simulated seconds",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.speed <= 0.0:
        raise SystemExit("--speed must be positive")

    # Imported here so --help works without the optional dependencies.
    from trailer_docking.simulators.mujoco.adapter import MujocoSimulator

    try:
        vehicle = load_vehicle_configuration(args.vehicle)
        if args.variant:
            vehicle = vehicle.with_trailer_variant(args.variant)
        dock = None if args.no_dock else load_dock_configuration(args.dock)
    except ValueError as error:
        raise SystemExit(
            f"error: {error} (run from the repository root or pass paths)"
        ) from error
    try:
        simulator = MujocoSimulator(vehicle, dock)
    except (ValueError, NotImplementedError, RuntimeError) as error:
        # A specification can be valid and still not be simulable by this
        # backend, for example a nonzero command delay.
        raise SystemExit(f"error: cannot build the MuJoCo scene: {error}") from error
    teleop = Teleop(vehicle.limits, simulator.controller_period)

    topdown = None
    plt = None
    if args.mode in ("2d", "both"):
        import matplotlib.pyplot as plt

        from trailer_docking.visualization.topdown import TopDownView

        topdown = TopDownView(vehicle, dock)
        # Keep matplotlib's own shortcuts (e.g. left/right = history) out of
        # the way of the driving keys.
        for name in ("keymap.back", "keymap.forward", "keymap.home"):
            plt.rcParams[name] = []
        topdown.figure.canvas.mpl_connect(
            "key_press_event",
            lambda event: teleop.press(MATPLOTLIB_KEYS.get(event.key, -1)),
        )
        plt.show(block=False)

    viewer = None
    if args.mode in ("3d", "both"):
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(
            simulator.model, simulator.data, key_callback=teleop.press
        )
        viewer.cam.lookat[:] = (9.0 if dock is not None else 0.0, 0.0, 0.5)
        viewer.cam.distance = 24.0
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -35.0

    print(__doc__)
    state: VehicleState = simulator.state
    try:
        while True:
            started = time.perf_counter()
            if viewer is not None and not viewer.is_running():
                break
            if (
                topdown is not None
                and plt is not None
                and not plt.fignum_exists(topdown.figure.number)
            ):
                break
            if args.duration is not None and state.simulation_time >= args.duration:
                break

            if teleop.reset_requested:
                teleop.clear()
                if topdown is not None:
                    topdown.clear_trail()
                if viewer is not None:
                    with viewer.lock():
                        state = simulator.reset()
                else:
                    state = simulator.reset()
            if args.demo:
                command = VehicleCommand(0.0, -vehicle.limits.reverse_speed_abs_max)
                if simulator.wall_contacts():
                    command = VehicleCommand(0.0, 0.0)
            else:
                command = teleop.command(simulator.steering_target)

            if viewer is not None:
                with viewer.lock():
                    state = simulator.step(command)
                viewer.sync()
            else:
                state = simulator.step(command)

            if topdown is not None and plt is not None:
                contacts = simulator.wall_contacts()
                note = f"WALL CONTACT: {', '.join(contacts)}" if contacts else ""
                topdown.update(state, note=note)
                topdown.figure.canvas.draw_idle()
                topdown.figure.canvas.flush_events()

            remaining = simulator.controller_period / args.speed - (
                time.perf_counter() - started
            )
            if remaining > 0.0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        pass
    finally:
        if viewer is not None:
            viewer.close()
        if plt is not None:
            plt.close("all")
    return 0


if __name__ == "__main__":  # pragma: no cover - interactive entry point
    raise SystemExit(main())
