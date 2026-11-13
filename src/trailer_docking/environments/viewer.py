"""Interactive keyboard viewer for the Gymnasium 2D docking environment.

Run from the repository root::

    .venv/bin/python -m trailer_docking.environments.viewer

Controls:

    Up / Down       forward / reverse speed
    Left / Right    increase / decrease desired steering angle
    Space           stop while holding steering
    R / Backspace   reset the episode
    Escape          close the window
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from trailer_docking.core.config import VehicleLimits
from trailer_docking.environments import TrailerDockingEnv

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SCENARIOS = {
    "compact": (
        REPOSITORY_ROOT / "configs/vehicles/compact_semitrailer_v1.yaml",
        REPOSITORY_ROOT / "configs/environments/nominal_dock_v1.yaml",
        REPOSITORY_ROOT / "configs/environments/gymnasium_docking_v1.yaml",
    ),
    "service-center": (
        REPOSITORY_ROOT / "configs/vehicles/full_size_semitrailer_v1.yaml",
        REPOSITORY_ROOT / "configs/environments/service_center_dock_v1.yaml",
        REPOSITORY_ROOT / "configs/environments/service_center_gymnasium_v1.yaml",
    ),
}


def _clip(value: float, minimum: float, maximum: float) -> float:
    return min(max(value, minimum), maximum)


@dataclass(slots=True)
class KeyboardControl:
    """Persistent desired steering and speed translated to normalized actions."""

    limits: VehicleLimits
    controller_period: float
    desired_speed: float = 0.0
    desired_steering: float = 0.0
    reset_requested: bool = False

    def press(self, key: str | None) -> None:
        steering_limit = self.limits.steering_angle_abs_max
        if key == "up":
            self.desired_speed = self.limits.forward_speed_max
        elif key == "down":
            self.desired_speed = -self.limits.reverse_speed_abs_max
        elif key in (" ", "space"):
            self.desired_speed = 0.0
        elif key == "left":
            self.desired_steering = _clip(
                self.desired_steering + 0.05, -steering_limit, steering_limit
            )
        elif key == "right":
            self.desired_steering = _clip(
                self.desired_steering - 0.05, -steering_limit, steering_limit
            )
        elif key in ("r", "backspace"):
            self.reset_requested = True

    def clear(self) -> None:
        self.desired_speed = 0.0
        self.desired_steering = 0.0
        self.reset_requested = False

    def action(self, steering_angle: float) -> NDArray[np.float32]:
        rate = (self.desired_steering - steering_angle) / self.controller_period
        normalized_rate = _clip(rate / self.limits.steering_rate_abs_max, -1.0, 1.0)
        speed_limit = (
            self.limits.forward_speed_max
            if self.desired_speed >= 0.0
            else self.limits.reverse_speed_abs_max
        )
        normalized_speed = self.desired_speed / speed_limit
        return np.array([normalized_rate, normalized_speed], dtype=np.float32)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--scenario",
        choices=tuple(SCENARIOS),
        default="service-center",
        help="visual scenario (default: service-center)",
    )
    parser.add_argument("--variant", choices=("short", "nominal", "long"))
    parser.add_argument("--demo", action="store_true", help="straight reverse demo")
    parser.add_argument(
        "--random-start", action="store_true", help="seeded lateral/yaw perturbation"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--speed", type=float, default=1.0, help="playback speed relative to real time"
    )
    parser.add_argument(
        "--duration", type=float, help="close after this many simulated seconds"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.speed <= 0.0:
        raise SystemExit("--speed must be positive")
    vehicle_path, dock_path, task_path = SCENARIOS[args.scenario]
    env = TrailerDockingEnv(
        vehicle_path=vehicle_path,
        dock_path=dock_path,
        task_path=task_path,
        trailer_variant=args.variant,
        action_mode="teleop",
        render_mode="human",
    )
    env.reset(seed=args.seed, options={"randomize": args.random_start})
    control = KeyboardControl(env.vehicle.limits, env.simulator.controller_period)
    figure = env.figure
    if figure is None:  # pragma: no cover - human mode always creates a figure
        raise RuntimeError("human renderer did not create a figure")
    figure.canvas.mpl_connect("key_press_event", lambda event: control.press(event.key))

    import matplotlib.pyplot as plt

    print(__doc__)
    ended = False
    try:
        while plt.fignum_exists(figure.number):
            started = time.perf_counter()
            if args.duration is not None and env.state.simulation_time >= args.duration:
                break
            if control.reset_requested:
                control.clear()
                env.reset(seed=args.seed, options={"randomize": args.random_start})
                ended = False
            if not ended:
                action = (
                    np.array([0.0, -1.0], dtype=np.float32)
                    if args.demo
                    else control.action(env.state.steering_angle)
                )
                _, _, terminated, truncated, info = env.step(action)
                ended = terminated or truncated
                if ended:
                    print(
                        f"Episode ended: {info['termination_reason']}; press R to reset"
                    )
            figure.canvas.flush_events()
            remaining = env.simulator.controller_period / args.speed - (
                time.perf_counter() - started
            )
            if remaining > 0.0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
    return 0


if __name__ == "__main__":  # pragma: no cover - interactive entry point
    raise SystemExit(main())
