"""Closed-loop controller runner shared by LQR, NMPC, and PPO adapters."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import pi
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from trailer_docking.controllers import (
    BasicNMPCController,
    Controller,
    ControllerContext,
    TrajectoryLQRController,
    validate_controller_action,
)
from trailer_docking.core.state import Point2D
from trailer_docking.environments import TrailerDockingEnv
from trailer_docking.environments.task import aligned_pose_for_docking_point
from trailer_docking.paths import ReferencePath, straight_reference

FrameCallback = Callable[[NDArray[np.uint8]], None]

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


@dataclass(frozen=True, slots=True)
class RolloutResult:
    """Compact outcome of one controller episode."""

    controller: str
    steps: int
    total_reward: float
    terminated: bool
    truncated: bool
    termination_reason: str | None
    success: bool
    collision: bool
    jackknifed: bool
    maximum_articulation: float
    final_lateral_error: float
    final_heading_error: float
    controller_compute_seconds: float
    solve_failures: int

    def to_dict(self) -> dict[str, str | int | float | bool | None]:
        return {
            "controller": self.controller,
            "steps": self.steps,
            "total_reward": self.total_reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "termination_reason": self.termination_reason,
            "success": self.success,
            "collision": self.collision,
            "jackknifed": self.jackknifed,
            "maximum_articulation": self.maximum_articulation,
            "final_lateral_error": self.final_lateral_error,
            "final_heading_error": self.final_heading_error,
            "controller_compute_seconds": self.controller_compute_seconds,
            "solve_failures": self.solve_failures,
        }


def run_episode(
    environment: TrailerDockingEnv,
    controller: Controller,
    *,
    seed: int = 0,
    reset_options: Mapping[str, Any] | None = None,
    realtime_factor: float | None = None,
    frame_callback: FrameCallback | None = None,
    frame_stride: int = 1,
) -> RolloutResult:
    """Run one episode through the common normalized steering-rate contract."""

    if environment.action_mode != "fixed_reverse":
        raise ValueError("controller evaluation requires action_mode='fixed_reverse'")
    if realtime_factor is not None and realtime_factor <= 0.0:
        raise ValueError("realtime_factor must be positive or None")
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    if frame_callback is not None and environment.render_mode != "rgb_array":
        raise ValueError("frame capture requires render_mode='rgb_array'")
    display_name = str(getattr(controller, "display_name", controller.name))
    environment.set_controller_label(display_name)
    observation, info = environment.reset(seed=seed, options=reset_options)
    controller.reset()
    if frame_callback is not None:
        initial_frame = environment.render()
        if initial_frame is None:
            raise RuntimeError("rgb_array renderer did not return an initial frame")
        frame_callback(initial_frame)
    total_reward = 0.0
    maximum_articulation = abs(environment.state.articulation_angle)
    compute_seconds = 0.0
    terminated = False
    truncated = False
    while not (terminated or truncated):
        context = ControllerContext(
            observation=observation,
            state=environment.state,
            path=environment.path,
            tracking=environment.tracking,
            vehicle=environment.vehicle,
            step_index=environment.episode_steps,
        )
        started = perf_counter()
        action = validate_controller_action(controller.act(context))
        compute_seconds += perf_counter() - started
        observation, reward, terminated, truncated, info = environment.step(action)
        total_reward += reward
        maximum_articulation = max(
            maximum_articulation, abs(environment.state.articulation_angle)
        )
        if frame_callback is not None and (
            environment.episode_steps % frame_stride == 0 or terminated or truncated
        ):
            frame = environment.render()
            if frame is None:
                raise RuntimeError("rgb_array renderer did not return a frame")
            frame_callback(frame)
        if realtime_factor is not None:
            import matplotlib.pyplot as plt

            plt.pause(environment.simulator.controller_period / realtime_factor)
    solve_failures = int(getattr(controller, "solve_failures", 0))
    return RolloutResult(
        controller=controller.name,
        steps=environment.episode_steps,
        total_reward=total_reward,
        terminated=terminated,
        truncated=truncated,
        termination_reason=info["termination_reason"],
        success=bool(info["is_success"]),
        collision=bool(info["collision"]),
        jackknifed=bool(info["jackknifed"]),
        maximum_articulation=maximum_articulation,
        final_lateral_error=environment.tracking.lateral_error,
        final_heading_error=environment.tracking.trailer_heading_error,
        controller_compute_seconds=compute_seconds,
        solve_failures=solve_failures,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a basic closed-loop 2D controller demonstration"
    )
    parser.add_argument("--controller", choices=("lqr", "nmpc"), default="lqr")
    parser.add_argument("--scenario", choices=tuple(SCENARIOS), default="compact")
    parser.add_argument("--lateral-offset", type=float, default=0.35)
    parser.add_argument("--heading-deg", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speed", type=float, default=4.0)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--record", type=Path, help="write an H.264 MP4 rollout")
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="record every Nth controller step (default: 1)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.speed <= 0.0:
        raise SystemExit("--speed must be positive")
    if args.frame_stride <= 0:
        raise SystemExit("--frame-stride must be positive")
    vehicle_path, dock_path, task_path = SCENARIOS[args.scenario]
    render_mode: Literal["human", "rgb_array"] | None = (
        "rgb_array" if args.record is not None else (
        None if args.no_render else "human"
        )
    )
    environment = TrailerDockingEnv(
        vehicle_path=vehicle_path,
        dock_path=dock_path,
        task_path=task_path,
        action_mode="fixed_reverse",
        render_mode=render_mode,
    )
    start_x = (
        environment.dock.geometry.interior_depth
        + environment.settings.start_standoff
    )
    path: ReferencePath = straight_reference(
        environment.dock.geometry.target_position,
        Point2D(start_x, 0.0),
    )
    heading = args.heading_deg * pi / 180.0
    pose = aligned_pose_for_docking_point(
        environment.vehicle,
        Point2D(start_x, args.lateral_offset),
        heading,
    )
    if args.controller == "lqr":
        controller: Controller = TrajectoryLQRController(environment.vehicle)
    else:
        controller = BasicNMPCController(environment.vehicle)
    recorder: Any | None = None
    try:
        if args.record is not None:
            from trailer_docking.evaluation.video import MP4Recorder

            source_fps = float(environment.metadata["render_fps"])
            output_fps = source_fps * args.speed / args.frame_stride
            recorder = MP4Recorder(args.record, fps=output_fps)
        result = run_episode(
            environment,
            controller,
            seed=args.seed,
            reset_options={
                "tractor_pose": pose,
                "trailer_yaw": heading,
                "reference_path": path,
            },
            realtime_factor=(
                args.speed if args.record is None and not args.no_render else None
            ),
            frame_callback=None if recorder is None else recorder.append,
            frame_stride=args.frame_stride,
        )
        if recorder is not None:
            recorder.hold_last(1.0)
    finally:
        if recorder is not None:
            recorder.close()
        environment.close()
    payload = result.to_dict()
    if recorder is not None:
        payload["video_path"] = str(recorder.path)
        payload["video_frames"] = recorder.frame_count
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if result.success else 1


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
