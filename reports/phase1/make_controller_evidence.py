"""Generate Phase 1 figures and metrics for the LQR and NMPC vertical slice.

The controller rollouts use the same perturbed-start scenario as the public
evaluation CLI.  Report montages are extracted from the labelled MP4 files so
the submitted figures and demonstration videos show the same runs.

Run from the repository root after generating the two videos documented in
``reports/phase1/ieee/README.md``::

    .venv/bin/python reports/phase1/make_controller_evidence.py
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from importlib.metadata import version
from math import ceil, pi
from pathlib import Path
from time import perf_counter
from typing import Any

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

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
from trailer_docking.evaluation.runner import REPOSITORY_ROOT, SCENARIOS
from trailer_docking.paths import straight_reference

REPORT_ROOT = Path(__file__).resolve().parent
OUTPUT = REPORT_ROOT / "ieee" / "figures"
VIDEO_ROOT = REPOSITORY_ROOT / "artifacts" / "controller_videos"
FRAME_STRIDE = 2
LATERAL_OFFSET_M = 0.35
HEADING_OFFSET_DEG = 3.0


@dataclass(slots=True)
class Trace:
    """Time history and terminal record for one deterministic rollout."""

    scenario: str
    controller: str
    time_s: list[float]
    lateral_error_m: list[float]
    heading_error_rad: list[float]
    articulation_rad: list[float]
    steering_rad: list[float]
    maximum_articulation_step: int
    result: dict[str, Any]


def _build_rollout(scenario: str, controller_name: str) -> tuple[
    TrailerDockingEnv, Controller, dict[str, Any]
]:
    vehicle_path, dock_path, task_path = SCENARIOS[scenario]
    environment = TrailerDockingEnv(
        vehicle_path=vehicle_path,
        dock_path=dock_path,
        task_path=task_path,
        action_mode="fixed_reverse",
    )
    start_x = (
        environment.dock.geometry.interior_depth
        + environment.settings.start_standoff
    )
    path = straight_reference(
        environment.dock.geometry.target_position,
        Point2D(start_x, 0.0),
    )
    heading = HEADING_OFFSET_DEG * pi / 180.0
    pose = aligned_pose_for_docking_point(
        environment.vehicle,
        Point2D(start_x, LATERAL_OFFSET_M),
        heading,
    )
    if controller_name == "lqr":
        controller: Controller = TrajectoryLQRController(environment.vehicle)
    else:
        controller = BasicNMPCController(environment.vehicle)
    options = {
        "tractor_pose": pose,
        "trailer_yaw": heading,
        "reference_path": path,
    }
    return environment, controller, options


def trace_rollout(scenario: str, controller_name: str) -> Trace:
    """Run the public controller contract and retain report-level diagnostics."""

    environment, controller, options = _build_rollout(scenario, controller_name)
    observation, info = environment.reset(seed=0, options=options)
    controller.reset()
    period = environment.simulator.controller_period
    times = [0.0]
    laterals = [environment.tracking.lateral_error]
    headings = [environment.tracking.trailer_heading_error]
    articulations = [environment.state.articulation_angle]
    steerings = [environment.state.steering_angle]
    maximum_step = 0
    maximum_articulation = abs(articulations[0])
    total_reward = 0.0
    compute_seconds = 0.0
    terminated = truncated = False

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
        total_reward += float(reward)
        times.append(environment.episode_steps * period)
        laterals.append(environment.tracking.lateral_error)
        headings.append(environment.tracking.trailer_heading_error)
        articulations.append(environment.state.articulation_angle)
        steerings.append(environment.state.steering_angle)
        if abs(articulations[-1]) > maximum_articulation:
            maximum_articulation = abs(articulations[-1])
            maximum_step = environment.episode_steps

    result = {
        "steps": environment.episode_steps,
        "simulated_time_s": environment.episode_steps * period,
        "total_reward": total_reward,
        "termination_reason": info["termination_reason"],
        "success": bool(info["is_success"]),
        "collision": bool(info["collision"]),
        "jackknifed": bool(info["jackknifed"]),
        "final_lateral_error_m": environment.tracking.lateral_error,
        "final_heading_error_rad": environment.tracking.trailer_heading_error,
        "maximum_articulation_rad": maximum_articulation,
        "controller_compute_seconds": compute_seconds,
        "mean_controller_compute_ms": (
            1000.0 * compute_seconds / environment.episode_steps
        ),
        "solve_failures": int(getattr(controller, "solve_failures", 0)),
        "configuration": {
            key: info[key]
            for key in (
                "vehicle_specification",
                "vehicle_fingerprint",
                "dock_specification",
                "dock_fingerprint",
                "task_specification",
                "task_fingerprint",
            )
        },
    }
    environment.close()
    return Trace(
        scenario=scenario,
        controller=controller_name,
        time_s=times,
        lateral_error_m=laterals,
        heading_error_rad=headings,
        articulation_rad=articulations,
        steering_rad=steerings,
        maximum_articulation_step=maximum_step,
        result=result,
    )


def _video_frames(trace: Trace) -> list[np.ndarray]:
    path = VIDEO_ROOT / f"{trace.controller}_{trace.scenario.replace('-', '_')}.mp4"
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}; generate the labelled controller videos first"
        )
    frames = list(iio.imiter(path, plugin="FFMPEG"))
    turn_index = min(
        ceil(trace.maximum_articulation_step / FRAME_STRIDE), len(frames) - 1
    )
    terminal_index = min(ceil(trace.result["steps"] / FRAME_STRIDE), len(frames) - 1)
    return [frames[0], frames[turn_index], frames[terminal_index]]


def save_montage(trace: Trace, filename: str) -> None:
    frames = _video_frames(trace)
    labels = ("Perturbed start", "Largest articulation", "Docking success")
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4))
    for axis, frame, label in zip(axes, frames, labels, strict=True):
        axis.imshow(frame)
        axis.set_title(label, fontsize=10)
        axis.axis("off")
    display = "LQR" if trace.controller == "lqr" else "NMPC"
    fig.suptitle(
        f"{display} closed-loop correction from 0.35 m / 3 degree perturbation",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_tracking_plot(lqr: Trace, nmpc: Trace) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.0), sharex=True)
    series = (
        ("lateral_error_m", "Lateral path error [m]", 1.0),
        ("heading_error_rad", "Trailer heading error [deg]", 180.0 / pi),
        ("articulation_rad", "Articulation [deg]", 180.0 / pi),
        ("steering_rad", "Steering angle [deg]", 180.0 / pi),
    )
    for axis, (attribute, ylabel, scale) in zip(axes.flat, series, strict=True):
        for trace, label, style in (
            (lqr, "LQR", "-"),
            (nmpc, "NMPC", "--"),
        ):
            values = np.asarray(getattr(trace, attribute)) * scale
            axis.plot(trace.time_s, values, style, linewidth=1.4, label=label)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.3)
    axes[1, 0].set_xlabel("Simulation time [s]")
    axes[1, 1].set_xlabel("Simulation time [s]")
    axes[0, 0].legend()
    fig.suptitle("Compact straight-path controller acceptance traces")
    fig.tight_layout()
    fig.savefig(OUTPUT / "compact_controller_traces.png", dpi=200)
    plt.close(fig)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for filename in ("mujoco_straight_reverse.png", "parity_2d_vs_mujoco.png"):
        shutil.copy2(REPORT_ROOT / "figures" / filename, OUTPUT / filename)
    traces: dict[str, Trace] = {}
    for controller in ("lqr", "nmpc"):
        key = f"{controller}_compact"
        traces[key] = trace_rollout("compact", controller)

    save_montage(traces["lqr_compact"], "lqr_compact_perturbed_tracking.png")
    save_montage(traces["nmpc_compact"], "nmpc_compact_perturbed_tracking.png")
    save_tracking_plot(traces["lqr_compact"], traces["nmpc_compact"])

    summary = {
        "provenance": {
            "git_revision": subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=REPOSITORY_ROOT,
                text=True,
            ).strip(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "packages": {
                name: version(name)
                for name in (
                    "casadi",
                    "gymnasium",
                    "imageio",
                    "matplotlib",
                    "numpy",
                )
            },
        },
        "scenario_definition": {
            "reference": "explicit straight path independent of reset pose",
            "lateral_offset_m": LATERAL_OFFSET_M,
            "aligned_tractor_and_trailer_heading_offset_deg": HEADING_OFFSET_DEG,
            "action": "normalized steering rate",
            "speed": "fixed configured reverse target",
        },
        "rollouts": {key: trace.result for key, trace in traces.items()},
        "figure_provenance": {
            "source": "labelled MP4 controller demonstrations",
            "video_directory": str(VIDEO_ROOT.relative_to(REPOSITORY_ROOT)),
            "frame_stride": FRAME_STRIDE,
            "selected_frames": "start, maximum-articulation step, terminal success",
        },
    }
    (REPORT_ROOT / "controller_evidence_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
