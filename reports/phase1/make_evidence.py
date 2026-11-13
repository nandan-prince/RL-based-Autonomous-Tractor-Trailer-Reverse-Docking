"""Regenerate the figures, animations, and numbers cited in the Phase 1 report.

Run from a checkout of ``main`` (revision 437c2bc or later) with the
``dev``, ``mujoco`` and ``viz`` extras installed::

    python reports/phase1/make_evidence.py

It only calls the package's public simulators and Gymnasium environment; it
contains no dynamics, reward, or success logic of its own.  Every rollout uses
a fixed open-loop command, so nothing here is a controller result.
"""

from __future__ import annotations

import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import gymnasium as gym
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from gymnasium.utils.env_checker import check_env
from PIL import Image

import trailer_docking
from trailer_docking.core.config import (
    load_dock_configuration,
    load_vehicle_configuration,
)
from trailer_docking.core.state import VehicleCommand
from trailer_docking.environments import TrailerDockingEnv
from trailer_docking.simulators.kinematic import KinematicSimulator
from trailer_docking.simulators.mujoco.adapter import MujocoSimulator

OUT = Path(__file__).resolve().parent / "figures"
ROOT = Path(trailer_docking.__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
VEHICLE = CONFIGS / "vehicles/compact_semitrailer_v1.yaml"
DOCK = CONFIGS / "environments/nominal_dock_v1.yaml"
SERVICE = {
    "vehicle_path": CONFIGS / "vehicles/full_size_semitrailer_v1.yaml",
    "dock_path": CONFIGS / "environments/service_center_dock_v1.yaml",
    "task_path": CONFIGS / "environments/service_center_gymnasium_v1.yaml",
}


def save_gif(frames: list[np.ndarray], path: Path, frame_ms: int = 80) -> None:
    images = [Image.fromarray(f).convert("P", palette=Image.ADAPTIVE) for f in frames]
    images[0].save(
        path, save_all=True, append_images=images[1:], duration=frame_ms, loop=0
    )


def save_strip(frames: list[np.ndarray], path: Path) -> None:
    picks = [frames[0], frames[len(frames) // 2], frames[-1]]
    Image.fromarray(np.concatenate(picks, axis=1)).save(path)


def rollout(env: TrailerDockingEnv, action: np.ndarray, every: int) -> dict[str, Any]:
    """Hold one fixed action until the episode ends, keeping rendered frames."""

    _, info = env.reset(seed=0)
    frames = [np.asarray(env.render())]
    total = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        _, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
        if info["episode_steps"] % every == 0 or terminated or truncated:
            frames.append(np.asarray(env.render()))
    return {
        "frames": frames,
        "steps": info["episode_steps"],
        "termination_reason": info["termination_reason"],
        "is_success": bool(info["is_success"]),
        "collision": bool(info["collision"]),
        "jackknifed": bool(info["jackknifed"]),
        "docking_error_longitudinal_m": float(info["docking_error_longitudinal"]),
        "docking_error_lateral_m": float(info["docking_error_lateral"]),
        "docking_error_yaw_rad": float(info["docking_error_yaw"]),
        "return": total,
    }


def gymnasium_evidence() -> dict[str, Any]:
    result: dict[str, Any] = {}

    env = gym.make("TrailerDocking-v0")
    check_env(env.unwrapped, skip_render_check=True)
    result["api_checker"] = "passed"
    result["observation_shape"] = list(env.observation_space.shape or ())
    result["action_shape"] = list(env.action_space.shape or ())
    env.close()

    compact = TrailerDockingEnv(render_mode="rgb_array")
    run = rollout(compact, np.array([0.0], dtype=np.float32), every=10)
    compact.close()
    frames = run.pop("frames")
    save_gif(frames, OUT / "gym2d_compact_straight_reverse.gif")
    save_strip(frames, OUT / "gym2d_compact_straight_reverse.png")
    result["compact_straight_reverse"] = run

    service = TrailerDockingEnv(
        action_mode="teleop", render_mode="rgb_array", **SERVICE
    )
    run = rollout(service, np.array([0.0, -1.0], dtype=np.float32), every=8)
    service.close()
    frames = run.pop("frames")
    save_gif(frames, OUT / "gym2d_service_center_straight_reverse.gif")
    Image.fromarray(frames[len(frames) // 2]).save(
        OUT / "gym2d_service_center.png"
    )
    result["service_center_straight_reverse"] = run

    # Uniform random steering from seeded random starts: shows that episodes
    # end through the event logic.  This is not a policy evaluation.
    random_env = TrailerDockingEnv()
    reasons: dict[str, int] = {}
    for seed in range(20):
        random_env.reset(seed=seed, options={"randomize": True})
        random_env.action_space.seed(seed)
        terminated = truncated = False
        info: dict[str, Any] = {}
        while not (terminated or truncated):
            _, _, terminated, truncated, info = random_env.step(
                random_env.action_space.sample()
            )
        reason = str(info["termination_reason"] or "timeout")
        reasons[reason] = reasons.get(reason, 0) + 1
    random_env.close()
    result["random_steering_20_seeded_episodes"] = reasons
    return result


def parity_evidence() -> dict[str, Any]:
    """Same open-loop reverse command sequence through both plants."""

    vehicle = load_vehicle_configuration(VEHICLE)
    kinematic = KinematicSimulator(vehicle)
    physics = MujocoSimulator(vehicle)
    physics.reset(settle_time=0.5)
    speed = -0.2
    commands = (
        [VehicleCommand(0.3, speed)] * 8
        + [VehicleCommand(0.0, speed)] * 60
        + [VehicleCommand(-0.3, speed)] * 16
        + [VehicleCommand(0.0, speed)] * 66
    )
    rows: dict[str, list[list[float]]] = {"2D kinematic": [], "MuJoCo": []}

    def record(name: str, state: Any) -> None:
        rows[name].append(
            [
                state.simulation_time,
                state.trailer_pose.x,
                state.trailer_pose.y,
                state.tractor_pose.x,
                state.tractor_pose.y,
                state.tractor_pose.yaw,
                state.trailer_pose.yaw,
                state.articulation_angle,
                state.steering_angle,
            ]
        )

    record("2D kinematic", kinematic.state)
    record("MuJoCo", physics.state)
    for command in commands:
        record("2D kinematic", kinematic.step(command))
        record("MuJoCo", physics.step(command))
    k = np.array(rows["2D kinematic"])
    m = np.array(rows["MuJoCo"])

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for data, style, label in ((k, "-", "2D kinematic"), (m, "--", "MuJoCo")):
        axes[0].plot(data[:, 3], data[:, 4], style, label=f"tractor axle, {label}")
        axes[0].plot(data[:, 1], data[:, 2], style, label=f"trailer axle, {label}")
        axes[1].plot(data[:, 0], np.degrees(data[:, 7]), style, label=label)
        axes[2].plot(data[:, 0], np.degrees(data[:, 8]), style, label=label)
    axes[0].set(xlabel="x [m]", ylabel="y [m]", title="Axle paths (reverse S-maneuver)")
    axes[1].set(xlabel="time [s]", ylabel="articulation [deg]", title="Articulation")
    axes[2].set(xlabel="time [s]", ylabel="steering [deg]", title="Virtual steering")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "parity_2d_vs_mujoco.png", dpi=140)
    plt.close(fig)

    position_gap = np.hypot(k[:, 3] - m[:, 3], k[:, 4] - m[:, 4])
    return {
        "commands": len(commands),
        "duration_s": float(k[-1, 0]),
        "final_tractor_position_gap_m": float(position_gap[-1]),
        "max_tractor_position_gap_m": float(position_gap.max()),
        "final_tractor_yaw_gap_rad": float(abs(k[-1, 5] - m[-1, 5])),
        "final_trailer_yaw_gap_rad": float(abs(k[-1, 6] - m[-1, 6])),
        "max_articulation_gap_rad": float(np.abs(k[:, 7] - m[:, 7]).max()),
    }


def mujoco_evidence() -> dict[str, Any]:
    """Offscreen renders of the generated scene during a fixed straight reverse."""

    vehicle = load_vehicle_configuration(VEHICLE)
    dock = load_dock_configuration(DOCK)
    sim = MujocoSimulator(vehicle, dock)
    sim.reset(settle_time=0.5)
    renderer = mujoco.Renderer(sim.model, height=480, width=640)

    def frame(camera: str) -> np.ndarray:
        renderer.update_scene(sim.data, camera=camera)
        return renderer.render().copy()

    frames = [frame("yard")]
    overhead_start = frame("overhead")
    steps = 0
    contacts: tuple[str, ...] = ()
    while not contacts and steps < 1200:
        sim.step(VehicleCommand(0.0, -vehicle.limits.reverse_speed_abs_max))
        steps += 1
        contacts = sim.wall_contacts()
        if steps % 10 == 0 or contacts:
            frames.append(frame("yard"))
    state = sim.state
    save_gif(frames, OUT / "mujoco_straight_reverse.gif")
    save_strip(frames, OUT / "mujoco_straight_reverse.png")
    Image.fromarray(np.concatenate([overhead_start, frame("overhead")], axis=1)).save(
        OUT / "mujoco_overhead_start_end.png"
    )
    renderer.close()
    return {
        "control_steps_until_wall_contact": steps,
        "wall_contacts": list(contacts),
        "final_articulation_rad": float(state.articulation_angle),
        "final_trailer_lateral_m": float(state.trailer_pose.y),
        "bodies": int(sim.model.nbody),
        "joints": int(sim.model.njnt),
        "actuators": int(sim.model.nu),
        "physics_timestep_s": float(sim.model.opt.timestep),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "packages": {
            name: version(name)
            for name in ("gymnasium", "mujoco", "numpy", "matplotlib", "PyYAML")
        },
        "gymnasium_2d": gymnasium_evidence(),
        "parity_2d_vs_mujoco": parity_evidence(),
        "mujoco_3d": mujoco_evidence(),
    }
    (OUT.parent / "evidence_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
