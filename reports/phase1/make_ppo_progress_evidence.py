"""Replay saved PPO runs and draw training evidence for the course report."""

from __future__ import annotations

import argparse
import csv
from itertools import pairwise
from pathlib import Path
from statistics import mean

import matplotlib
import yaml
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon, Rectangle

from trailer_docking.evaluation.episodes import resolve_episode
from trailer_docking.evaluation.scenarios import Scenario
from trailer_docking.training.environment import ScenarioDockingEnv, episode_outcome
from trailer_docking.training.evaluate import load_policy

matplotlib.use("Agg")

RUN = "20261126-0014_ppo_ppo-ui-run-1-of-1"
CURVE_RUN = "20261126-1213_ppo_ppo-ui-run-1-of-1"
SEED = 100000
FIGURES = Path(__file__).resolve().parent / "ieee" / "figures"


def replay(env, episode, checkpoint):
    policy = load_policy(checkpoint)
    observation, info = env.reset(seed=SEED, options={"resolved_episode": episode})
    states = [env.state]
    approach_errors = []
    while True:
        observation, _, terminal, timeout, info = env.step(policy(observation))
        states.append(env.state)
        if info["docking_error_longitudinal"] >= 0:
            approach_errors.append(abs(env.task_env.tracking.lateral_error))
        if terminal or timeout:
            return states, episode_outcome(info), mean(approach_errors)


def plot_training_success(run_directory: Path, output: Path) -> None:
    with (run_directory / "metrics.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    bounds = (0, 50_000, 100_000, 150_000, int(rows[-1]["timesteps"]))
    labels = ("0-50k", "50-100k", "100-150k", "150k-end")
    counts = []
    for lo, hi in pairwise(bounds):
        window = [row for row in rows if lo < int(row["timesteps"]) <= hi]
        counts.append((sum(row["outcome"] == "success" for row in window), len(window)))
    figure, axis = plt.subplots(figsize=(4.1, 2.3))
    bars = axis.bar(
        labels, [100 * wins / size for wins, size in counts], color="#3866a9"
    )
    axis.bar_label(bars, labels=[f"{wins}/{size}" for wins, size in counts], padding=3)
    axis.set(
        xlabel="Training step at episode completion",
        ylabel="Episodes docked [%]",
        ylim=(0, 42),
    )
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(output, dpi=240)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gentle-curve", action="store_true")
    args = parser.parse_args()
    run = CURVE_RUN if args.gentle_curve else RUN
    run_directory = Path("results") / run
    figure_name = (
        "ppo_gentle_curve_replay.png"
        if args.gentle_curve
        else "ppo_perturbed_replay.png"
    )
    manifest = yaml.safe_load(
        (run_directory / "manifest.yaml").read_text(encoding="utf-8")
    )
    scenario = Scenario.from_dict(manifest["scenario"])
    episode = resolve_episode(scenario, SEED)
    env = ScenarioDockingEnv(
        scenario,
        backend="2d",
        max_episode_steps=manifest["config"]["max_episode_steps"],
    )
    directory = Path("artifacts") / run / "checkpoints"
    figure, axes = plt.subplots(2, 3, figsize=(10.5, 3.6), sharex=True, sharey=True)
    vehicle = env.built.vehicle
    bay = env.built.dock.geometry
    target = bay.target_position
    dt = vehicle.timing.controller_period
    try:
        if args.gentle_curve:
            for label, checkpoint in (
                ("Initialization", Path(manifest["initial_policy_path"])),
                ("20k checkpoint", directory / "policy_20000.zip"),
            ):
                states, outcome, error = replay(env, episode, checkpoint)
                print(label, outcome, len(states) - 1, f"approach MAE {error:.3f} m")
        first = (
            (0, "policy_100000.zip", "100k checkpoint")
            if args.gentle_curve
            else (0, "policy_20000.zip", "20k checkpoint")
        )
        for row, name, label in (
            first,
            (1, "policy_final.zip", "Final checkpoint"),
        ):
            states, outcome, approach_error = replay(env, episode, directory / name)
            indices = (0, (len(states) - 1) // 2, len(states) - 1)
            print(
                label, outcome, len(states) - 1, f"approach MAE {approach_error:.3f} m"
            )
            for column, (step, phase) in enumerate(
                zip(
                    indices,
                    (
                        "Start",
                        "Mid-rollout",
                        "Docked" if outcome == "success" else outcome.replace("_", " "),
                    ),
                    strict=True,
                )
            ):
                axis = axes[row, column]
                state = states[step]
                axis.add_patch(
                    Rectangle(
                        (0, -bay.interior_width / 2),
                        bay.interior_depth,
                        bay.interior_width,
                        facecolor="#f6eab1",
                    )
                )
                axis.plot(
                    [p.x for p in episode.reference_path.points],
                    [p.y for p in episode.reference_path.points],
                    "--",
                    color="#44714f",
                    linewidth=1,
                )
                trace = [
                    vehicle.trailer.rear_face(s.trailer_pose)
                    for s in states[: step + 1]
                ]
                axis.plot([p.x for p in trace], [p.y for p in trace], color="#ae4438")
                for footprint, color in (
                    (vehicle.trailer.footprint(state.trailer_pose), "#c7c9ce"),
                    (vehicle.tractor.footprint(state.tractor_pose), "#3866a9"),
                ):
                    axis.add_patch(
                        Polygon(
                            [(p.x, p.y) for p in footprint],
                            facecolor=color,
                            edgecolor="#222222",
                        )
                    )
                axis.plot(target.x, target.y, marker="x", color="#216f32", zorder=5)
                axis.plot(trace[-1].x, trace[-1].y, marker="o", ms=3, color="#ae4438")
                axis.set_title(f"{phase.capitalize()} · {step * dt:.1f} s", fontsize=10)
                axis.set(xlim=(-2.8, 22.5), ylim=(-5.3, 5.3), aspect="equal")
                axis.grid(alpha=0.2)
                if column == 0:
                    axis.set_ylabel(f"{label}\ny [m]")
                if row == 1:
                    axis.set_xlabel("x [m]")
    finally:
        env.close()
    figure.tight_layout()
    figure.savefig(args.output or FIGURES / figure_name, dpi=220, bbox_inches="tight")
    plt.close(figure)
    if args.gentle_curve:
        plot_training_success(run_directory, FIGURES / "ppo_gentle_curve_training.png")


if __name__ == "__main__":
    main()
