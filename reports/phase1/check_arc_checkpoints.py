"""Measure arc tracking from saved PPO checkpoints on fixed starts."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from statistics import mean

import yaml

from trailer_docking.evaluation.episodes import resolve_episodes
from trailer_docking.evaluation.scenarios import Scenario
from trailer_docking.training.environment import ScenarioDockingEnv, episode_outcome
from trailer_docking.training.evaluate import load_policy
from trailer_docking.training.session import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="training run ID, rather than its campaign ID")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=["final"],
        help="checkpoint steps, final, or initial for the warm-start policy",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--output", type=Path, help="new CSV file (default: stdout)")
    args = parser.parse_args()
    manifest = yaml.safe_load(
        (Path("results") / args.run / "manifest.yaml").read_text(encoding="utf-8")
    )
    scenario = Scenario.from_dict(manifest["scenario"])
    if (
        scenario.planner not in ("quarter_arc", "arc_30")
        or manifest["config"]["backend"] != "2d"
    ):
        parser.error("this diagnostic requires a supported 2D arc scenario")
    episodes = resolve_episodes(scenario, episodes=args.episodes, seed=args.seed)
    directory = Path("artifacts") / args.run / "checkpoints"
    env = ScenarioDockingEnv(
        scenario,
        backend="2d",
        max_episode_steps=manifest["config"]["max_episode_steps"],
    )
    columns = (
        "checkpoint",
        "checkpoint_sha256",
        "seed",
        "episode_fingerprint",
        "outcome",
        "steps",
        "projected_arc_fraction",
        "arc_fraction_within_0_5_m",
        "mean_arc_lateral_error_m",
        "peak_arc_lateral_error_m",
        "mean_straight_approach_error_m",
        "straight_drifted_gt_0_5_m",
        "straight_recovered_to_0_3_m",
    )
    stream = (
        args.output.open("x", encoding="utf-8", newline="")
        if args.output
        else sys.stdout
    )
    try:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for name in args.checkpoints:
            if name == "initial":
                if not manifest["initial_policy_path"]:
                    parser.error("this run has no warm-start policy")
                checkpoint = Path(manifest["initial_policy_path"])
            else:
                checkpoint = directory / f"policy_{name}.zip"
            policy = load_policy(checkpoint)
            checksum = file_sha256(checkpoint)
            for episode in episodes:
                observation, _ = env.reset(
                    seed=episode.seed, options={"resolved_episode": episode}
                )
                route = env.reference_path
                join = route.segment_lengths[0]
                arc_length = route.length - join
                furthest, within_corridor = 0.0, 0.0
                stayed_in_corridor = True
                steps = 0
                errors = []
                approach_errors = []
                drifted = recovered = False
                while True:
                    observation, _, ended, truncated, info = env.step(
                        policy(observation)
                    )
                    tracking = env.task_env.tracking
                    error = abs(tracking.lateral_error)
                    coverage = min(
                        max((route.length - tracking.progress) / arc_length, 0.0), 1.0
                    )
                    furthest = max(furthest, coverage)
                    if tracking.progress >= join:
                        errors.append(error)
                    if 0.0 <= info["docking_error_longitudinal"] <= join:
                        approach_errors.append(error)
                        drifted = drifted or error > 0.5
                        recovered = recovered or (drifted and error <= 0.3)
                    stayed_in_corridor = stayed_in_corridor and error <= 0.5
                    if stayed_in_corridor:
                        within_corridor = max(within_corridor, coverage)
                    steps += 1
                    if ended or truncated:
                        break
                writer.writerow(
                    {
                        "checkpoint": name,
                        "checkpoint_sha256": checksum,
                        "seed": episode.seed,
                        "episode_fingerprint": episode.episode_fingerprint,
                        "outcome": episode_outcome(info),
                        "steps": steps,
                        "projected_arc_fraction": furthest,
                        "arc_fraction_within_0_5_m": within_corridor,
                        "mean_arc_lateral_error_m": mean(errors) if errors else None,
                        "peak_arc_lateral_error_m": max(errors) if errors else None,
                        "mean_straight_approach_error_m": (
                            mean(approach_errors) if approach_errors else None
                        ),
                        "straight_drifted_gt_0_5_m": drifted,
                        "straight_recovered_to_0_3_m": recovered,
                    }
                )
                stream.flush()
    finally:
        env.close()
        if args.output:
            stream.close()


if __name__ == "__main__":
    main()
