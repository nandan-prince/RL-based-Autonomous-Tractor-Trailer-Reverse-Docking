"""Command-line entry point for training and docking tests.

Usage::

    python -m trailer_docking.training.cli train [--scenario PATH]
        [--timesteps N] [--seed N] [--output-root DIR]
    python -m trailer_docking.training.cli test [--scenario PATH]
        [--policy PATH] [--episodes N] [--seed N]
"""

from __future__ import annotations

import argparse
import sys
import time
import zipfile
from collections.abc import Sequence
from pathlib import Path

from trailer_docking.core.config import ConfigurationError
from trailer_docking.evaluation.scenarios import Scenario
from trailer_docking.training.config import TrainingConfig
from trailer_docking.training.evaluate import (
    Policy,
    load_policy,
    run_docking_test,
    straight_reverse_policy,
    summarize,
)
from trailer_docking.training.session import TrainingProgress, TrainingSession

PROGRESS_PERIOD = 3.0
POLL_PERIOD = 0.2
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m trailer_docking.training.cli",
        description="Train a PPO docking policy or run a docking test.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="train PPO on a scenario")
    train.add_argument("--scenario", type=Path, help="scenario YAML file")
    train.add_argument("--timesteps", type=int, help="training budget")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--output-root", type=Path, default=Path("."))

    test = commands.add_parser("test", help="run a docking test")
    test.add_argument("--scenario", type=Path, help="scenario YAML file")
    test.add_argument("--policy", type=Path, help="PPO .zip (default: no steering)")
    test.add_argument("--episodes", type=int, default=1)
    test.add_argument("--seed", type=int, default=0)
    return parser


def _load_scenario(path: Path | None) -> Scenario:
    return Scenario() if path is None else Scenario.load(path)


def _progress_line(progress: TrainingProgress) -> str:
    rate = progress.recent_success_rate
    mean_return = progress.recent_mean_return
    return (
        f"[{progress.status}] {progress.timesteps}/{progress.total_timesteps} steps"
        f" | episodes {progress.episodes}"
        f" | recent success {'-' if rate is None else f'{rate:.0%}'}"
        f" | recent return {'-' if mean_return is None else f'{mean_return:.2f}'}"
        f" | last {progress.last_outcome or '-'}"
    )


def _train(arguments: argparse.Namespace) -> int:
    scenario = _load_scenario(arguments.scenario)
    scenario.build()  # reject an invalid scenario before starting the thread
    if arguments.timesteps is None:
        config = TrainingConfig(seed=arguments.seed)
    else:
        config = TrainingConfig(
            total_timesteps=arguments.timesteps, seed=arguments.seed
        )
    session = TrainingSession(scenario, config, output_root=arguments.output_root)
    session.start()
    last_report = time.monotonic()
    while session.progress.status in ("running", "stopping"):
        try:
            session.join(POLL_PERIOD)
            if time.monotonic() - last_report >= PROGRESS_PERIOD:
                last_report = time.monotonic()
                print(_progress_line(session.progress), flush=True)
        except KeyboardInterrupt:
            print("stopping after the current step...", flush=True)
            session.stop()
    session.join()
    progress = session.progress
    print(_progress_line(progress))
    print(f"experiment: {progress.experiment_id}")
    print(f"checkpoint: {progress.checkpoint_path}")
    if progress.status == "failed":
        print(f"error: {progress.error}", file=sys.stderr)
        return EXIT_FAILED
    return EXIT_OK


def _test(arguments: argparse.Namespace) -> int:
    scenario = _load_scenario(arguments.scenario)
    if arguments.seed < 0:
        raise ValueError("--seed cannot be negative")
    policy: Policy = straight_reverse_policy
    if arguments.policy is not None:
        try:
            policy = load_policy(arguments.policy)
        except FileNotFoundError:
            raise
        except (OSError, zipfile.BadZipFile, KeyError, EOFError) as error:
            raise ValueError(
                f"{arguments.policy} is not a saved policy: {error}"
            ) from error
    results = run_docking_test(
        scenario, policy, episodes=arguments.episodes, seed=arguments.seed
    )
    for index, result in enumerate(results):
        print(
            f"episode {index}: {result.outcome} in {result.steps} steps"
            f" | reward {result.total_reward:.2f}"
            f" | longitudinal {result.final_longitudinal_error:+.3f} m"
            f" | lateral {result.final_lateral_error:+.3f} m"
            f" | yaw {result.final_yaw_error:+.3f} rad"
            f" | max articulation {result.max_abs_articulation:.3f} rad"
        )
    print("summary:")
    for key, value in summarize(results).items():
        print(
            f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}"
        )
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return its exit code."""

    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "train":
            return _train(arguments)
        return _test(arguments)
    except (ConfigurationError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE
    except ValueError as error:
        # Invalid numeric options (for example --timesteps 0 or --episodes 0).
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE
    except ImportError as error:
        # Loading a saved policy needs PyTorch and Stable-Baselines3, which
        # are optional so the rest of the package installs without them.
        print(
            f"error: {error}. Install the optional training dependencies with"
            " pip install -e '.[rl,train]'.",
            file=sys.stderr,
        )
        return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
