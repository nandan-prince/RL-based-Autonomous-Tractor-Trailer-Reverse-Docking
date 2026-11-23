"""Command-line entry point for training campaigns, tests, and bundles.

Usage::

    python -m trailer_docking.training.cli train [--scenario PATH]
        [--method ppo|nmpc] [--timesteps N] [--runs N] [--seed N]
        [--backend 2d|3d]
        [--eval-episodes N] [--eval-backends 2d,3d] [--output-root DIR]
    python -m trailer_docking.training.cli test [--scenario PATH]
        [--policy PATH | --bundle PATH | --nmpc | --lqr] [--backend 2d|3d]
        [--episodes N] [--seed N]
    python -m trailer_docking.training.cli compare [--scenario PATH]
        [--controllers lqr,nmpc,ppo] [--policy PATH]
        [--backends 2d,3d] [--episodes N] [--seed N]
    python -m trailer_docking.training.cli curriculum --config PATH
        [--initial-policy PATH] [--start-stage INDEX] [--output-root DIR]
    python -m trailer_docking.training.cli reward-study --config PATH
        [--initial-policy PATH] [--output-root DIR]
    python -m trailer_docking.training.cli export --policy PATH
        --output PATH [--scenario PATH]

``train`` runs a campaign: it repeats the training run, tests each final
policy, and writes the best one as a portable bundle.  ``test --bundle`` runs
a bundle in a freshly built simulator using the scenario stored inside it.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from trailer_docking.controllers.configuration import (
    DEFAULT_CONTROLLER_CONFIG_PATH,
    ControllerConfigurations,
)
from trailer_docking.core.config import ConfigurationError
from trailer_docking.evaluation.episodes import resolve_episodes
from trailer_docking.evaluation.scenarios import Scenario
from trailer_docking.evaluation.suites import SPLIT_NAMES, ExperimentSuite
from trailer_docking.training.bundle import export_bundle, load_bundle, run_bundle
from trailer_docking.training.campaign import TrainingCampaign
from trailer_docking.training.comparison import (
    ComparisonMethod,
    run_controlled_comparison,
    save_comparison_evidence,
)
from trailer_docking.training.config import BACKENDS, DEVICES, TrainingConfig
from trailer_docking.training.curriculum import (
    DEFAULT_CURRICULUM_PATH,
    CurriculumConfiguration,
    CurriculumRunner,
)
from trailer_docking.training.evaluate import (
    EpisodeResult,
    Policy,
    load_policy,
    resolve_policy_path,
    run_docking_test,
    straight_reverse_policy,
    summarize,
)
from trailer_docking.training.methods import (
    METHODS,
    MethodUnavailableError,
    lqr_controller_factory,
    nmpc_controller_factory,
)
from trailer_docking.training.session import TrainingProgress
from trailer_docking.training.study import (
    DEFAULT_REWARD_STUDY_PATH,
    PPORewardStudyConfiguration,
    PPORewardStudyRunner,
)

PROGRESS_PERIOD = 3.0
POLL_PERIOD = 0.2
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2


def _backend_list(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def _controller_list(text: str) -> tuple[str, ...]:
    values = tuple(part.strip().lower() for part in text.split(",") if part.strip())
    unknown = sorted(set(values) - {"lqr", "nmpc", "ppo"})
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown controller(s): {', '.join(unknown)}"
        )
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m trailer_docking.training.cli",
        description="Train PPO docking policies, test them, and export bundles.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="run a training campaign")
    train.add_argument("--scenario", type=Path, help="scenario YAML file")
    train.add_argument("--timesteps", type=int, help="training budget per run")
    train.add_argument("--runs", type=int, help="number of training runs")
    train.add_argument("--seed", type=int, default=0, help="seed of the first run")
    train.add_argument("--backend", choices=BACKENDS, help="training simulator")
    train.add_argument(
        "--method",
        choices=METHODS,
        help="ppo trains a policy; nmpc skips training and runs the controller",
    )
    train.add_argument(
        "--eval-episodes", type=int, help="test episodes per run (0 skips tests)"
    )
    train.add_argument(
        "--eval-backends",
        type=_backend_list,
        help="comma-separated simulators to test in, for example 2d,3d",
    )
    train.add_argument("--output-root", type=Path, default=Path("."))
    train.add_argument(
        "--controller-config",
        type=Path,
        default=DEFAULT_CONTROLLER_CONFIG_PATH,
        help="versioned LQR/NMPC settings YAML",
    )
    train.add_argument("--learning-rate", type=float)
    train.add_argument("--n-steps", type=int)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--n-epochs", type=int)
    train.add_argument("--gamma", type=float)
    train.add_argument("--gae-lambda", type=float)
    train.add_argument("--clip-range", type=float)
    train.add_argument("--clip-range-vf", type=float)
    train.add_argument(
        "--normalize-advantage", action=argparse.BooleanOptionalAction, default=None
    )
    train.add_argument("--entropy-coefficient", type=float)
    train.add_argument("--value-coefficient", type=float)
    train.add_argument("--max-grad-norm", type=float)
    train.add_argument("--target-kl", type=float)
    train.add_argument("--device", choices=DEVICES)

    test = commands.add_parser("test", help="run a docking test")
    test.add_argument("--scenario", type=Path, help="scenario YAML file")
    source = test.add_mutually_exclusive_group()
    source.add_argument("--policy", type=Path, help="PPO .zip (default: no steering)")
    source.add_argument("--bundle", type=Path, help="exported policy bundle")
    source.add_argument(
        "--nmpc", action="store_true", help="run the NMPC controller instead"
    )
    source.add_argument(
        "--lqr", action="store_true", help="run the trajectory LQR instead"
    )
    test.add_argument("--backend", choices=BACKENDS, default="2d")
    test.add_argument("--episodes", type=int, default=1)
    test.add_argument("--seed", type=int, default=0)
    test.add_argument(
        "--controller-config",
        type=Path,
        default=DEFAULT_CONTROLLER_CONFIG_PATH,
        help="versioned LQR/NMPC settings YAML",
    )

    compare = commands.add_parser(
        "compare", help="run controllers on one frozen paired episode set"
    )
    compare.add_argument("--scenario", type=Path, help="scenario YAML file")
    compare.add_argument(
        "--controllers",
        type=_controller_list,
        default=("lqr", "nmpc"),
        help="comma-separated lqr,nmpc,ppo (default: lqr,nmpc)",
    )
    compare.add_argument(
        "--policy", type=Path, help="PPO .zip, required when ppo is selected"
    )
    compare.add_argument(
        "--backends",
        type=_backend_list,
        default=("2d",),
        help="comma-separated 2d,3d (default: 2d)",
    )
    compare.add_argument("--episodes", type=int, default=1)
    compare.add_argument("--seed", type=int, default=0)
    compare.add_argument(
        "--controller-config",
        type=Path,
        default=DEFAULT_CONTROLLER_CONFIG_PATH,
        help="versioned LQR/NMPC settings YAML",
    )
    compare.add_argument("--suite", type=Path, help="experiment-suite YAML")
    compare.add_argument("--split", choices=SPLIT_NAMES, default="validation")
    compare.add_argument(
        "--save-results",
        action="store_true",
        help="write an immutable formal evidence package (requires --suite)",
    )
    compare.add_argument("--purpose", default="controller-comparison")
    compare.add_argument("--output-root", type=Path, default=Path("."))

    curriculum = commands.add_parser(
        "curriculum", help="run a gated, warm-started PPO curriculum"
    )
    curriculum.add_argument(
        "--config", type=Path, default=DEFAULT_CURRICULUM_PATH
    )
    curriculum.add_argument("--initial-policy", type=Path)
    curriculum.add_argument("--start-stage", type=int, default=0)
    curriculum.add_argument("--output-root", type=Path, default=Path("."))

    study = commands.add_parser(
        "reward-study", help="run a controlled multi-seed PPO reward study"
    )
    study.add_argument("--config", type=Path, default=DEFAULT_REWARD_STUDY_PATH)
    study.add_argument("--initial-policy", type=Path)
    study.add_argument("--output-root", type=Path, default=Path("."))

    export = commands.add_parser("export", help="bundle a saved policy")
    export.add_argument("--policy", type=Path, required=True, help="PPO .zip")
    export.add_argument("--output", type=Path, required=True, help="bundle to write")
    export.add_argument("--scenario", type=Path, help="scenario it was trained on")
    export.add_argument(
        "--manifest",
        type=Path,
        help="the run's manifest.yaml, to record its real training settings",
    )
    return parser


def _load_scenario(path: Path | None) -> Scenario:
    return Scenario() if path is None else Scenario.load(path)


def _progress_line(progress: TrainingProgress) -> str:
    rate = progress.recent_success_rate
    mean_return = progress.recent_mean_return
    return (
        f"[{progress.status}] {progress.timesteps}/{progress.total_timesteps} steps"
        f" | run {progress.run}/{progress.runs}"
        f" | episodes {progress.episodes}"
        f" | recent success {'-' if rate is None else f'{rate:.0%}'}"
        f" | recent return {'-' if mean_return is None else f'{mean_return:.2f}'}"
        f" | {progress.phase or progress.last_outcome or '-'}"
    )


def _training_config(arguments: argparse.Namespace) -> TrainingConfig:
    chosen: dict[str, Any] = {"seed": arguments.seed}
    for option, field in (
        ("timesteps", "total_timesteps"),
        ("runs", "runs"),
        ("backend", "backend"),
        ("method", "method"),
        ("eval_episodes", "evaluation_episodes"),
        ("eval_backends", "evaluation_backends"),
        ("learning_rate", "learning_rate"),
        ("n_steps", "n_steps"),
        ("batch_size", "batch_size"),
        ("n_epochs", "n_epochs"),
        ("gamma", "gamma"),
        ("gae_lambda", "gae_lambda"),
        ("clip_range", "clip_range"),
        ("clip_range_vf", "clip_range_vf"),
        ("normalize_advantage", "normalize_advantage"),
        ("entropy_coefficient", "entropy_coefficient"),
        ("value_coefficient", "value_coefficient"),
        ("max_grad_norm", "max_grad_norm"),
        ("target_kl", "target_kl"),
        ("device", "device"),
    ):
        value = getattr(arguments, option)
        if value is not None:
            chosen[field] = value
    return TrainingConfig(**chosen)


def _train(arguments: argparse.Namespace) -> int:
    scenario = _load_scenario(arguments.scenario)
    scenario.build()  # reject an invalid scenario before starting the thread
    config = _training_config(arguments)
    controller_configurations = ControllerConfigurations.load(
        arguments.controller_config
    )
    campaign = TrainingCampaign(
        scenario,
        config,
        output_root=arguments.output_root,
        controller_loader=lambda: nmpc_controller_factory(
            controller_configurations.nmpc
        ),
        controller_configurations=controller_configurations,
    )
    campaign.start()
    last_report = time.monotonic()
    while campaign.progress.status in ("running", "stopping"):
        try:
            campaign.join(POLL_PERIOD)
            if time.monotonic() - last_report >= PROGRESS_PERIOD:
                last_report = time.monotonic()
                print(_progress_line(campaign.progress), flush=True)
        except KeyboardInterrupt:
            print("stopping after the current step...", flush=True)
            campaign.stop()
    campaign.join()
    progress = campaign.progress
    print(_progress_line(progress))
    if config.method == "nmpc" and (arguments.runs or arguments.timesteps):
        print("note: --runs and --timesteps are ignored for nmpc (nothing is trained)")
    for run in campaign.run_summaries:
        tests = "; ".join(
            f"{item['backend']}: {item['success_rate']:.0%} of {item['episodes']}"
            for item in run["evaluation"]
        )
        print(
            f"run {run['run']} (seed {run['seed']}): {run['status']}"
            f" | {run['experiment_id']} | {tests or 'not tested'}"
        )
    print(f"campaign: {progress.experiment_id}")
    print(f"best checkpoint: {progress.checkpoint_path}")
    print(f"bundle: {progress.bundle_path}")
    if progress.status == "failed":
        print(f"error: {progress.error}", file=sys.stderr)
        return EXIT_FAILED
    return EXIT_OK


def _print_results(results: Sequence[EpisodeResult]) -> None:
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


def _test(arguments: argparse.Namespace) -> int:
    if arguments.seed < 0:
        raise ValueError("--seed cannot be negative")
    if arguments.bundle is not None:
        bundle = load_bundle(arguments.bundle)
        print(f"bundle: {bundle.description()}")
        if not bundle.fingerprint_matches:
            print(
                "warning: the base specifications changed since this bundle"
                " was trained",
                file=sys.stderr,
            )
        results = run_bundle(
            bundle,
            backend=arguments.backend,
            episodes=arguments.episodes,
            seed=arguments.seed,
            scenario=(
                None
                if arguments.scenario is None
                else Scenario.load(arguments.scenario)
            ),
        )
        _print_results(results)
        return EXIT_OK
    scenario = _load_scenario(arguments.scenario)
    controller_configurations = ControllerConfigurations.load(
        arguments.controller_config
    )
    if arguments.nmpc:
        results = run_docking_test(
            scenario,
            None,
            episodes=arguments.episodes,
            seed=arguments.seed,
            backend=arguments.backend,
            controller_factory=nmpc_controller_factory(
                controller_configurations.nmpc
            ),
        )
        _print_results(results)
        return EXIT_OK
    if arguments.lqr:
        results = run_docking_test(
            scenario,
            None,
            episodes=arguments.episodes,
            seed=arguments.seed,
            backend=arguments.backend,
            controller_factory=lqr_controller_factory(controller_configurations.lqr),
        )
        _print_results(results)
        return EXIT_OK
    policy: Policy = straight_reverse_policy
    if arguments.policy is not None:
        policy = load_policy(arguments.policy)
    results = run_docking_test(
        scenario,
        policy,
        episodes=arguments.episodes,
        seed=arguments.seed,
        backend=arguments.backend,
    )
    _print_results(results)
    return EXIT_OK


def _compare(arguments: argparse.Namespace) -> int:
    if arguments.seed < 0:
        raise ValueError("--seed cannot be negative")
    if not arguments.controllers:
        raise ValueError("--controllers cannot be empty")
    if not arguments.backends or any(
        item not in BACKENDS for item in arguments.backends
    ):
        raise ValueError(f"--backends must contain only {', '.join(BACKENDS)}")
    if "ppo" in arguments.controllers and arguments.policy is None:
        raise ValueError("--policy is required when ppo is selected")
    scenario = _load_scenario(arguments.scenario)
    controller_configurations = ControllerConfigurations.load(
        arguments.controller_config
    )
    suite = None if arguments.suite is None else ExperimentSuite.load(arguments.suite)
    if arguments.save_results and suite is None:
        raise ValueError("--save-results requires --suite")
    if suite is None:
        episodes = arguments.episodes
        seed = arguments.seed
        resolved = resolve_episodes(scenario, episodes=episodes, seed=seed)
    else:
        chosen = suite.split(arguments.split)
        episodes = chosen.episodes
        seed = chosen.seed
        resolved = suite.resolve(scenario, arguments.split)
    methods: list[ComparisonMethod] = []
    for name in arguments.controllers:
        if name == "lqr":
            methods.append(
                ComparisonMethod(
                    name,
                    controller_factory=lqr_controller_factory(
                        controller_configurations.lqr
                    ),
                )
            )
        elif name == "nmpc":
            methods.append(
                ComparisonMethod(
                    name,
                    controller_factory=nmpc_controller_factory(
                        controller_configurations.nmpc
                    ),
                )
            )
        else:
            assert arguments.policy is not None
            methods.append(ComparisonMethod(name, policy=load_policy(arguments.policy)))
    runs = run_controlled_comparison(
        scenario,
        tuple(methods),
        backends=arguments.backends,
        episodes=episodes,
        seed=seed,
        resolved=resolved,
    )
    for run in runs:
        print(f"\n[{run.method} | {run.backend}]")
        print("episodes: " + ", ".join(run.episode_fingerprints))
        _print_results(run.results)
    if arguments.save_results:
        assert suite is not None
        from trailer_docking.training.session import file_sha256

        sources: dict[str, Any] = {
            "lqr": {"controller": controller_configurations.specification_id},
            "nmpc": {"controller": controller_configurations.specification_id},
        }
        if arguments.policy is not None:
            policy_path = resolve_policy_path(arguments.policy)
            sources["ppo"] = {
                "checkpoint": policy_path.as_posix(),
                "sha256": file_sha256(policy_path),
            }
        evidence = save_comparison_evidence(
            scenario,
            suite,
            arguments.split,
            resolved,
            runs,
            controller_configurations,
            method_sources={name: sources[name] for name in arguments.controllers},
            purpose=arguments.purpose,
            output_root=arguments.output_root,
        )
        print(f"\nevidence: {evidence.experiment_id}")
        print(f"configuration: {evidence.configuration_path}")
        print(f"results: {evidence.results_directory}")
    return EXIT_OK


def _export(arguments: argparse.Namespace) -> int:
    scenario = _load_scenario(arguments.scenario)
    training: TrainingConfig | None = None
    source: dict[str, Any] = {"checkpoint": arguments.policy.name}
    if arguments.manifest is not None:
        # Without the run's manifest the training settings are unknown and
        # are recorded as such, never filled with defaults.
        try:
            manifest = yaml.safe_load(arguments.manifest.read_text(encoding="utf-8"))
            training = TrainingConfig.from_dict(manifest["config"])
            source["experiment_id"] = manifest.get("experiment_id")
            if arguments.scenario is None:
                scenario = Scenario.from_dict(manifest["scenario"])
        except (OSError, KeyError, TypeError, yaml.YAMLError) as error:
            raise ValueError(
                f"{arguments.manifest} is not a run manifest: {error}"
            ) from error
    path = export_bundle(
        arguments.policy,
        scenario,
        arguments.output,
        training=training,
        source=source,
    )
    print(f"bundle: {path}")
    return EXIT_OK


def _curriculum(arguments: argparse.Namespace) -> int:
    configuration = CurriculumConfiguration.load(arguments.config)
    last_report = 0.0

    def report(stage: int, name: str, progress: TrainingProgress) -> None:
        nonlocal last_report
        now = time.monotonic()
        if now - last_report < PROGRESS_PERIOD and progress.status == "running":
            return
        last_report = now
        print(
            f"[stage {stage + 1}/{len(configuration.stages)}: {name}] "
            f"{_progress_line(progress)}",
            flush=True,
        )

    runner = CurriculumRunner(
        configuration,
        output_root=arguments.output_root,
        initial_policy=arguments.initial_policy,
        start_stage=arguments.start_stage,
        on_progress=report,
    )
    result = runner.run()
    print(f"curriculum: {result.experiment_id}")
    print(f"status: {result.status}")
    print(f"completed stages: {result.completed_stages}")
    print(f"checkpoint: {result.final_checkpoint}")
    print(f"results: {result.results_directory}")
    return EXIT_OK if result.status == "complete" else EXIT_FAILED


def _reward_study(arguments: argparse.Namespace) -> int:
    configuration = PPORewardStudyConfiguration.load(arguments.config)
    last_report = 0.0

    def report(condition: str, seed: int, progress: TrainingProgress) -> None:
        nonlocal last_report
        now = time.monotonic()
        if now - last_report < PROGRESS_PERIOD and progress.status == "running":
            return
        last_report = now
        print(
            f"[condition {condition} | seed {seed}] {_progress_line(progress)}",
            flush=True,
        )

    runner = PPORewardStudyRunner(
        configuration,
        output_root=arguments.output_root,
        initial_policy=arguments.initial_policy,
        on_progress=report,
    )
    result = runner.run()
    print(f"reward study: {result.experiment_id}")
    print(f"status: {result.status}")
    print(f"completed cells: {result.completed_cells}")
    print(f"configuration: {result.configuration_path}")
    print(f"results: {result.results_directory}")
    return EXIT_OK if result.status == "complete" else EXIT_FAILED


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return its exit code."""

    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "train":
            return _train(arguments)
        if arguments.command == "export":
            return _export(arguments)
        if arguments.command == "compare":
            return _compare(arguments)
        if arguments.command == "curriculum":
            return _curriculum(arguments)
        if arguments.command == "reward-study":
            return _reward_study(arguments)
        return _test(arguments)
    except (ConfigurationError, FileNotFoundError, FileExistsError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE
    except MethodUnavailableError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_FAILED
    except ValueError as error:
        # Invalid numeric options (for example --timesteps 0 or --episodes 0)
        # and files that are not policies or bundles.
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE
    except ImportError as error:
        # Loading a saved policy needs PyTorch and Stable-Baselines3, which
        # are optional so the rest of the package installs without them.
        print(
            f"error: {error}. Install the optional training dependencies with"
            " pip install -e '.[train]'.",
            file=sys.stderr,
        )
        return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
