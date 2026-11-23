"""Controlled multi-seed PPO reward studies on one path distribution."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from statistics import mean, pstdev
from types import MappingProxyType
from typing import Any, cast

import yaml

from trailer_docking.evaluation.episodes import ResolvedEpisode, resolve_episodes
from trailer_docking.evaluation.scenarios import Scenario
from trailer_docking.evaluation.suites import ExperimentSuite
from trailer_docking.training.config import TrainingConfig
from trailer_docking.training.evaluate import load_policy, run_docking_test, summarize
from trailer_docking.training.session import (
    STATUS_FINISHED,
    STATUS_IDLE,
    STATUS_RUNNING,
    STATUS_STOPPING,
    TrainingSession,
    _git_revision,
    _package_versions,
    _timestamp,
    file_sha256,
    new_experiment_id,
    runtime_summary,
)

CURRENT_SCHEMA_VERSION = 1
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REWARD_STUDY_PATH = Path("configs/training/ppo_curved_reward_study_v1.yaml")
REWARD_FIELDS = frozenset(
    {
        "path_lateral_weight",
        "path_heading_weight",
        "reward_progress",
        "reward_time",
        "reward_lateral_improvement",
        "reward_lateral_deviation_multiplier",
        "reward_heading_improvement",
        "reward_articulation",
        "reward_steering_rate",
        "reward_success",
        "reward_collision",
        "reward_jackknife",
        "reward_out_of_bounds",
    }
)


@dataclass(frozen=True, slots=True)
class RewardCondition:
    """One named set of reward-only scenario overrides."""

    name: str
    rewards: Mapping[str, float]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("reward condition name cannot be empty")
        values = dict(self.rewards)
        unknown = sorted(set(values) - REWARD_FIELDS)
        if unknown:
            raise ValueError(f"reward condition has non-reward fields {unknown}")
        if not values:
            raise ValueError("reward condition must specify at least one coefficient")
        # ScenarioOverrides performs the canonical numeric/range validation.
        from trailer_docking.evaluation.scenarios import ScenarioOverrides

        ScenarioOverrides(**cast(Any, values))
        object.__setattr__(self, "rewards", MappingProxyType(values))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "rewards": dict(self.rewards)}

    @classmethod
    def from_dict(cls, data: Any) -> RewardCondition:
        if not isinstance(data, dict) or set(data) != {"name", "rewards"}:
            raise ValueError("reward condition fields do not match the schema")
        return cls(str(data["name"]), data["rewards"])


@dataclass(frozen=True, slots=True)
class PPORewardStudyConfiguration:
    """A controlled reward study with owned training and validation seeds."""

    specification_id: str
    training: TrainingConfig
    suite: ExperimentSuite
    training_seeds: tuple[int, ...]
    base_scenario: Scenario
    conditions: tuple[RewardCondition, ...]
    schema_version: int = CURRENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            not isinstance(self.specification_id, str)
            or not self.specification_id.strip()
        ):
            raise ValueError("study specification_id cannot be empty")
        if self.schema_version != CURRENT_SCHEMA_VERSION:
            raise ValueError("unsupported reward-study schema version")
        if self.training.method != "ppo" or self.training.runs != 1:
            raise ValueError("a reward study requires one-run PPO training settings")
        if self.training.evaluation_episodes != 0:
            raise ValueError("reward-study validation is controlled by its suite")
        seeds = tuple(self.training_seeds)
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("training_seeds must be non-empty and unique")
        if any(seed not in self.suite.training_seeds for seed in seeds):
            raise ValueError("reward-study seed is not owned by the experiment suite")
        conditions = tuple(self.conditions)
        if len(conditions) < 2:
            raise ValueError("a reward study needs at least two conditions")
        names = [condition.name for condition in conditions]
        if len(set(names)) != len(names):
            raise ValueError("reward condition names must be unique")
        self.base_scenario.build()
        for condition in conditions:
            self.scenario_for(condition).build()
        object.__setattr__(self, "training_seeds", seeds)
        object.__setattr__(self, "conditions", conditions)

    def scenario_for(self, condition: RewardCondition) -> Scenario:
        return replace(
            self.base_scenario,
            name=f"{self.base_scenario.name}-{condition.name}",
            overrides=replace(
                self.base_scenario.overrides, **cast(Any, dict(condition.rewards))
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "specification_id": self.specification_id,
            "schema_version": self.schema_version,
            "training": self.training.to_dict(),
            "suite": self.suite.to_dict(),
            "training_seeds": list(self.training_seeds),
            "base_scenario": self.base_scenario.to_dict(),
            "conditions": [condition.to_dict() for condition in self.conditions],
        }

    @classmethod
    def from_dict(cls, data: Any) -> PPORewardStudyConfiguration:
        expected = {
            "specification_id",
            "schema_version",
            "training",
            "suite",
            "training_seeds",
            "base_scenario",
            "conditions",
        }
        if not isinstance(data, dict) or set(data) != expected:
            raise ValueError("reward-study fields do not match the schema")
        training = {"evaluation_episodes": 0, **dict(data["training"])}
        return cls(
            specification_id=data["specification_id"],
            schema_version=data["schema_version"],
            training=TrainingConfig.from_dict(training),
            suite=ExperimentSuite.from_dict(data["suite"]),
            training_seeds=tuple(data["training_seeds"]),
            base_scenario=Scenario.from_dict(data["base_scenario"]),
            conditions=tuple(
                RewardCondition.from_dict(item) for item in data["conditions"]
            ),
        )

    @classmethod
    def load(
        cls, path: str | Path = DEFAULT_REWARD_STUDY_PATH
    ) -> PPORewardStudyConfiguration:
        source = Path(path)
        if not source.is_absolute() and not source.exists():
            source = REPOSITORY_ROOT / source
        try:
            data = yaml.safe_load(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
            raise ValueError(f"cannot read reward study {source}: {error}") from error
        return cls.from_dict(data)

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PPORewardStudyResult:
    experiment_id: str
    status: str
    completed_cells: int
    configuration_path: Path
    results_directory: Path


SessionFactory = Callable[..., Any]
TestRunner = Callable[..., Any]
ProgressCallback = Callable[[str, int, Any], None]


def _physical_episode_fingerprint(episode: ResolvedEpisode) -> str:
    content = episode.to_dict()
    content.pop("scenario_fingerprint")
    content.pop("episode_fingerprint")
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


class PPORewardStudyRunner:
    """Train every reward condition on every selected seed, then validate."""

    def __init__(
        self,
        configuration: PPORewardStudyConfiguration,
        *,
        output_root: str | Path = ".",
        initial_policy: str | Path | None = None,
        session_factory: SessionFactory = TrainingSession,
        policy_loader: Callable[[Path], Any] = load_policy,
        test_runner: TestRunner = run_docking_test,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        self.configuration = configuration
        self.output_root = Path(output_root)
        self.initial_policy = None if initial_policy is None else Path(initial_policy)
        self._session_factory = session_factory
        self._policy_loader = policy_loader
        self._test_runner = test_runner
        self._on_progress = on_progress

    def _reserve(self) -> tuple[str, Path, Path]:
        base = new_experiment_id(
            self.configuration.specification_id, method="ppo-study"
        )
        for attempt in range(1, 10_000):
            identifier = base if attempt == 1 else f"{base}-{attempt}"
            target = self.output_root / "results" / identifier
            configuration = (
                self.output_root
                / "configs"
                / "experiments"
                / f"{identifier}.yaml"
            )
            if target.exists() or configuration.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            configuration.parent.mkdir(parents=True, exist_ok=True)
            target.mkdir()
            return identifier, target, configuration
        raise FileExistsError("could not reserve a unique reward-study ID")

    @staticmethod
    def _write_validation(path: Path, results: Any) -> None:
        rows = [item.to_dict() for item in results]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["episode", *rows[0]])
            writer.writeheader()
            for index, row in enumerate(rows):
                writer.writerow({"episode": index, **row})

    def _write_records(self, directory: Path, records: list[dict[str, Any]]) -> None:
        fields = [
            "condition",
            "seed",
            "training_experiment_id",
            "checkpoint",
            "success_rate",
            "mean_total_reward",
            "collision_rate",
            "jackknife_rate",
            "mean_abs_lateral_error",
            "mean_abs_yaw_error",
        ]
        with (directory / "cells.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for record in records:
                validation = record.get("validation", {})
                writer.writerow(
                    {
                        "condition": record["condition"],
                        "seed": record["seed"],
                        "training_experiment_id": record.get(
                            "training_experiment_id"
                        ),
                        "checkpoint": record.get("checkpoint"),
                        **{field: validation.get(field) for field in fields[4:]},
                    }
                )

    def _summary(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        for condition in self.configuration.conditions:
            cells = [
                item
                for item in records
                if item["condition"] == condition.name and "validation" in item
            ]
            rates = [float(item["validation"]["success_rate"]) for item in cells]
            rewards = [
                float(item["validation"]["mean_total_reward"]) for item in cells
            ]
            summaries.append(
                {
                    "condition": condition.name,
                    "completed_seeds": len(cells),
                    "mean_success_rate": mean(rates) if rates else None,
                    "success_rate_std": pstdev(rates) if rates else None,
                    "mean_reward": mean(rewards) if rewards else None,
                    "reward_std": pstdev(rewards) if rewards else None,
                }
            )
        return summaries

    def _record(
        self,
        directory: Path,
        experiment_id: str,
        status: str,
        records: list[dict[str, Any]],
        error: str | None,
    ) -> None:
        manifest = {
            "experiment_id": experiment_id,
            "kind": "ppo-reward-study",
            "status": status,
            "error": error,
            "updated": _timestamp(),
            "configuration": self.configuration.to_dict(),
            "configuration_fingerprint": self.configuration.fingerprint,
            "configuration_path": (
                "configs/experiments/"
                f"{experiment_id}.yaml"
            ),
            "initial_policy_path": (
                None if self.initial_policy is None else self.initial_policy.as_posix()
            ),
            "initial_policy_sha256": file_sha256(self.initial_policy),
            "cells": records,
            "versions": _package_versions(),
            "runtime": runtime_summary(),
            "git": _git_revision(),
            "note": (
                "Conditions share the base scenario, training settings, selected "
                "training seeds, and validation split. Only declared reward "
                "coefficients differ. Validation is for tuning, not final testing."
            ),
        }
        (directory / "manifest.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
        )
        summary = {
            "experiment_id": experiment_id,
            "status": status,
            "completed_cells": sum("validation" in item for item in records),
            "conditions": self._summary(records),
        }
        (directory / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        self._write_records(directory, records)

    def run(self) -> PPORewardStudyResult:
        experiment_id, directory, configuration_path = self._reserve()
        serialized_configuration = yaml.safe_dump(
            self.configuration.to_dict(), sort_keys=False
        )
        configuration_path.write_text(serialized_configuration, encoding="utf-8")
        (directory / "study.yaml").write_text(
            serialized_configuration, encoding="utf-8"
        )
        records: list[dict[str, Any]] = []
        status = "running"
        error: str | None = None
        active_session: Any = None
        canonical_inputs: dict[int, tuple[str, ...]] = {}
        self._record(directory, experiment_id, status, records, error)
        try:
            validation = self.configuration.suite.validation
            for condition in self.configuration.conditions:
                scenario = self.configuration.scenario_for(condition)
                resolved = resolve_episodes(
                    scenario, episodes=validation.episodes, seed=validation.seed
                )
                physical_inputs = tuple(
                    _physical_episode_fingerprint(item) for item in resolved
                )
                previous = canonical_inputs.setdefault(validation.seed, physical_inputs)
                if previous != physical_inputs:
                    raise RuntimeError(
                        "reward conditions resolved different physical validation "
                        "episodes"
                    )
                for seed in self.configuration.training_seeds:
                    training = replace(
                        self.configuration.training,
                        seed=seed,
                        purpose=(
                            f"{self.configuration.specification_id} "
                            f"{condition.name} seed {seed}"
                        ),
                    )
                    session = self._session_factory(
                        scenario,
                        training,
                        output_root=self.output_root,
                        campaign_id=experiment_id,
                        initial_policy=self.initial_policy,
                    )
                    active_session = session
                    session.start()
                    while session.progress.status in (
                        STATUS_IDLE,
                        STATUS_RUNNING,
                        STATUS_STOPPING,
                    ):
                        session.join(0.2)
                        if self._on_progress is not None:
                            self._on_progress(condition.name, seed, session.progress)
                    progress = session.progress
                    active_session = None
                    record: dict[str, Any] = {
                        "condition": condition.name,
                        "rewards": dict(condition.rewards),
                        "seed": seed,
                        "training_experiment_id": progress.experiment_id,
                        "training_status": progress.status,
                        "checkpoint": (
                            None
                            if progress.checkpoint_path is None
                            else progress.checkpoint_path.as_posix()
                        ),
                    }
                    if (
                        progress.status != STATUS_FINISHED
                        or progress.checkpoint_path is None
                    ):
                        records.append(record)
                        status = "failed"
                        error = progress.error or "study training cell did not finish"
                        break
                    policy = self._policy_loader(progress.checkpoint_path)
                    results = self._test_runner(
                        scenario,
                        policy,
                        episodes=validation.episodes,
                        seed=validation.seed,
                        max_episode_steps=training.max_episode_steps,
                        backend=training.backend,
                        resolved_episodes=resolved,
                    )
                    safe_name = condition.name.replace(" ", "-")
                    self._write_validation(
                        directory / f"validation_{safe_name}_seed{seed}.csv", results
                    )
                    record["validation"] = summarize(results)
                    record["physical_episode_fingerprints"] = list(physical_inputs)
                    record["episode_fingerprints"] = [
                        item.episode_fingerprint for item in resolved
                    ]
                    records.append(record)
                    self._record(directory, experiment_id, status, records, None)
                if status == "failed":
                    break
            else:
                status = "complete"
        except KeyboardInterrupt:
            if active_session is not None:
                active_session.stop()
                active_session.join(30)
            status = "stopped"
            error = "reward study interrupted by user"
        except BaseException as exception:
            status = "failed"
            error = str(exception) or type(exception).__name__
        self._record(directory, experiment_id, status, records, error)
        return PPORewardStudyResult(
            experiment_id,
            status,
            sum("validation" in item for item in records),
            configuration_path,
            directory,
        )
