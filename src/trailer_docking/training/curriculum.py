"""Configured, gated PPO curriculum with safe policy warm starts."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml

from trailer_docking.evaluation.episodes import resolve_episodes
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
DEFAULT_CURRICULUM_PATH = Path("configs/training/ppo_curriculum_v1.yaml")


@dataclass(frozen=True, slots=True)
class CurriculumStage:
    """One training distribution, budget, and advancement gate."""

    name: str
    timesteps: int
    minimum_validation_success: float
    scenario: Scenario

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("curriculum stage name cannot be empty")
        if (
            isinstance(self.timesteps, bool)
            or not isinstance(self.timesteps, int)
            or self.timesteps < 1
        ):
            raise ValueError("curriculum stage timesteps must be positive")
        threshold = self.minimum_validation_success
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not 0.0 <= threshold <= 1.0
        ):
            raise ValueError("minimum_validation_success must be between 0 and 1")
        self.scenario.build()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "timesteps": self.timesteps,
            "minimum_validation_success": self.minimum_validation_success,
            "scenario": self.scenario.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Any) -> CurriculumStage:
        expected = {
            "name",
            "timesteps",
            "minimum_validation_success",
            "scenario",
        }
        if not isinstance(data, dict) or set(data) != expected:
            raise ValueError("curriculum stage fields do not match the schema")
        return cls(
            name=data["name"],
            timesteps=data["timesteps"],
            minimum_validation_success=data["minimum_validation_success"],
            scenario=Scenario.from_dict(data["scenario"]),
        )


@dataclass(frozen=True, slots=True)
class CurriculumConfiguration:
    """Complete staged PPO definition, including its validation ownership."""

    specification_id: str
    training: TrainingConfig
    suite: ExperimentSuite
    stages: tuple[CurriculumStage, ...]
    schema_version: int = CURRENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            not isinstance(self.specification_id, str)
            or not self.specification_id.strip()
        ):
            raise ValueError("curriculum specification_id cannot be empty")
        if self.schema_version != CURRENT_SCHEMA_VERSION:
            raise ValueError("unsupported curriculum schema version")
        if self.training.method != "ppo":
            raise ValueError("a curriculum can train only PPO")
        if self.training.runs != 1:
            raise ValueError("a curriculum configuration represents one training seed")
        if self.training.evaluation_episodes != 0:
            raise ValueError("curriculum validation is controlled by its suite")
        if self.training.seed not in self.suite.training_seeds:
            raise ValueError("curriculum training seed is not owned by the suite")
        stages = tuple(self.stages)
        if not stages:
            raise ValueError("curriculum needs at least one stage")
        names = [item.name for item in stages]
        if len(set(names)) != len(names):
            raise ValueError("curriculum stage names must be unique")
        object.__setattr__(self, "stages", stages)

    def to_dict(self) -> dict[str, Any]:
        return {
            "specification_id": self.specification_id,
            "schema_version": self.schema_version,
            "training": self.training.to_dict(),
            "suite": self.suite.to_dict(),
            "stages": [item.to_dict() for item in self.stages],
        }

    @classmethod
    def from_dict(cls, data: Any) -> CurriculumConfiguration:
        expected = {"specification_id", "schema_version", "training", "suite", "stages"}
        if not isinstance(data, dict) or set(data) != expected:
            raise ValueError("curriculum fields do not match the schema")
        training_data = {"evaluation_episodes": 0, **dict(data["training"])}
        return cls(
            specification_id=data["specification_id"],
            schema_version=data["schema_version"],
            training=TrainingConfig.from_dict(training_data),
            suite=ExperimentSuite.from_dict(data["suite"]),
            stages=tuple(CurriculumStage.from_dict(item) for item in data["stages"]),
        )

    @classmethod
    def load(
        cls, path: str | Path = DEFAULT_CURRICULUM_PATH
    ) -> CurriculumConfiguration:
        source = Path(path)
        if not source.is_absolute() and not source.exists():
            source = REPOSITORY_ROOT / source
        try:
            data = yaml.safe_load(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
            raise ValueError(f"cannot read curriculum {source}: {error}") from error
        return cls.from_dict(data)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8"
        )
        return target

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CurriculumResult:
    """Terminal status and final checkpoint of one curriculum execution."""

    experiment_id: str
    status: str
    completed_stages: int
    final_checkpoint: Path | None
    results_directory: Path


SessionFactory = Callable[..., Any]
TestRunner = Callable[..., Any]
ProgressCallback = Callable[[int, str, Any], None]


class CurriculumRunner:
    """Execute stages sequentially and advance only through validation gates."""

    def __init__(
        self,
        configuration: CurriculumConfiguration,
        *,
        output_root: str | Path = ".",
        initial_policy: str | Path | None = None,
        start_stage: int = 0,
        session_factory: SessionFactory = TrainingSession,
        policy_loader: Callable[[Path], Any] = load_policy,
        test_runner: TestRunner = run_docking_test,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        if not 0 <= start_stage < len(configuration.stages):
            raise ValueError("start_stage is outside the configured curriculum")
        if start_stage > 0 and initial_policy is None:
            raise ValueError("starting after stage 0 requires an initial policy")
        self.configuration = configuration
        self.output_root = Path(output_root)
        self.initial_policy = None if initial_policy is None else Path(initial_policy)
        self.start_stage = start_stage
        self._session_factory = session_factory
        self._policy_loader = policy_loader
        self._test_runner = test_runner
        self._on_progress = on_progress

    def _reserve(self) -> tuple[str, Path]:
        base = new_experiment_id(
            self.configuration.specification_id, method="ppo-curriculum"
        )
        for attempt in range(1, 10_000):
            identifier = base if attempt == 1 else f"{base}-{attempt}"
            target = self.output_root / "results" / identifier
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.mkdir()
            return identifier, target
        raise FileExistsError("could not reserve a unique curriculum experiment ID")

    @staticmethod
    def _write_validation(path: Path, results: Any) -> None:
        rows = [item.to_dict() for item in results]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["episode", *rows[0]])
            writer.writeheader()
            for index, row in enumerate(rows):
                writer.writerow({"episode": index, **row})

    def _record(
        self,
        directory: Path,
        experiment_id: str,
        status: str,
        stages: list[dict[str, Any]],
        checkpoint: Path | None,
        error: str | None,
    ) -> None:
        manifest = {
            "experiment_id": experiment_id,
            "kind": "ppo-curriculum",
            "status": status,
            "error": error,
            "updated": _timestamp(),
            "configuration": self.configuration.to_dict(),
            "configuration_fingerprint": self.configuration.fingerprint,
            "start_stage": self.start_stage,
            "initial_policy_path": (
                None if self.initial_policy is None else self.initial_policy.as_posix()
            ),
            "initial_policy_sha256": file_sha256(self.initial_policy),
            "final_checkpoint": (
                None if checkpoint is None else checkpoint.as_posix()
            ),
            "final_checkpoint_sha256": file_sha256(checkpoint),
            "stage_runs": stages,
            "versions": _package_versions(),
            "runtime": runtime_summary(),
            "git": _git_revision(),
            "note": (
                "Stages warm-start policy weights only; optimizer and learning-rate "
                "schedule state restart at every stage. Validation seeds may guide "
                "advancement and must not be reported as held-out test results."
            ),
        }
        (directory / "manifest.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
        )
        summary = {
            "experiment_id": experiment_id,
            "status": status,
            "completed_stages": sum(
                item.get("gate_passed", False) for item in stages
            ),
            "stages": stages,
        }
        (directory / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )

    def run(self) -> CurriculumResult:
        experiment_id, directory = self._reserve()
        (directory / "curriculum.yaml").write_text(
            yaml.safe_dump(self.configuration.to_dict(), sort_keys=False),
            encoding="utf-8",
        )
        stage_records: list[dict[str, Any]] = []
        checkpoint = self.initial_policy
        status = "running"
        error: str | None = None
        active_session: Any = None
        self._record(directory, experiment_id, status, stage_records, checkpoint, error)
        try:
            validation = self.configuration.suite.validation
            for index in range(self.start_stage, len(self.configuration.stages)):
                stage = self.configuration.stages[index]
                training = replace(
                    self.configuration.training,
                    total_timesteps=stage.timesteps,
                    purpose=(
                        f"{self.configuration.specification_id} stage {index + 1} "
                        f"{stage.name}"
                    ),
                )
                session = self._session_factory(
                    stage.scenario,
                    training,
                    output_root=self.output_root,
                    campaign_id=experiment_id,
                    initial_policy=checkpoint,
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
                        self._on_progress(index, stage.name, session.progress)
                progress = session.progress
                active_session = None
                record: dict[str, Any] = {
                    "stage": index,
                    "name": stage.name,
                    "training_experiment_id": progress.experiment_id,
                    "training_status": progress.status,
                    "training_timesteps": progress.timesteps,
                    "checkpoint": (
                        None
                        if progress.checkpoint_path is None
                        else progress.checkpoint_path.as_posix()
                    ),
                    "gate": stage.minimum_validation_success,
                }
                if (
                    progress.status != STATUS_FINISHED
                    or progress.checkpoint_path is None
                ):
                    record["gate_passed"] = False
                    stage_records.append(record)
                    status = "failed"
                    error = progress.error or "stage training did not finish"
                    break
                checkpoint = progress.checkpoint_path
                policy = self._policy_loader(checkpoint)
                resolved = resolve_episodes(
                    stage.scenario,
                    episodes=validation.episodes,
                    seed=validation.seed,
                )
                results = self._test_runner(
                    stage.scenario,
                    policy,
                    episodes=validation.episodes,
                    seed=validation.seed,
                    max_episode_steps=training.max_episode_steps,
                    backend=training.backend,
                    resolved_episodes=resolved,
                )
                self._write_validation(
                    directory / f"validation_stage{index + 1}.csv", results
                )
                validation_summary = summarize(results)
                passed = (
                    float(validation_summary["success_rate"])
                    >= stage.minimum_validation_success
                )
                record["validation"] = validation_summary
                record["episode_fingerprints"] = [
                    item.episode_fingerprint for item in resolved
                ]
                record["gate_passed"] = passed
                stage_records.append(record)
                self._record(
                    directory,
                    experiment_id,
                    "running" if passed else "threshold_not_met",
                    stage_records,
                    checkpoint,
                    None,
                )
                if not passed:
                    status = "threshold_not_met"
                    break
            else:
                status = "complete"
        except KeyboardInterrupt:
            if active_session is not None:
                active_session.stop()
                active_session.join(30)
            status = "stopped"
            error = "curriculum interrupted by user"
        except BaseException as exception:
            status = "failed"
            error = str(exception) or type(exception).__name__
        self._record(directory, experiment_id, status, stage_records, checkpoint, error)
        return CurriculumResult(
            experiment_id=experiment_id,
            status=status,
            completed_stages=sum(
                item.get("gate_passed", False) for item in stage_records
            ),
            final_checkpoint=checkpoint,
            results_directory=directory,
        )
