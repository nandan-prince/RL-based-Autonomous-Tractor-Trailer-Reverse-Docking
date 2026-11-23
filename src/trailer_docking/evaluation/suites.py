"""Versioned train/validation/test seed partitions for controlled studies."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml

from trailer_docking.evaluation.episodes import ResolvedEpisode, resolve_episodes
from trailer_docking.evaluation.scenarios import Scenario

CURRENT_SCHEMA_VERSION = 1
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SPLIT_NAMES = ("validation", "test")
DEFAULT_SUITE_PATH = Path("configs/experiments/compact_docking_suite_v1.yaml")


@dataclass(frozen=True, slots=True)
class EvaluationSplit:
    """One contiguous, immutable evaluation seed range."""

    seed: int
    episodes: int

    def __post_init__(self) -> None:
        for name in ("seed", "episodes"):
            value = getattr(self, name)
            minimum = 0 if name == "seed" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(range(self.seed, self.seed + self.episodes))

    def to_dict(self) -> dict[str, int]:
        return {"seed": self.seed, "episodes": self.episodes}

    @classmethod
    def from_dict(cls, data: Any) -> EvaluationSplit:
        if not isinstance(data, dict) or set(data) != {"seed", "episodes"}:
            raise ValueError("evaluation split needs exactly seed and episodes")
        return cls(seed=data["seed"], episodes=data["episodes"])


@dataclass(frozen=True, slots=True)
class ExperimentSuite:
    """Frozen seed ownership for training, tuning, and final evaluation."""

    specification_id: str
    training_seeds: tuple[int, ...]
    validation: EvaluationSplit
    test: EvaluationSplit
    schema_version: int = CURRENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            not isinstance(self.specification_id, str)
            or not self.specification_id.strip()
        ):
            raise ValueError("suite specification_id cannot be empty")
        if self.schema_version != CURRENT_SCHEMA_VERSION:
            raise ValueError("unsupported experiment-suite schema version")
        seeds = tuple(self.training_seeds)
        if not seeds or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in seeds
        ):
            raise ValueError("training_seeds must be non-empty non-negative integers")
        if len(set(seeds)) != len(seeds):
            raise ValueError("training_seeds cannot contain duplicates")
        object.__setattr__(self, "training_seeds", seeds)
        train = set(seeds)
        validation = set(self.validation.seeds)
        test = set(self.test.seeds)
        if train & validation or train & test or validation & test:
            raise ValueError("training, validation, and test seeds must be disjoint")

    def split(self, name: str) -> EvaluationSplit:
        if name not in SPLIT_NAMES:
            raise ValueError(f"split must be one of {', '.join(SPLIT_NAMES)}")
        return self.validation if name == "validation" else self.test

    def resolve(self, scenario: Scenario, name: str) -> tuple[ResolvedEpisode, ...]:
        chosen = self.split(name)
        return resolve_episodes(scenario, episodes=chosen.episodes, seed=chosen.seed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "specification_id": self.specification_id,
            "schema_version": self.schema_version,
            "training_seeds": list(self.training_seeds),
            "validation": self.validation.to_dict(),
            "test": self.test.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Any) -> ExperimentSuite:
        expected = {
            "specification_id",
            "schema_version",
            "training_seeds",
            "validation",
            "test",
        }
        if not isinstance(data, dict) or set(data) != expected:
            raise ValueError("experiment suite fields do not match the schema")
        return cls(
            specification_id=data["specification_id"],
            schema_version=data["schema_version"],
            training_seeds=tuple(data["training_seeds"]),
            validation=EvaluationSplit.from_dict(data["validation"]),
            test=EvaluationSplit.from_dict(data["test"]),
        )

    @classmethod
    def load(cls, path: str | Path = DEFAULT_SUITE_PATH) -> ExperimentSuite:
        source = Path(path)
        if not source.is_absolute() and not source.exists():
            source = REPOSITORY_ROOT / source
        try:
            data = yaml.safe_load(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
            raise ValueError(
                f"cannot read experiment suite {source}: {error}"
            ) from error
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
        serialized = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return sha256(serialized.encode("utf-8")).hexdigest()
