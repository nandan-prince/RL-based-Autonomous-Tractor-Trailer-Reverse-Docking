"""Paired controller comparison over one frozen set of episodes."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from trailer_docking.controllers.configuration import ControllerConfigurations
from trailer_docking.evaluation.episodes import ResolvedEpisode, resolve_episodes
from trailer_docking.evaluation.scenarios import Scenario
from trailer_docking.evaluation.suites import ExperimentSuite
from trailer_docking.training.evaluate import (
    EpisodeResult,
    Policy,
    run_docking_test,
    summarize,
)
from trailer_docking.training.methods import ControllerFactory


@dataclass(frozen=True, slots=True)
class ComparisonMethod:
    """One policy or controller factory participating in a comparison."""

    name: str
    policy: Policy | None = None
    controller_factory: ControllerFactory | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("comparison method name cannot be empty")
        if (self.policy is None) == (self.controller_factory is None):
            raise ValueError("comparison method needs exactly one control source")


@dataclass(frozen=True, slots=True)
class ComparisonRun:
    """Results for one method/backend cell of a paired experiment."""

    method: str
    backend: str
    episode_fingerprints: tuple[str, ...]
    results: tuple[EpisodeResult, ...]


@dataclass(frozen=True, slots=True)
class ComparisonEvidence:
    """Paths and ID of one immutable formal comparison package."""

    experiment_id: str
    configuration_path: Path
    results_directory: Path


def run_controlled_comparison(
    scenario: Scenario,
    methods: tuple[ComparisonMethod, ...],
    *,
    backends: tuple[str, ...] = ("2d",),
    episodes: int = 1,
    seed: int = 0,
    max_episode_steps: int | None = 900,
    resolved: tuple[ResolvedEpisode, ...] | None = None,
) -> tuple[ComparisonRun, ...]:
    """Evaluate every method/backend cell on identical resolved episodes."""

    if not methods:
        raise ValueError("comparison needs at least one method")
    if not backends:
        raise ValueError("comparison needs at least one backend")
    episode_set = (
        resolve_episodes(scenario, episodes=episodes, seed=seed)
        if resolved is None
        else resolved
    )
    if len(episode_set) != episodes:
        raise ValueError("resolved episode count must equal episodes")
    fingerprints = tuple(item.episode_fingerprint for item in episode_set)
    runs: list[ComparisonRun] = []
    for backend in backends:
        for method in methods:
            results = run_docking_test(
                scenario,
                method.policy,
                episodes=episodes,
                seed=seed,
                max_episode_steps=max_episode_steps,
                backend=backend,
                controller_factory=method.controller_factory,
                resolved_episodes=episode_set,
            )
            runs.append(
                ComparisonRun(
                    method=method.name,
                    backend=backend,
                    episode_fingerprints=fingerprints,
                    results=tuple(results),
                )
            )
    return tuple(runs)


def _reserve_evidence_paths(output_root: Path, purpose: str) -> ComparisonEvidence:
    from trailer_docking.training.session import new_experiment_id

    base = new_experiment_id(purpose, method="comparison")
    for attempt in range(1, 10_000):
        experiment_id = base if attempt == 1 else f"{base}-{attempt}"
        configuration = (
            output_root / "configs" / "experiments" / f"{experiment_id}.yaml"
        )
        results = output_root / "results" / experiment_id
        if configuration.exists() or results.exists():
            continue
        configuration.parent.mkdir(parents=True, exist_ok=True)
        results.parent.mkdir(parents=True, exist_ok=True)
        results.mkdir()
        return ComparisonEvidence(experiment_id, configuration, results)
    raise FileExistsError("could not reserve a unique comparison experiment ID")


def save_comparison_evidence(
    scenario: Scenario,
    suite: ExperimentSuite,
    split: str,
    resolved: tuple[ResolvedEpisode, ...],
    runs: tuple[ComparisonRun, ...],
    controller_configurations: ControllerConfigurations,
    *,
    method_sources: dict[str, Any] | None = None,
    purpose: str = "controller-comparison",
    max_episode_steps: int | None = 900,
    output_root: str | Path = ".",
) -> ComparisonEvidence:
    """Write configuration, provenance, metrics, summary, and caveats once."""

    chosen = suite.split(split)
    if len(resolved) != chosen.episodes:
        raise ValueError("resolved episodes do not match the selected suite split")
    expected = tuple(item.episode_fingerprint for item in resolved)
    if not runs or any(run.episode_fingerprints != expected for run in runs):
        raise ValueError("comparison runs do not share the resolved episode set")
    evidence = _reserve_evidence_paths(Path(output_root), purpose)
    sources = dict(method_sources or {})
    backends = tuple(dict.fromkeys(run.backend for run in runs))
    methods = tuple(dict.fromkeys(run.method for run in runs))
    configuration = {
        "experiment_id": evidence.experiment_id,
        "kind": "paired-controller-comparison",
        "purpose": purpose,
        "scenario": scenario.to_dict(),
        "scenario_fingerprint": scenario.fingerprint,
        "suite": suite.to_dict(),
        "suite_fingerprint": suite.fingerprint,
        "split": split,
        "resolved_episodes": [item.to_dict() for item in resolved],
        "controllers": controller_configurations.to_dict(),
        "controller_fingerprint": controller_configurations.fingerprint,
        "method_sources": sources,
        "methods": list(methods),
        "backends": list(backends),
        "max_episode_steps": max_episode_steps,
    }
    evidence.configuration_path.write_text(
        yaml.safe_dump(configuration, sort_keys=False), encoding="utf-8"
    )

    rows: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    for run in runs:
        cells.append(
            {
                "method": run.method,
                "backend": run.backend,
                **summarize(run.results),
            }
        )
        for index, result in enumerate(run.results):
            rows.append(
                {
                    "method": run.method,
                    "backend": run.backend,
                    "episode": index,
                    **result.to_dict(),
                }
            )
    metrics_path = evidence.results_directory / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "experiment_id": evidence.experiment_id,
        "status": "complete",
        "split": split,
        "episodes_per_cell": chosen.episodes,
        "cells": cells,
    }
    (evidence.results_directory / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    from trailer_docking.training.session import (
        _git_revision,
        _mujoco_tuning,
        _package_versions,
        _timestamp,
        runtime_summary,
    )

    manifest = {
        "experiment_id": evidence.experiment_id,
        "kind": "paired-controller-comparison",
        "status": "complete",
        "finished": _timestamp(),
        "configuration_path": evidence.configuration_path.relative_to(
            Path(output_root)
        ).as_posix(),
        "scenario_fingerprint": scenario.fingerprint,
        "suite_fingerprint": suite.fingerprint,
        "controller_fingerprint": controller_configurations.fingerprint,
        "episode_fingerprints": list(expected),
        "methods": list(methods),
        "backends": list(backends),
        "versions": _package_versions(),
        "runtime": runtime_summary(),
        "git": _git_revision(),
        "mujoco_tuning": _mujoco_tuning("3d" if "3d" in backends else "2d"),
        "metrics_path": "metrics.csv",
        "summary_path": "summary.json",
        "notes_path": "notes.md",
    }
    (evidence.results_directory / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    note = (
        f"# {evidence.experiment_id}\n\n"
        f"Paired `{split}` evaluation from `{suite.specification_id}`. Every "
        "method/backend cell used the exact resolved episode fingerprints "
        "stored in the experiment configuration.\n\n"
        "Interpretation remains limited by the scenario distribution, simulator "
        "fidelity, controller tuning, and any method-source caveats recorded in "
        "the configuration. Do not relabel validation results as final test "
        "results.\n"
    )
    (evidence.results_directory / "notes.md").write_text(note, encoding="utf-8")
    return evidence
