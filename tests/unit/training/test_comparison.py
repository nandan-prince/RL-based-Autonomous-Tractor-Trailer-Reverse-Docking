"""Paired comparison orchestration tests."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from trailer_docking.controllers.configuration import ControllerConfigurations
from trailer_docking.evaluation.episodes import resolve_episodes
from trailer_docking.evaluation.scenarios import Scenario
from trailer_docking.evaluation.suites import EvaluationSplit, ExperimentSuite
from trailer_docking.training.comparison import (
    ComparisonMethod,
    run_controlled_comparison,
    save_comparison_evidence,
)


def _zero(_: np.ndarray) -> np.ndarray:
    return np.zeros((1,), dtype=np.float32)


def test_comparison_reuses_exact_episode_set(aligned_scenario: Scenario) -> None:
    resolved = resolve_episodes(aligned_scenario, episodes=2, seed=21)
    runs = run_controlled_comparison(
        aligned_scenario,
        (
            ComparisonMethod("first", policy=_zero),
            ComparisonMethod("second", policy=_zero),
        ),
        episodes=2,
        seed=999,
        max_episode_steps=2,
        resolved=resolved,
    )

    expected = tuple(item.episode_fingerprint for item in resolved)
    assert len(runs) == 2
    assert all(run.episode_fingerprints == expected for run in runs)
    assert all(
        tuple(result.episode_fingerprint for result in run.results) == expected
        for run in runs
    )
    assert all(
        tuple(result.episode_seed for result in run.results) == (21, 22)
        for run in runs
    )


def test_comparison_method_requires_one_source() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        ComparisonMethod("none")
    with pytest.raises(ValueError, match="exactly one"):
        ComparisonMethod("both", policy=_zero, controller_factory=lambda _: None)  # type: ignore[arg-type]


def test_comparison_evidence_is_complete_and_immutable(
    aligned_scenario: Scenario, tmp_path
) -> None:
    suite = ExperimentSuite(
        "unit-suite-v1", (0,), EvaluationSplit(21, 2), EvaluationSplit(31, 2)
    )
    resolved = suite.resolve(aligned_scenario, "validation")
    runs = run_controlled_comparison(
        aligned_scenario,
        (ComparisonMethod("zero", policy=_zero),),
        episodes=2,
        max_episode_steps=2,
        resolved=resolved,
    )
    evidence = save_comparison_evidence(
        aligned_scenario,
        suite,
        "validation",
        resolved,
        runs,
        ControllerConfigurations(),
        method_sources={"zero": {"kind": "unit-policy"}},
        output_root=tmp_path,
        purpose="unit comparison",
        max_episode_steps=2,
    )
    assert evidence.configuration_path.is_file()
    assert evidence.results_directory.is_dir()
    configuration = yaml.safe_load(
        evidence.configuration_path.read_text(encoding="utf-8")
    )
    assert configuration["suite_fingerprint"] == suite.fingerprint
    assert len(configuration["resolved_episodes"]) == 2
    manifest = yaml.safe_load(
        (evidence.results_directory / "manifest.yaml").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "complete"
    assert manifest["episode_fingerprints"] == [
        item.episode_fingerprint for item in resolved
    ]
    assert (evidence.results_directory / "metrics.csv").is_file()
    assert (evidence.results_directory / "summary.json").is_file()
    assert (evidence.results_directory / "notes.md").is_file()

    repeated = save_comparison_evidence(
        aligned_scenario,
        suite,
        "validation",
        resolved,
        runs,
        ControllerConfigurations(),
        output_root=tmp_path,
        purpose="unit comparison",
        max_episode_steps=2,
    )
    assert repeated.experiment_id != evidence.experiment_id
