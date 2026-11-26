"""Versioned experiment split validation and resolution."""

from __future__ import annotations

from dataclasses import replace

import pytest

from trailer_docking.evaluation.scenarios import Scenario
from trailer_docking.evaluation.suites import (
    DEFAULT_SUITE_PATH,
    EvaluationSplit,
    ExperimentSuite,
)


def test_default_suite_is_disjoint_and_round_trips(tmp_path) -> None:
    suite = ExperimentSuite.load(DEFAULT_SUITE_PATH)
    assert suite.training_seeds == (0, 1, 2)
    assert suite.validation.episodes == 20
    assert suite.test.episodes == 100
    assert set(suite.training_seeds).isdisjoint(suite.validation.seeds)
    assert set(suite.validation.seeds).isdisjoint(suite.test.seeds)
    assert ExperimentSuite.load(suite.save(tmp_path / "suite.yaml")) == suite
    assert len(suite.fingerprint) == 64


def test_suite_resolves_a_named_split() -> None:
    suite = ExperimentSuite(
        "small-v1",
        (0,),
        EvaluationSplit(10, 2),
        EvaluationSplit(20, 2),
    )
    resolved = suite.resolve(Scenario(), "validation")
    assert tuple(item.seed for item in resolved) == (10, 11)
    with pytest.raises(ValueError, match="split"):
        suite.resolve(Scenario(), "training")


def test_suite_rejects_leakage_and_malformed_splits() -> None:
    base = ExperimentSuite(
        "small-v1", (0,), EvaluationSplit(10, 2), EvaluationSplit(20, 2)
    )
    with pytest.raises(ValueError, match="disjoint"):
        replace(base, test=EvaluationSplit(11, 2))
    with pytest.raises(ValueError, match="episodes"):
        EvaluationSplit(10, 0)
    data = base.to_dict()
    data["unknown"] = 1
    with pytest.raises(ValueError, match="schema"):
        ExperimentSuite.from_dict(data)
