"""Controlled PPO reward studies on mixed predefined curved paths."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from trailer_docking.evaluation.scenarios import (
    DomainRandomization,
    Scenario,
    StartDistribution,
)
from trailer_docking.evaluation.suites import EvaluationSplit, ExperimentSuite
from trailer_docking.training import cli
from trailer_docking.training.config import TrainingConfig
from trailer_docking.training.evaluate import EpisodeResult
from trailer_docking.training.study import (
    DEFAULT_REWARD_STUDY_PATH,
    PPORewardStudyConfiguration,
    PPORewardStudyRunner,
    RewardCondition,
)


def _result(success: bool) -> EpisodeResult:
    return EpisodeResult(
        outcome="success" if success else "time_limit",
        success=success,
        steps=2,
        total_reward=1.0 if success else -1.0,
        final_longitudinal_error=0.0,
        final_lateral_error=0.0,
        final_yaw_error=0.0,
        max_abs_articulation=0.0,
    )


def _configuration() -> PPORewardStudyConfiguration:
    return PPORewardStudyConfiguration(
        "unit-curved-reward-study-v1",
        TrainingConfig(
            total_timesteps=8,
            n_steps=8,
            batch_size=4,
            evaluation_episodes=0,
        ),
        ExperimentSuite(
            "unit-suite-v1", (0,), EvaluationSplit(100, 2), EvaluationSplit(200, 2)
        ),
        (0,),
        Scenario(
            name="curved-unit",
            start=StartDistribution((5, 5), (-0.5, 0.5), (0, 0), (0, 0)),
            randomization=DomainRandomization(
                reference_planners=("curve_gentle", "curve_late")
            ),
            planner="curve_gentle",
        ),
        (
            RewardCondition("baseline", {"path_lateral_weight": 0.5}),
            RewardCondition("tracking", {"path_lateral_weight": 1.5}),
        ),
    )


def test_default_reward_study_loads_and_is_controlled() -> None:
    configuration = PPORewardStudyConfiguration.load(DEFAULT_REWARD_STUDY_PATH)
    assert [item.name for item in configuration.conditions] == [
        "baseline",
        "tracking-heavy",
        "articulation-aware",
    ]
    assert configuration.training_seeds == (0, 1, 2)
    scenarios = [configuration.scenario_for(item) for item in configuration.conditions]
    assert all(
        scenario.randomization == configuration.base_scenario.randomization
        for scenario in scenarios
    )
    assert len(configuration.fingerprint) == 64


def test_reward_condition_rejects_non_reward_changes() -> None:
    with pytest.raises(ValueError, match="non-reward"):
        RewardCondition("bad", {"reverse_speed": 0.1})


def test_reward_study_runs_every_condition_on_the_same_inputs(tmp_path) -> None:
    calls: list[dict[str, Any]] = []

    class FakeSession:
        def __init__(self, scenario, config, **options):
            checkpoint = tmp_path / f"policy-{len(calls)}.zip"
            checkpoint.write_bytes(b"weights")
            calls.append({"scenario": scenario, "config": config, **options})
            self.progress = SimpleNamespace(
                status="finished",
                checkpoint_path=checkpoint,
                experiment_id=f"cell-{len(calls)}",
                timesteps=config.total_timesteps,
                error=None,
            )

        def start(self) -> None:
            pass

        def join(self, timeout: float | None = None) -> None:
            pass

    def test_runner(scenario, policy, **options):
        return [_result(True) for _ in range(options["episodes"])]

    result = PPORewardStudyRunner(
        _configuration(),
        output_root=tmp_path,
        session_factory=FakeSession,
        policy_loader=lambda path: path.name,
        test_runner=test_runner,
    ).run()

    assert result.status == "complete"
    assert result.completed_cells == 2
    assert result.configuration_path.is_file()
    assert len(calls) == 2
    manifest = yaml.safe_load(
        (result.results_directory / "manifest.yaml").read_text(encoding="utf-8")
    )
    first, second = manifest["cells"]
    assert first["physical_episode_fingerprints"] == second[
        "physical_episode_fingerprints"
    ]
    assert first["episode_fingerprints"] != second["episode_fingerprints"]
    assert (result.results_directory / "cells.csv").is_file()
    assert (result.results_directory / "summary.json").is_file()


def test_reward_study_cli_runs_a_tiny_real_study(tmp_path, capsys) -> None:
    pytest.importorskip("stable_baselines3")
    base = _configuration()
    tiny = replace(
        base,
        specification_id="tiny-curved-reward-study-v1",
        training=replace(
            base.training,
            total_timesteps=64,
            n_steps=64,
            batch_size=32,
            max_episode_steps=10,
            checkpoint_interval=64,
        ),
        suite=ExperimentSuite(
            "tiny-suite-v1", (0,), EvaluationSplit(100, 1), EvaluationSplit(200, 1)
        ),
    )
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(tiny.to_dict(), sort_keys=False), encoding="utf-8")

    assert cli.main(
        [
            "reward-study",
            "--config",
            str(path),
            "--output-root",
            str(tmp_path),
        ]
    ) == 0
    output = capsys.readouterr().out
    assert "status: complete" in output
    assert "completed cells: 2" in output
