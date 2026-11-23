"""Configured PPO curriculum, validation gates, and warm-start chaining."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from trailer_docking.evaluation.scenarios import Scenario, StartDistribution
from trailer_docking.evaluation.suites import EvaluationSplit, ExperimentSuite
from trailer_docking.training import cli
from trailer_docking.training.config import TrainingConfig
from trailer_docking.training.curriculum import (
    DEFAULT_CURRICULUM_PATH,
    CurriculumConfiguration,
    CurriculumRunner,
    CurriculumStage,
)
from trailer_docking.training.evaluate import EpisodeResult


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


def _configuration() -> CurriculumConfiguration:
    scenario = Scenario(
        name="curriculum-unit",
        start=StartDistribution((4, 4), (0, 0), (0, 0), (0, 0)),
    )
    return CurriculumConfiguration(
        "unit-curriculum-v1",
        TrainingConfig(
            total_timesteps=8,
            n_steps=8,
            batch_size=4,
            evaluation_episodes=0,
        ),
        ExperimentSuite(
            "unit-suite-v1", (0,), EvaluationSplit(100, 2), EvaluationSplit(200, 2)
        ),
        (
            CurriculumStage("first", 8, 0.5, scenario),
            CurriculumStage("second", 8, 1.0, scenario),
            CurriculumStage("third", 8, 0.0, scenario),
        ),
    )


def test_default_curriculum_loads_and_round_trips(tmp_path) -> None:
    configuration = CurriculumConfiguration.load(DEFAULT_CURRICULUM_PATH)
    assert [stage.name for stage in configuration.stages] == [
        "aligned-straight",
        "perturbed-straight",
        "randomized-docking",
    ]
    assert configuration.training.evaluation_episodes == 0
    path = configuration.save(tmp_path / "curriculum.yaml")
    assert CurriculumConfiguration.load(path) == configuration
    assert len(configuration.fingerprint) == 64


def test_curriculum_chains_checkpoints_and_stops_at_failed_gate(tmp_path) -> None:
    calls: list[dict[str, Any]] = []
    validation_calls = 0

    class FakeSession:
        def __init__(self, scenario, config, **options):
            checkpoint = tmp_path / f"policy-{len(calls)}.zip"
            checkpoint.write_bytes(b"weights")
            calls.append({"scenario": scenario, "config": config, **options})
            self.progress = SimpleNamespace(
                status="finished",
                checkpoint_path=checkpoint,
                experiment_id=f"stage-{len(calls)}",
                timesteps=config.total_timesteps,
                error=None,
            )

        def start(self) -> None:
            pass

        def join(self) -> None:
            pass

    def test_runner(scenario, policy, **options):
        nonlocal validation_calls
        validation_calls += 1
        successes = 2 if validation_calls == 1 else 1
        return [_result(index < successes) for index in range(options["episodes"])]

    runner = CurriculumRunner(
        _configuration(),
        output_root=tmp_path,
        session_factory=FakeSession,
        policy_loader=lambda path: path.name,
        test_runner=test_runner,
    )
    result = runner.run()
    assert result.status == "threshold_not_met"
    assert result.completed_stages == 1
    assert len(calls) == 2
    assert calls[0]["initial_policy"] is None
    assert calls[1]["initial_policy"] == tmp_path / "policy-0.zip"
    assert (result.results_directory / "validation_stage1.csv").is_file()
    assert (result.results_directory / "validation_stage2.csv").is_file()
    assert not (result.results_directory / "validation_stage3.csv").exists()
    manifest = yaml.safe_load(
        (result.results_directory / "manifest.yaml").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "threshold_not_met"
    assert len(manifest["stage_runs"]) == 2


def test_curriculum_resume_requires_policy() -> None:
    with pytest.raises(ValueError, match="initial policy"):
        CurriculumRunner(_configuration(), start_stage=1)


def test_curriculum_cli_runs_a_tiny_real_stage(tmp_path, capsys) -> None:
    pytest.importorskip("stable_baselines3")
    base = _configuration()
    tiny = CurriculumConfiguration(
        "tiny-cli-curriculum-v1",
        replace(
            base.training,
            total_timesteps=64,
            n_steps=64,
            batch_size=32,
            max_episode_steps=10,
            checkpoint_interval=64,
        ),
        ExperimentSuite(
            "tiny-cli-suite-v1",
            (0,),
            EvaluationSplit(100, 1),
            EvaluationSplit(200, 1),
        ),
        (replace(base.stages[0], timesteps=64, minimum_validation_success=0.0),),
    )
    path = tiny.save(tmp_path / "tiny-curriculum.yaml")
    assert cli.main(
        [
            "curriculum",
            "--config",
            str(path),
            "--output-root",
            str(tmp_path),
        ]
    ) == 0
    output = capsys.readouterr().out
    assert "status: complete" in output
    assert "completed stages: 1" in output
