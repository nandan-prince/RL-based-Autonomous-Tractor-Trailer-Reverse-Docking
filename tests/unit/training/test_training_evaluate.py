import numpy as np
import pytest

pytest.importorskip("gymnasium")

from trailer_docking.core.state import VehicleState
from trailer_docking.evaluation.scenarios import Scenario
from trailer_docking.evaluation.suites import EvaluationSplit, ExperimentSuite
from trailer_docking.training import cli
from trailer_docking.training.evaluate import (
    EpisodeResult,
    load_policy,
    resolve_policy_path,
    run_docking_test,
    straight_reverse_policy,
    summarize,
)


def _result(
    outcome: str, *, lateral: float = 0.0, reward: float = 0.0
) -> EpisodeResult:
    return EpisodeResult(
        outcome=outcome,
        success=outcome == "success",
        steps=10,
        total_reward=reward,
        final_longitudinal_error=0.0,
        final_lateral_error=lateral,
        final_yaw_error=0.0,
        max_abs_articulation=0.0,
    )


def test_straight_reverse_policy_is_a_zero_action() -> None:
    action = straight_reverse_policy(np.ones(5, dtype=np.float32))  # any size
    assert action.shape == (1,)
    assert action.dtype == np.float32
    assert float(action[0]) == 0.0


def test_straight_reverse_docks_from_an_aligned_start(
    aligned_scenario: Scenario,
) -> None:
    states: list[VehicleState] = []
    results = run_docking_test(
        aligned_scenario, straight_reverse_policy, on_state=states.append
    )
    assert len(results) == 1
    result = results[0]
    assert result.outcome == "success"
    assert result.success is True
    assert 300 < result.steps < 900
    # One state per reset plus one per step.
    assert len(states) == result.steps + 1
    assert abs(result.final_lateral_error) < 0.05
    assert abs(result.final_yaw_error) < 0.05
    assert result.max_abs_articulation < 0.05
    assert result.total_reward > 0.0
    assert result.to_dict()["outcome"] == "success"
    # Maneuver metrics from the proposal's evaluation list.
    assert result.maneuver_time == pytest.approx(0.1 * result.steps)
    assert result.path_length == pytest.approx(0.02 * result.steps, rel=0.02)
    assert result.steering_rate_rms == pytest.approx(0.0, abs=1e-9)
    assert result.mean_policy_time_ms >= 0.0
    assert result.controller == "ppo-policy"
    assert summarize(results) == {
        "episodes": 1,
        "success_rate": 1.0,
        "collision_rate": 0.0,
        "jackknife_rate": 0.0,
        "mean_total_reward": result.total_reward,
        "mean_abs_lateral_error": abs(result.final_lateral_error),
        "mean_abs_longitudinal_error": abs(result.final_longitudinal_error),
        "mean_abs_yaw_error": abs(result.final_yaw_error),
        "max_abs_articulation": result.max_abs_articulation,
        "mean_maneuver_time": result.maneuver_time,
        "mean_path_length": result.path_length,
        "mean_steering_rate_rms": result.steering_rate_rms,
        "mean_policy_time_ms": result.mean_policy_time_ms,
        "solve_failures": 0,
        "outcome_success": 1,
    }


def test_time_limit_and_seeded_episodes(aligned_scenario: Scenario) -> None:
    first = run_docking_test(
        aligned_scenario, straight_reverse_policy, episodes=2, max_episode_steps=15
    )
    second = run_docking_test(
        aligned_scenario, straight_reverse_policy, episodes=2, max_episode_steps=15
    )
    assert [r.outcome for r in first] == ["time_limit", "time_limit"]
    assert [r.steps for r in first] == [15, 15]
    assert not any(r.success for r in first)
    assert first == second


def test_should_stop_returns_only_completed_episodes(
    aligned_scenario: Scenario,
) -> None:
    calls = 0

    def should_stop() -> bool:
        nonlocal calls
        calls += 1
        # Episode 0 needs 1 + 10 checks; stop in the middle of episode 1.
        return calls > 16

    results = run_docking_test(
        aligned_scenario,
        straight_reverse_policy,
        episodes=5,
        max_episode_steps=10,
        should_stop=should_stop,
    )
    assert len(results) == 1
    assert results[0].steps == 10

    assert (
        run_docking_test(
            aligned_scenario, straight_reverse_policy, should_stop=lambda: True
        )
        == []
    )


def test_run_docking_test_rejects_zero_episodes(aligned_scenario: Scenario) -> None:
    with pytest.raises(ValueError, match="episodes"):
        run_docking_test(aligned_scenario, straight_reverse_policy, episodes=0)


def test_summarize() -> None:
    empty = summarize([])
    assert empty["episodes"] == 0
    assert all(value == 0 for value in empty.values())
    assert not any(key.startswith("outcome_") for key in empty)
    summary = summarize(
        [
            _result("success", lateral=0.1, reward=12.0),
            _result("jackknife", lateral=-0.5, reward=-9.0),
            _result("time_limit", lateral=0.3, reward=0.0),
            _result("success", lateral=-0.1, reward=13.0),
        ]
    )
    assert summary["episodes"] == 4
    assert summary["success_rate"] == 0.5
    assert summary["mean_total_reward"] == pytest.approx(4.0)
    assert summary["mean_abs_lateral_error"] == pytest.approx(0.25)
    assert summary["jackknife_rate"] == 0.25
    assert summary["collision_rate"] == 0.0
    assert summary["outcome_success"] == 2
    assert summary["outcome_jackknife"] == 1
    assert summary["outcome_time_limit"] == 1
    assert "outcome_collision" not in summary


def test_load_policy_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="policy file not found"):
        load_policy(tmp_path / "nope.zip")


def test_resolve_policy_path_accepts_omitted_zip_suffix(tmp_path) -> None:
    archive = tmp_path / "policy.zip"
    archive.write_bytes(b"test")

    assert resolve_policy_path(tmp_path / "policy") == archive


def test_cli_test_subcommand(aligned_scenario: Scenario, tmp_path, capsys) -> None:
    path = aligned_scenario.save(tmp_path / "aligned.yaml")
    assert cli.main(["test", "--scenario", str(path), "--episodes", "1"]) == 0
    output = capsys.readouterr().out
    assert "episode 0: success" in output
    assert "success_rate: 1.000" in output
    assert "outcome_success: 1" in output
    assert cli.main(
        ["test", "--scenario", str(path), "--lqr", "--episodes", "1"]
    ) == 0
    assert "episode 0: success" in capsys.readouterr().out

    assert cli.main(
        [
            "compare",
            "--scenario",
            str(path),
            "--controllers",
            "lqr",
            "--episodes",
            "1",
            "--seed",
            "7",
        ]
    ) == 0
    comparison = capsys.readouterr().out
    assert "[lqr | 2d]" in comparison
    assert "episodes: " in comparison
    assert "episode 0: success" in comparison


def test_cli_exposes_ppo_hyperparameters() -> None:
    arguments = cli._parser().parse_args(
        [
            "train",
            "--learning-rate",
            "0.001",
            "--n-steps",
            "128",
            "--batch-size",
            "64",
            "--n-epochs",
            "4",
            "--gamma",
            "0.98",
            "--gae-lambda",
            "0.9",
            "--clip-range",
            "0.15",
            "--entropy-coefficient",
            "0.02",
            "--target-kl",
            "0.03",
            "--device",
            "cpu",
        ]
    )
    config = cli._training_config(arguments)
    assert config.learning_rate == 0.001
    assert config.n_steps == 128
    assert config.batch_size == 64
    assert config.n_epochs == 4
    assert config.gamma == 0.98
    assert config.gae_lambda == 0.9
    assert config.clip_range == 0.15
    assert config.entropy_coefficient == 0.02
    assert config.target_kl == 0.03


def test_cli_can_save_a_formal_paired_comparison(
    aligned_scenario: Scenario, tmp_path, capsys
) -> None:
    scenario_path = aligned_scenario.save(tmp_path / "scenario.yaml")
    suite_path = ExperimentSuite(
        "cli-unit-suite-v1",
        (0,),
        EvaluationSplit(20, 1),
        EvaluationSplit(30, 1),
    ).save(tmp_path / "suite.yaml")
    code = cli.main(
        [
            "compare",
            "--scenario",
            str(scenario_path),
            "--controllers",
            "lqr",
            "--suite",
            str(suite_path),
            "--split",
            "validation",
            "--save-results",
            "--purpose",
            "cli-unit",
            "--output-root",
            str(tmp_path),
        ]
    )
    assert code == 0
    output = capsys.readouterr().out
    assert "[lqr | 2d]" in output
    assert "evidence:" in output
    configurations = list((tmp_path / "configs" / "experiments").glob("*.yaml"))
    results = [path for path in (tmp_path / "results").iterdir() if path.is_dir()]
    assert len(configurations) == 1
    assert len(results) == 1
    assert (results[0] / "manifest.yaml").is_file()


def test_cli_errors_return_2(aligned_scenario: Scenario, tmp_path, capsys) -> None:
    assert cli.main(["test", "--scenario", str(tmp_path / "missing.yaml")]) == 2
    error = capsys.readouterr().err
    assert error.startswith("error: ")
    assert error.count("\n") == 1

    path = aligned_scenario.save(tmp_path / "aligned.yaml")
    code = cli.main(
        ["test", "--scenario", str(path), "--policy", str(tmp_path / "none.zip")]
    )
    assert code == 2
    assert capsys.readouterr().err.startswith("error: policy file not found")

    assert cli.main(["train", "--scenario", str(tmp_path / "missing.yaml")]) == 2
    assert capsys.readouterr().err.startswith("error: ")
    assert cli.main(["train", "--scenario", str(path), "--timesteps", "0"]) == 2
    assert "total_timesteps" in capsys.readouterr().err
