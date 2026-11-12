import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("mujoco")

from trailer_docking.core.state import VehicleState
from trailer_docking.evaluation.scenarios import Scenario
from trailer_docking.training import cli
from trailer_docking.training.evaluate import (
    EpisodeResult,
    load_policy,
    run_docking_test,
    straight_reverse_policy,
    summarize,
)


def _result(
    outcome: str, *, lateral: float = 0.0, reward: float = 0.0
) -> EpisodeResult:
    return EpisodeResult(
        outcome=outcome,
        success=outcome == "docked",
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
    assert result.outcome == "docked"
    assert result.success is True
    assert 300 < result.steps < 900
    # One state per reset plus one per step.
    assert len(states) == result.steps + 1
    assert abs(result.final_lateral_error) < 0.05
    assert abs(result.final_yaw_error) < 0.05
    assert result.max_abs_articulation < 0.05
    assert result.total_reward > 0.0
    assert result.to_dict()["outcome"] == "docked"
    assert summarize(results) == {
        "episodes": 1,
        "success_rate": 1.0,
        "mean_total_reward": result.total_reward,
        "mean_abs_lateral_error": abs(result.final_lateral_error),
        "outcome_docked": 1,
    }


def test_timeout_and_seeded_episodes(aligned_scenario: Scenario) -> None:
    first = run_docking_test(
        aligned_scenario, straight_reverse_policy, episodes=2, max_episode_steps=15
    )
    second = run_docking_test(
        aligned_scenario, straight_reverse_policy, episodes=2, max_episode_steps=15
    )
    assert [r.outcome for r in first] == ["timeout", "timeout"]
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
    assert summarize([]) == {
        "episodes": 0,
        "success_rate": 0.0,
        "mean_total_reward": 0.0,
        "mean_abs_lateral_error": 0.0,
    }
    summary = summarize(
        [
            _result("docked", lateral=0.1, reward=12.0),
            _result("jackknife", lateral=-0.5, reward=-9.0),
            _result("timeout", lateral=0.3, reward=0.0),
            _result("docked", lateral=-0.1, reward=13.0),
        ]
    )
    assert summary["episodes"] == 4
    assert summary["success_rate"] == 0.5
    assert summary["mean_total_reward"] == pytest.approx(4.0)
    assert summary["mean_abs_lateral_error"] == pytest.approx(0.25)
    assert summary["outcome_docked"] == 2
    assert summary["outcome_jackknife"] == 1
    assert summary["outcome_timeout"] == 1
    assert "outcome_collision" not in summary


def test_load_policy_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="policy file not found"):
        load_policy(tmp_path / "nope.zip")


def test_cli_test_subcommand(aligned_scenario: Scenario, tmp_path, capsys) -> None:
    path = aligned_scenario.save(tmp_path / "aligned.yaml")
    assert cli.main(["test", "--scenario", str(path), "--episodes", "1"]) == 0
    output = capsys.readouterr().out
    assert "episode 0: docked" in output
    assert "success_rate: 1.000" in output
    assert "outcome_docked: 1" in output


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
