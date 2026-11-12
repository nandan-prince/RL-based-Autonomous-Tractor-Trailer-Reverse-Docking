import csv
import time

import numpy as np
import pytest
import yaml

pytest.importorskip("gymnasium")
pytest.importorskip("mujoco")

from trailer_docking.core.state import VehicleState
from trailer_docking.environments.docking_env import DockingEnv
from trailer_docking.evaluation.scenarios import Scenario
from trailer_docking.training import (
    TrainingConfig,
    TrainingSession,
    cli,
    load_policy,
    run_docking_test,
)

TINY = {"n_steps": 64, "batch_size": 32, "max_episode_steps": 40}


def test_tiny_training_run_writes_the_experiment_package(
    aligned_scenario: Scenario, tmp_path
) -> None:
    pytest.importorskip("stable_baselines3")
    config = TrainingConfig(
        total_timesteps=256, checkpoint_interval=100, purpose="unit test", **TINY
    )
    session = TrainingSession(aligned_scenario, config, output_root=tmp_path)
    session.start()
    with pytest.raises(RuntimeError, match="already started"):
        session.start()
    session.join(120)

    progress = session.progress
    assert progress.error is None
    assert progress.status == "finished"
    assert progress.timesteps == 256
    assert progress.total_timesteps == 256
    assert progress.experiment_id is not None
    assert progress.experiment_id.endswith("_ppo_unit-test")
    assert isinstance(session.spectator_state, VehicleState)

    # 256 steps of 40-step episodes: six complete timeouts.
    history = session.episode_history
    assert progress.episodes == len(history) == 6
    assert [record.index for record in history] == list(range(6))
    assert [record.timesteps for record in history] == [40, 80, 120, 160, 200, 240]
    assert all(record.outcome == "timeout" and record.steps == 40 for record in history)
    assert progress.last_outcome == "timeout"
    assert progress.recent_success_rate == 0.0
    assert progress.recent_mean_return == pytest.approx(
        sum(record.total_reward for record in history) / 6
    )

    results = tmp_path / "results" / progress.experiment_id
    checkpoints = tmp_path / "artifacts" / progress.experiment_id / "checkpoints"
    final = checkpoints / "policy_final.zip"
    assert final.is_file()
    assert progress.checkpoint_path == final
    assert (checkpoints / "policy_100.zip").is_file()
    assert (checkpoints / "policy_200.zip").is_file()

    manifest = yaml.safe_load((results / "manifest.yaml").read_text(encoding="utf-8"))
    assert manifest["experiment_id"] == progress.experiment_id
    assert manifest["purpose"] == "unit test"
    assert manifest["status"] == "finished"
    assert manifest["started"] and manifest["finished"]
    assert manifest["scenario"] == aligned_scenario.to_dict()
    assert manifest["scenario_fingerprint"] == aligned_scenario.fingerprint
    assert manifest["config"] == config.to_dict()
    assert manifest["timesteps"] == 256
    assert manifest["episodes"] == 6
    assert set(manifest["versions"]) == {
        "stable_baselines3",
        "gymnasium",
        "torch",
        "mujoco",
    }
    assert all(manifest["versions"].values())
    assert manifest["checkpoint_path"] == (
        f"artifacts/{progress.experiment_id}/checkpoints/policy_final.zip"
    )
    assert "placeholder" in manifest["note"]

    with (results / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6
    assert list(rows[0]) == [
        "index",
        "timesteps",
        "outcome",
        "success",
        "steps",
        "total_reward",
    ]
    assert rows[-1]["timesteps"] == "240"
    assert float(rows[0]["total_reward"]) == pytest.approx(history[0].total_reward)

    policy = load_policy(final)
    tested = run_docking_test(aligned_scenario, policy, max_episode_steps=5)
    assert tested[0].steps == 5
    env = DockingEnv(aligned_scenario)
    action = policy(np.zeros(env.observation_space.shape, dtype=np.float32))
    env.close()
    assert action.shape == (1,)
    assert -1.0 <= float(action[0]) <= 1.0

    # The saved policy is usable from the CLI as well.
    path = aligned_scenario.save(tmp_path / "aligned.yaml")
    assert cli.main(["test", "--scenario", str(path), "--policy", str(final)]) == 0


def test_same_minute_runs_get_distinct_experiment_ids(
    aligned_scenario: Scenario, tmp_path, monkeypatch
) -> None:
    pytest.importorskip("stable_baselines3")
    from trailer_docking.training import session as session_module

    monkeypatch.setattr(
        session_module, "new_experiment_id", lambda purpose: "20260101-0000_ppo_x"
    )
    config = TrainingConfig(total_timesteps=64, **TINY)
    identifiers = []
    for _ in range(3):
        session = TrainingSession(aligned_scenario, config, output_root=tmp_path)
        session.start()
        session.join(120)
        assert session.progress.status == "finished"
        identifiers.append(session.progress.experiment_id)
    assert identifiers == [
        "20260101-0000_ppo_x",
        "20260101-0000_ppo_x-2",
        "20260101-0000_ppo_x-3",
    ]
    for identifier in identifiers:
        assert (tmp_path / "results" / identifier / "manifest.yaml").is_file()


def test_stop_ends_a_long_run_promptly(aligned_scenario: Scenario, tmp_path) -> None:
    pytest.importorskip("stable_baselines3")
    config = TrainingConfig(total_timesteps=1_000_000, **TINY)
    session = TrainingSession(aligned_scenario, config, output_root=tmp_path)
    session.start()
    deadline = time.monotonic() + 60
    while session.progress.timesteps < 70 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert session.progress.status == "running"

    requested = time.monotonic()
    session.stop()
    assert session.progress.status in ("stopping", "stopped")
    session.join(30)
    assert time.monotonic() - requested < 10

    progress = session.progress
    assert progress.status == "stopped"
    assert progress.error is None
    assert 70 <= progress.timesteps < 1_000_000
    # At least one PPO update ran, so the partial policy is kept.
    assert progress.checkpoint_path is not None
    assert progress.checkpoint_path.name == "policy_final.zip"
    assert progress.checkpoint_path.is_file()
    assert progress.experiment_id is not None
    results = tmp_path / "results" / progress.experiment_id
    manifest = yaml.safe_load((results / "manifest.yaml").read_text(encoding="utf-8"))
    assert manifest["status"] == "stopped"
    assert (results / "metrics.csv").is_file()
    session.stop()  # safe after the run ended
    assert session.progress.status == "stopped"


def test_failing_simulator_factory_reports_failure(
    aligned_scenario: Scenario, tmp_path
) -> None:
    pytest.importorskip("stable_baselines3")

    def broken_factory(vehicle, dock):
        raise RuntimeError("simulator exploded")

    session = TrainingSession(
        aligned_scenario,
        TrainingConfig(total_timesteps=64, **TINY),
        output_root=tmp_path,
        simulator_factory=broken_factory,
    )
    session.start()
    session.join(60)
    progress = session.progress
    assert progress.status == "failed"
    assert progress.error == "simulator exploded"
    assert progress.checkpoint_path is None
    assert session.spectator_state is None
    assert progress.experiment_id is not None

    # A failed run is preserved, with zero episodes, rather than discarded.
    results = tmp_path / "results" / progress.experiment_id
    manifest = yaml.safe_load((results / "manifest.yaml").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["error"] == "simulator exploded"
    assert manifest["episodes"] == 0
    assert (results / "metrics.csv").read_text(encoding="utf-8").splitlines() == [
        "index,timesteps,outcome,success,steps,total_reward"
    ]
