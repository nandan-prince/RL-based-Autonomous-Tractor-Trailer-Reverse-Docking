"""Training campaigns, the method switch, and portable policy bundles."""

from __future__ import annotations

import json
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml

from trailer_docking.evaluation.scenarios import Scenario, ScenarioOverrides
from trailer_docking.training import cli
from trailer_docking.training.bundle import (
    BUNDLE_FORMAT,
    BundleError,
    export_bundle,
    load_bundle,
    run_bundle,
)
from trailer_docking.training.campaign import TrainingCampaign, run_ranking
from trailer_docking.training.config import TrainingConfig
from trailer_docking.training.evaluate import (
    EpisodeResult,
    run_docking_test,
)
from trailer_docking.training.methods import (
    MethodUnavailableError,
    nmpc_controller_factory,
    validate_method,
)
from trailer_docking.training.session import TrainingProgress, new_experiment_id

TINY = {"n_steps": 64, "batch_size": 32, "max_episode_steps": 40}


# ----------------------------------------------------------------- config


def test_campaign_settings_are_validated() -> None:
    config = TrainingConfig(
        runs=3,
        backend="3d",
        evaluation_backends=["2d", "3d"],
        method="nmpc",  # type: ignore[arg-type]
    )
    assert config.evaluation_backends == ("2d", "3d")
    assert TrainingConfig.from_dict(config.to_dict()) == config
    for changes in (
        {"backend": "mujoco"},
        {"evaluation_backends": ("2d", "2d")},
        {"evaluation_backends": ("4d",)},
        {"evaluation_backends": "2d"},
        {"evaluation_episodes": -1},
        {"evaluation_seed": -1},
        {"method": "lqr"},
        {"runs": True},
    ):
        with pytest.raises(ValueError):
            TrainingConfig(**changes)  # type: ignore[arg-type]
    assert TrainingConfig(evaluation_episodes=0).evaluation_episodes == 0
    with pytest.raises(ValueError):
        TrainingConfig.from_dict({"unknown": 1})
    with pytest.raises(ValueError):
        TrainingConfig.from_dict([])
    assert validate_method("ppo") == "ppo"


# ------------------------------------------------- campaign logic with fakes


class FakeSession:
    """A finished-instantly stand-in for ``TrainingSession``."""

    def __init__(
        self,
        scenario: Scenario,
        config: TrainingConfig,
        root: Path,
        fail: bool,
        checkpoint: bytes,
    ) -> None:
        self.config = config
        self._fail = fail
        self._content = checkpoint
        index = config.seed
        self._checkpoint = root / f"policy_seed{index}.zip"
        self._status = "idle"
        self.episode_history = (SimpleNamespace(index=0),)
        self.spectator_state = None
        self.stopped = False

    def start(self) -> None:
        if not self._fail:
            self._checkpoint.write_bytes(self._content)
        self._status = "failed" if self._fail else "finished"

    def stop(self) -> None:
        self.stopped = True

    def join(self, timeout: float | None = None) -> None:
        return None

    @property
    def progress(self) -> TrainingProgress:
        return TrainingProgress(
            status=self._status,
            timesteps=100,
            total_timesteps=self.config.total_timesteps,
            episodes=1,
            recent_success_rate=0.0,
            recent_mean_return=1.0,
            last_outcome="collision",
            error="boom" if self._fail else None,
            experiment_id=f"exp-seed{self.config.seed}",
            checkpoint_path=None if self._fail else self._checkpoint,
        )


def _result(success: bool, reward: float) -> EpisodeResult:
    return EpisodeResult(
        outcome="success" if success else "collision",
        success=success,
        steps=10,
        total_reward=reward,
        final_longitudinal_error=0.0,
        final_lateral_error=0.0,
        final_yaw_error=0.0,
        max_abs_articulation=0.0,
    )


def _campaign(
    tmp_path: Path,
    config: TrainingConfig,
    *,
    successes: dict[str, int],
    checkpoint: bytes = b"",
    fail_seed: int | None = None,
    **options: Any,
) -> tuple[TrainingCampaign, list[dict[str, Any]], list[FakeSession]]:
    calls: list[dict[str, Any]] = []
    sessions: list[FakeSession] = []

    def session_factory(scenario: Scenario, run: TrainingConfig, **kwargs: Any) -> Any:
        session = FakeSession(
            scenario, run, tmp_path, run.seed == fail_seed, checkpoint
        )
        session.kwargs = kwargs  # type: ignore[attr-defined]
        sessions.append(session)
        return session

    def policy_loader(path: Path) -> str:
        return path.name

    def test_runner(scenario: Scenario, policy: Any, **kwargs: Any) -> Any:
        calls.append({"policy": policy, **kwargs})
        key = policy if isinstance(policy, str) else "controller"
        good = successes.get(key, 0)
        return [_result(i < good, float(good)) for i in range(kwargs["episodes"])]

    campaign = TrainingCampaign(
        Scenario(),
        config,
        output_root=tmp_path,
        session_factory=session_factory,
        test_runner=test_runner,
        policy_loader=policy_loader,
        **options,
    )
    return campaign, calls, sessions


def test_campaign_trains_every_seed_tests_them_alike_and_exports_the_best(
    tmp_path: Path, policy_checkpoint: bytes
) -> None:
    config = TrainingConfig(
        runs=3,
        seed=5,
        total_timesteps=1000,
        evaluation_episodes=4,
        evaluation_backends=("2d", "3d"),
        purpose="unit",
    )
    campaign, calls, sessions = _campaign(
        tmp_path,
        config,
        successes={"policy_seed5.zip": 1, "policy_seed6.zip": 3, "policy_seed7.zip": 2},
        checkpoint=policy_checkpoint,
    )
    assert campaign.progress.status == "idle"
    campaign.start()
    with pytest.raises(RuntimeError, match="already started"):
        campaign.start()
    campaign.join(30)

    progress = campaign.progress
    assert progress.status == "finished" and progress.error is None
    assert [s.config.seed for s in sessions] == [5, 6, 7]
    assert all(s.config.runs == 1 for s in sessions)
    assert sessions[1].config.purpose == "unit run 2 of 3"
    assert all("live_3d" not in s.kwargs for s in sessions)  # type: ignore[attr-defined]
    assert progress.runs == 3 and progress.run == 3
    assert progress.total_timesteps == 3000 and progress.timesteps == 300
    assert progress.episodes == 3 == len(campaign.episode_history)
    assert progress.experiment_id is not None
    assert progress.experiment_id.endswith("_ppo_unit-campaign")

    # Every run is tested on both simulators with the same held-out seed.
    assert [(c["policy"], c["backend"]) for c in calls] == [
        ("policy_seed5.zip", "2d"),
        ("policy_seed5.zip", "3d"),
        ("policy_seed6.zip", "2d"),
        ("policy_seed6.zip", "3d"),
        ("policy_seed7.zip", "2d"),
        ("policy_seed7.zip", "3d"),
    ]
    assert {c["seed"] for c in calls} == {config.evaluation_seed}
    assert {c["episodes"] for c in calls} == {4}
    assert all("on_frame" not in c for c in calls)

    runs = campaign.run_summaries
    assert [run["evaluation"][0]["success_rate"] for run in runs] == [0.25, 0.75, 0.5]
    assert all(run["evaluation"][1]["backend"] == "3d" for run in runs)
    assert all(run["evaluation"][0]["complete"] for run in runs)

    # The best run by first-backend success rate is the exported result.
    assert progress.checkpoint_path == tmp_path / "policy_seed6.zip"
    assert progress.bundle_path is not None and progress.bundle_path.is_file()
    bundle = load_bundle(progress.bundle_path)
    assert bundle.metadata["source"]["experiment_id"] == "exp-seed6"
    assert bundle.metadata["source"]["campaign_id"] == progress.experiment_id
    assert bundle.training.seed == 6
    assert bundle.metadata["evaluation"][0]["success_rate"] == 0.75
    assert bundle.policy_bytes == policy_checkpoint
    assert bundle.metadata["source"]["status"] == "finished"
    assert bundle.metadata["source"]["timesteps"] == 100
    assert bundle.observation_size == 21

    results = tmp_path / "results" / progress.experiment_id
    manifest = yaml.safe_load((results / "manifest.yaml").read_text("utf-8"))
    assert manifest["kind"] == "campaign" and manifest["status"] == "finished"
    assert manifest["method"] == "ppo"
    assert manifest["run_experiment_ids"] == ["exp-seed5", "exp-seed6", "exp-seed7"]
    assert manifest["best_run_experiment_id"] == "exp-seed6"
    assert manifest["config"] == config.to_dict()
    assert manifest["bundle_path"].endswith("policy_bundle.dockpolicy.zip")
    assert manifest["mujoco_tuning"]["steering_kp"] > 0  # tests ran in 3D
    assert manifest["bundle_sha256"] and manifest["runtime"]["python"]
    assert all(s.kwargs["campaign_id"] == progress.experiment_id for s in sessions)  # type: ignore[attr-defined]
    summary = json.loads((results / "summary.json").read_text("utf-8"))
    assert summary["runs_finished"] == 3 and summary["best_run"]["seed"] == 6
    assert len(summary["runs"]) == 3
    # Checkpoint paths are recorded relative to the output root.
    assert summary["best_run"]["checkpoint_path"] == "policy_seed6.zip"
    # Every test episode is kept, so spread can be computed without a rerun.
    assert manifest["evaluation_files"] == sorted(
        f"evaluation_run{run}_{backend}.csv"
        for run in (1, 2, 3)
        for backend in ("2d", "3d")
    )
    rows = (results / "evaluation_run2_2d.csv").read_text("utf-8").splitlines()
    assert rows[0].startswith("episode,seed,outcome,success,steps")
    assert len(rows) == 5 and rows[1].startswith(f"0,{config.evaluation_seed},success")


def test_ranking_prefers_success_then_reward_and_untested_runs_last() -> None:
    def run(rate: float, reward: float, complete: bool = True) -> dict[str, Any]:
        return {
            "evaluation": [
                {
                    "success_rate": rate,
                    "mean_total_reward": reward,
                    "complete": complete,
                }
            ]
        }

    assert run_ranking(run(0.5, 1.0)) > run_ranking(run(0.4, 99.0))
    assert run_ranking(run(0.5, 2.0)) > run_ranking(run(0.5, 1.0))
    assert run_ranking(run(0.0, -500.0)) > run_ranking({"evaluation": []})
    # A test cut short by Stop cannot outrank a complete one, however lucky.
    assert run_ranking(run(0.1, -9.0)) > run_ranking(run(1.0, 50.0, complete=False))
    # A non-finite reward loses a tie instead of making the choice arbitrary.
    assert run_ranking(run(0.0, -5.0)) > run_ranking(run(0.0, float("nan")))


def test_campaign_without_tests_exports_the_first_run(
    tmp_path: Path, policy_checkpoint: bytes
) -> None:
    config = TrainingConfig(runs=2, evaluation_episodes=0)
    campaign, calls, _ = _campaign(
        tmp_path, config, successes={}, checkpoint=policy_checkpoint
    )
    campaign.start()
    campaign.join(30)
    assert calls == []
    assert campaign.progress.status == "finished"
    assert campaign.progress.bundle_path is not None
    assert all(run["evaluation"] == [] for run in campaign.run_summaries)
    assert campaign.progress.checkpoint_path == tmp_path / "policy_seed0.zip"


def test_a_failed_run_is_recorded_and_the_other_runs_still_produce_a_result(
    tmp_path: Path, policy_checkpoint: bytes
) -> None:
    config = TrainingConfig(runs=3, seed=0, evaluation_episodes=1)
    campaign, _, sessions = _campaign(
        tmp_path,
        config,
        successes={"policy_seed2.zip": 1},
        fail_seed=1,
        checkpoint=policy_checkpoint,
    )
    campaign.start()
    campaign.join(30)
    progress = campaign.progress
    assert progress.status == "failed"
    assert progress.error == "run 2 failed: boom"
    assert len(sessions) == 3  # one bad seed does not stop the others
    results = tmp_path / "results" / str(progress.experiment_id)
    summary = json.loads((results / "summary.json").read_text("utf-8"))
    assert [run["status"] for run in summary["runs"]] == [
        "finished",
        "failed",
        "finished",
    ]
    assert summary["runs"][1]["error"] == "boom"
    assert summary["status"] == "failed" and summary["runs_finished"] == 2
    # The usable runs are still ranked and exported.
    assert summary["best_run"]["run"] == 3
    assert progress.bundle_path is not None and progress.bundle_path.is_file()


def test_a_failing_test_keeps_the_trained_run_on_record(
    tmp_path: Path, policy_checkpoint: bytes
) -> None:
    def exploding(scenario: Scenario, policy: Any, **options: Any) -> Any:
        raise RuntimeError("mujoco exploded")

    campaign, _, _ = _campaign(
        tmp_path,
        TrainingConfig(runs=1, evaluation_episodes=1),
        successes={},
        checkpoint=policy_checkpoint,
    )
    campaign._test_runner = exploding
    campaign.start()
    campaign.join(30)
    progress = campaign.progress
    assert progress.status == "failed"
    assert progress.error == "testing run 1 failed: mujoco exploded"
    (run,) = campaign.run_summaries
    assert run["experiment_id"] == "exp-seed0"
    assert run["evaluation_error"] == "mujoco exploded"
    assert progress.bundle_path is not None  # the policy itself is still usable


def test_a_test_cut_short_by_stop_is_not_chosen_over_a_complete_one(
    tmp_path: Path, policy_checkpoint: bytes
) -> None:
    config = TrainingConfig(runs=2, evaluation_episodes=4)
    campaign, _, _ = _campaign(
        tmp_path, config, successes={}, checkpoint=policy_checkpoint
    )

    def runner(scenario: Scenario, policy: Any, **options: Any) -> Any:
        if policy == "policy_seed0.zip":
            return [_result(i < 1, 1.0) for i in range(4)]  # 25 %, complete
        campaign.stop()
        return [_result(True, 9.0)]  # one lucky episode, then stopped

    campaign._test_runner = runner
    campaign.start()
    campaign.join(30)
    assert campaign.progress.status == "stopped"
    runs = campaign.run_summaries
    assert [run["evaluation"][0]["complete"] for run in runs] == [True, False]
    assert campaign.progress.checkpoint_path == tmp_path / "policy_seed0.zip"


def test_the_campaign_record_exists_while_it_is_still_running(
    tmp_path: Path, policy_checkpoint: bytes
) -> None:
    seen: list[dict[str, Any]] = []
    campaign, _, _ = _campaign(
        tmp_path,
        TrainingConfig(runs=1, evaluation_episodes=1),
        successes={},
        checkpoint=policy_checkpoint,
    )

    def runner(scenario: Scenario, policy: Any, **options: Any) -> Any:
        results = tmp_path / "results" / str(campaign.progress.experiment_id)
        seen.append(yaml.safe_load((results / "manifest.yaml").read_text("utf-8")))
        return [_result(False, 0.0)]

    campaign._test_runner = runner
    campaign.start()
    campaign.join(30)
    assert seen[0]["status"] == "running" and seen[0]["finished"] is None
    assert campaign.progress.status == "finished"


def test_a_worker_that_dies_never_leaves_the_status_running(tmp_path: Path) -> None:
    pytest.importorskip("stable_baselines3")
    from trailer_docking.training.session import TrainingSession

    def dying(scenario: Scenario, **options: Any) -> Any:
        raise SystemExit(3)

    session = TrainingSession(
        Scenario(),
        TrainingConfig(total_timesteps=64, **TINY),
        output_root=tmp_path,
        environment_factory=dying,
    )
    session.start()
    session.join(30)
    assert session.progress.status == "failed"


def test_campaign_stopped_before_start_trains_nothing(tmp_path: Path) -> None:
    campaign, calls, sessions = _campaign(
        tmp_path, TrainingConfig(runs=2), successes={}
    )
    assert new_experiment_id("x", method="nmpc").endswith("_nmpc_x")
    campaign.stop()
    campaign.start()
    campaign.join(30)
    assert campaign.progress.status == "stopped"
    assert sessions == [] and calls == []
    assert campaign.progress.bundle_path is None
    campaign.stop()  # safe after the end
    assert campaign.progress.status == "stopped"


def test_campaign_passes_live_3d_only_when_asked(
    tmp_path: Path, policy_checkpoint: bytes
) -> None:
    config = TrainingConfig(
        runs=1, evaluation_episodes=1, evaluation_backends=("2d", "3d")
    )
    campaign, calls, sessions = _campaign(
        tmp_path, config, successes={}, live_3d=True, checkpoint=policy_checkpoint
    )
    campaign.start()
    campaign.join(30)
    assert sessions[0].kwargs["live_3d"] is True  # type: ignore[attr-defined]
    assert ["on_frame" in call for call in calls] == [False, True]
    frame = np.zeros((2, 2, 3), np.uint8)
    calls[1]["on_frame"](frame)
    assert campaign.spectator_frame is frame


# ------------------------------------------------------------ method switch


def test_nmpc_is_reported_as_not_implemented_until_the_hook_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(MethodUnavailableError, match="not implemented yet"):
        nmpc_controller_factory()

    import trailer_docking.controllers.nmpc as module

    def make_docking_controller(env: Any) -> Any:
        return lambda observation: np.zeros(1, np.float32)

    monkeypatch.setattr(
        module, "make_docking_controller", make_docking_controller, raising=False
    )
    assert nmpc_controller_factory() is make_docking_controller


def test_a_controller_factory_is_rebuilt_from_the_environment_each_episode(
    aligned_scenario: Scenario,
) -> None:
    seen: list[Any] = []

    def factory(env: Any) -> Any:
        # A model-based controller reads the shared state and reference path.
        seen.append((env.state.simulation_time, env.reference_path.length))
        return lambda observation: [0.0]  # any shape-(1,) array-like

    results = run_docking_test(
        aligned_scenario, None, episodes=2, controller_factory=factory
    )
    assert [r.outcome for r in results] == ["success", "success"]
    assert len(seen) == 2 and all(time == 0.0 for time, _ in seen)
    with pytest.raises(ValueError, match="either a policy or"):
        run_docking_test(aligned_scenario, None)
    with pytest.raises(ValueError, match="either a policy or"):
        run_docking_test(aligned_scenario, lambda o: o, controller_factory=factory)


def test_nmpc_campaign_skips_training_and_only_runs_the_controller(
    tmp_path: Path,
) -> None:
    config = TrainingConfig(
        method="nmpc", runs=4, evaluation_episodes=3, evaluation_backends=("2d", "3d")
    )
    marker = object()
    campaign, calls, sessions = _campaign(
        tmp_path,
        config,
        successes={"controller": 2},
        controller_loader=lambda: marker,
    )
    campaign.start()
    campaign.join(30)
    progress = campaign.progress
    assert progress.status == "finished"
    assert sessions == []  # nothing is trained
    assert progress.total_timesteps == 0 and progress.runs == 1
    assert "_nmpc_" in str(progress.experiment_id)
    assert [c["backend"] for c in calls] == ["2d", "3d"]
    assert all(c["controller_factory"] is marker and c["policy"] is None for c in calls)
    assert progress.bundle_path is None and progress.checkpoint_path is None
    (run,) = campaign.run_summaries
    assert run["evaluation"][0]["success_rate"] == pytest.approx(2 / 3)
    results = tmp_path / "results" / str(progress.experiment_id)
    manifest = yaml.safe_load((results / "manifest.yaml").read_text("utf-8"))
    assert manifest["method"] == "nmpc"
    summary = json.loads((results / "summary.json").read_text("utf-8"))
    assert summary["best_run"]["evaluation"][1]["backend"] == "3d"


def test_nmpc_campaign_fails_cleanly_while_the_controller_is_missing(
    tmp_path: Path,
) -> None:
    campaign = TrainingCampaign(
        Scenario(), TrainingConfig(method="nmpc"), output_root=tmp_path
    )
    campaign.start()
    campaign.join(30)
    progress = campaign.progress
    assert progress.status == "failed"
    assert "NMPC is not implemented yet" in str(progress.error)
    results = tmp_path / "results" / str(progress.experiment_id)
    assert (
        yaml.safe_load((results / "manifest.yaml").read_text("utf-8"))["status"]
        == "failed"
    )


# ----------------------------------------------------- real run and bundles


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    """One real two-run PPO campaign on the 2D simulator."""

    pytest.importorskip("stable_baselines3")
    root = tmp_path_factory.mktemp("campaign")
    scenario = Scenario(
        name="aligned",
        overrides=ScenarioOverrides(trailer_hitch_to_axle=3.5),
    )
    config = TrainingConfig(
        runs=2,
        total_timesteps=128,
        evaluation_episodes=2,
        evaluation_backends=("2d", "3d"),
        purpose="real",
        **TINY,
    )
    campaign = TrainingCampaign(scenario, config, output_root=root)
    campaign.start()
    campaign.join(300)
    return SimpleNamespace(
        root=root, scenario=scenario, config=config, campaign=campaign
    )


def test_real_campaign_produces_a_finished_result(trained: SimpleNamespace) -> None:
    progress = trained.campaign.progress
    assert progress.status == "finished", progress.error
    assert progress.timesteps == 2 * 128
    runs = trained.campaign.run_summaries
    assert [run["seed"] for run in runs] == [0, 1]
    assert len({run["experiment_id"] for run in runs}) == 2
    for run in runs:
        assert [item["backend"] for item in run["evaluation"]] == ["2d", "3d"]
        assert all(item["episodes"] == 2 for item in run["evaluation"])
        # Each run keeps its own experiment package as well.
        manifest = trained.root / "results" / run["experiment_id"] / "manifest.yaml"
        assert yaml.safe_load(manifest.read_text("utf-8"))["status"] == "finished"
    assert progress.bundle_path is not None
    assert progress.bundle_path.parent.name == progress.experiment_id


def test_bundle_round_trip_runs_in_a_fresh_simulator(
    trained: SimpleNamespace, tmp_path: Path
) -> None:
    path = trained.campaign.progress.bundle_path
    bundle = load_bundle(path)
    assert bundle.metadata["format"] == BUNDLE_FORMAT
    assert bundle.scenario == trained.scenario
    assert bundle.fingerprint_matches
    assert bundle.observation_size == 21
    assert bundle.training.max_episode_steps == 40
    assert "trained in 2d" in bundle.description()
    assert set(bundle.metadata["provenance"]) == {
        "versions",
        "git",
        "mujoco_tuning",
    }
    assert bundle.training is not None

    policy = bundle.load_policy()
    action = policy(np.zeros(21, np.float32))
    assert action.shape == (1,) and -1.0 <= float(action[0]) <= 1.0

    # The same bundle replays identically in two fresh simulators, and runs
    # in 3D although it was trained in 2D.
    first = run_bundle(path, episodes=2, seed=3)
    second = run_bundle(bundle, episodes=2, seed=3)
    assert first == second and len(first) == 2
    assert all(result.steps <= 40 for result in first)
    in_3d = run_bundle(bundle, backend="3d", episodes=1, seed=3)
    assert len(in_3d) == 1 and in_3d[0].steps <= 40
    with pytest.raises(ValueError):
        run_bundle(bundle, backend="4d")

    # A different truck can be tried with the same policy.
    other = replace(
        trained.scenario, overrides=ScenarioOverrides(trailer_variant="short")
    )
    assert len(run_bundle(bundle, scenario=other, episodes=1)) == 1

    # The command line runs the bundle too.
    assert cli.main(["test", "--bundle", str(path), "--backend", "3d"]) == 0
    copy = tmp_path / "copy.dockpolicy.zip"
    copy.write_bytes(Path(path).read_bytes())
    assert load_bundle(copy).policy_bytes == bundle.policy_bytes


def test_export_refuses_to_overwrite_and_checks_inputs(
    trained: SimpleNamespace, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    checkpoint = trained.campaign.progress.checkpoint_path
    target = export_bundle(checkpoint, trained.scenario, tmp_path)
    assert target.name == "aligned.dockpolicy.zip"
    # Unknown training settings are recorded as unknown, never as defaults,
    # and the replay then uses the task's own step limit.
    unknown = load_bundle(target)
    assert unknown.training is None and unknown.max_episode_steps is None
    assert "training settings unknown" in unknown.description()
    assert unknown.metadata["training"] is None
    replay = run_bundle(unknown, episodes=1)
    assert replay[0].steps <= 600
    with pytest.raises(FileExistsError):
        export_bundle(checkpoint, trained.scenario, target)
    with pytest.raises(FileNotFoundError):
        export_bundle(tmp_path / "missing.zip", trained.scenario, tmp_path / "x.zip")

    out = tmp_path / "cli.dockpolicy.zip"
    run_id = trained.campaign.run_summaries[0]["experiment_id"]
    manifest = trained.root / "results" / run_id / "manifest.yaml"
    arguments = ["export", "--policy", str(checkpoint), "--output", str(out)]
    assert cli.main([*arguments, "--manifest", str(manifest)]) == 0
    exported = load_bundle(out)
    # With the run's manifest the real settings and scenario are recorded.
    assert exported.training is not None
    assert exported.training.max_episode_steps == 40
    assert exported.scenario == trained.scenario
    assert exported.metadata["source"]["experiment_id"] == run_id
    assert (
        cli.main([*arguments[:-1], str(tmp_path / "n.zip"), "--manifest", str(out)])
        == 2
    )
    assert cli.main(["export", "--policy", str(checkpoint), "--output", str(out)]) == 2
    assert "refusing to overwrite" in capsys.readouterr().err


def test_loading_a_policy_never_unpickles_its_metadata(
    policy_checkpoint: bytes, tmp_path: Path
) -> None:
    """A policy file from someone else must not be able to run code.

    Stable-Baselines3 stores pickled objects in the archive's ``data`` member
    and ``PPO.load`` unpickles them.  The loader here reads only the weights.
    """

    import base64
    import io
    import pickle

    from trailer_docking.training.evaluate import (
        PolicyFormatError,
        load_policy,
        policy_from_bytes,
        policy_observation_size,
    )

    marker = tmp_path / "executed.txt"

    class Payload:
        def __reduce__(self) -> Any:
            return (Path.write_text, (marker, "code ran"))

    poisoned = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(policy_checkpoint)) as source:
        weights = source.read("policy.pth")
    serialized = base64.b64encode(pickle.dumps(Payload())).decode()
    with zipfile.ZipFile(poisoned, "w") as archive:
        archive.writestr("policy.pth", weights)
        archive.writestr(
            "data", json.dumps({"policy_class": {":serialized:": serialized}})
        )
    path = tmp_path / "poisoned.zip"
    path.write_bytes(poisoned.getvalue())

    policy = load_policy(path)
    assert policy(np.zeros(21, np.float32)).shape == (1,)
    bundle = export_bundle(path, Scenario(), tmp_path / "poisoned.dockpolicy.zip")
    assert len(run_bundle(bundle, episodes=1)) == 1
    assert not marker.exists()

    assert policy_observation_size(policy_checkpoint) == 21
    # Pickled weights are refused by the weights-only loader as well.
    evil = io.BytesIO()
    with zipfile.ZipFile(evil, "w") as archive:
        archive.writestr("policy.pth", pickle.dumps(Payload()))
    for content in (evil.getvalue(), b"not a zip", policy_checkpoint[:200]):
        with pytest.raises(PolicyFormatError):
            policy_from_bytes(content)
    assert not marker.exists()
    garbage = tmp_path / "garbage.zip"
    garbage.write_bytes(evil.getvalue())
    assert cli.main(["test", "--policy", str(garbage)]) == 2
    with pytest.raises(BundleError):
        export_bundle(garbage, Scenario(), tmp_path / "garbage.dockpolicy.zip")


def test_unusable_bundles_are_rejected(tmp_path: Path, capsys: Any) -> None:
    with pytest.raises(FileNotFoundError):
        load_bundle(tmp_path / "missing.zip")
    text = tmp_path / "text.zip"
    text.write_text("not a zip", encoding="utf-8")
    with pytest.raises(BundleError, match="not a policy bundle"):
        load_bundle(text)

    def write(name: str, metadata: Any, policy: bool = True) -> Path:
        path = tmp_path / name
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("bundle.yaml", yaml.safe_dump(metadata))
            if policy:
                archive.writestr("policy.zip", b"x")
        return path

    assert cli.main(["test", "--policy", str(text)]) == 2
    good = {
        "format": BUNDLE_FORMAT,
        "format_version": 1,
        "scenario": Scenario().to_dict(),
        "training": {},
        "observation_size": 21,
    }
    assert load_bundle(write("good.zip", good)).scenario == Scenario()
    for name, metadata, policy in (
        ("other.zip", {"format": "something else"}, True),
        ("list.zip", [1, 2], True),
        ("version.zip", {**good, "format_version": 99}, True),
        ("nopolicy.zip", good, False),
        ("scenario.zip", {**good, "scenario": {"start": {"standoff": [9, 1]}}}, True),
        ("training.zip", {**good, "training": {"runs": 0}}, True),
        ("training2.zip", {**good, "training": [1]}, True),
        ("size.zip", {k: v for k, v in good.items() if k != "observation_size"}, True),
    ):
        with pytest.raises(BundleError):
            load_bundle(write(name, metadata, policy))
    # Settings added by a later version are ignored rather than fatal, and
    # a malformed source does not break the description.
    newer = load_bundle(
        write("newer.zip", {**good, "training": {"gamma": 0.99}, "source": "x"})
    )
    assert newer.training == TrainingConfig() and "unknown run" in newer.description()
    # A bundle whose policy expects another observation layout cannot run.
    mismatched = load_bundle(write("obs.zip", {**good, "observation_size": 9}))
    with pytest.raises(BundleError, match="policy expects 9"):
        run_bundle(mismatched)
    assert cli.main(["test", "--bundle", str(text)]) == 2
    assert "error:" in capsys.readouterr().err


def test_cli_train_runs_a_campaign(tmp_path: Path, capsys: Any) -> None:
    pytest.importorskip("stable_baselines3")
    code = cli.main(
        [
            "train",
            "--timesteps",
            "64",
            "--runs",
            "2",
            "--eval-episodes",
            "1",
            "--eval-backends",
            "2d,3d",
            "--output-root",
            str(tmp_path),
        ]
    )
    output = capsys.readouterr().out
    assert code == 0, output
    assert "run 1 (seed 0): finished" in output
    assert "run 2 (seed 1): finished" in output
    assert "2d:" in output and "3d:" in output
    bundles = list(tmp_path.glob("artifacts/*/policy_bundle.dockpolicy.zip"))
    assert len(bundles) == 1
    assert "runs_finished" in next(
        tmp_path.glob("results/*campaign/summary.json")
    ).read_text("utf-8")
    assert cli.main(["train", "--runs", "0", "--output-root", str(tmp_path)]) == 2
    assert cli.main(["train", "--eval-backends", "", "--timesteps", "64"]) == 2
    assert cli.main(["train", "--method", "nmpc", "--eval-episodes", "0"]) == 2
    assert cli.main(["train", "--seed", "100000"]) == 2  # overlaps the test seeds
    assert cli.main(["train", "--eval-backends", "9d"]) == 2
    assert cli.main(["test", "--backend", "3d", "--episodes", "0"]) == 2
