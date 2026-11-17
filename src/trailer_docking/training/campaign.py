"""Training campaigns: repeat a training run, test each policy, keep the best.

A :class:`TrainingCampaign` is what the UI and CLI start.  It runs
``config.runs`` :class:`~trailer_docking.training.session.TrainingSession`
runs one after another with consecutive seeds, tests each final policy on the
requested simulator backends, selects the best run, and writes one portable
policy bundle.  A campaign of one run is simply a training run followed by its
test and export.

It exposes the same interface as a session (``start``, ``stop``, ``join``,
``progress``, ``spectator_state``, ``spectator_route``, ``episode_history``),
so a view can watch either.  Outputs::

    <output_root>/results/<campaign_id>/manifest.yaml
    <output_root>/results/<campaign_id>/summary.json
    <output_root>/results/<campaign_id>/evaluation_run<k>_<backend>.csv
    <output_root>/artifacts/<campaign_id>/policy_bundle.dockpolicy.zip

Each run additionally writes its own experiment package under its own ID.
"""

from __future__ import annotations

import csv
import json
import threading
from collections.abc import Callable
from dataclasses import replace
from itertools import count
from math import isfinite
from pathlib import Path
from typing import Any

import yaml

from trailer_docking.core.state import Point2D, VehicleState
from trailer_docking.evaluation.scenarios import Scenario
from trailer_docking.training.bundle import BUNDLE_SUFFIX, export_bundle
from trailer_docking.training.config import TrainingConfig
from trailer_docking.training.methods import (
    METHOD_NMPC,
    METHOD_PPO,
    nmpc_controller_factory,
)
from trailer_docking.training.session import (
    REWARD_NOTE,
    STATUS_FAILED,
    STATUS_FINISHED,
    STATUS_IDLE,
    STATUS_RUNNING,
    STATUS_STOPPED,
    STATUS_STOPPING,
    EpisodeRecord,
    TrainingProgress,
    TrainingSession,
    _git_revision,
    _mujoco_tuning,
    _package_versions,
    _timestamp,
    file_sha256,
    new_experiment_id,
    runtime_summary,
)

POLL_PERIOD = 0.1
BUNDLE_NAME = f"policy_bundle{BUNDLE_SUFFIX}"
_SESSION_ACTIVE = frozenset({STATUS_IDLE, STATUS_RUNNING, STATUS_STOPPING})

SessionFactory = Callable[..., Any]
TestRunner = Callable[..., Any]


def _default_session_factory(
    scenario: Scenario,
    config: TrainingConfig,
    *,
    output_root: Path,
    live_3d: bool = False,
    campaign_id: str | None = None,
) -> Any:
    return TrainingSession(
        scenario,
        config,
        output_root=output_root,
        live_3d=live_3d,
        campaign_id=campaign_id,
    )


def _default_test_runner(scenario: Scenario, policy: Any, **options: Any) -> Any:
    from trailer_docking.training.evaluate import run_docking_test

    return run_docking_test(scenario, policy, **options)


def _default_policy_loader(path: Path) -> Any:
    from trailer_docking.training.evaluate import load_policy

    return load_policy(path)


def _summarize(results: Any) -> dict[str, Any]:
    from trailer_docking.training.evaluate import summarize

    return dict(summarize(results))


def run_ranking(run: dict[str, Any]) -> tuple[float, float]:
    """Sort key for choosing the best run: success rate, then mean reward.

    The first evaluation backend decides, and only if every requested test
    episode ran.  When no run was fully tested all runs rank equally and the
    first one is chosen.  A run that was not tested, or whose test was cut short,
    ranks below every fully tested run, so a single lucky episode cannot
    outrank a complete test.
    """

    evaluations = run.get("evaluation") or []
    if not evaluations or not evaluations[0].get("complete", False):
        return (-1.0, float("-inf"))
    first = evaluations[0]
    reward = float(first["mean_total_reward"])
    if not isfinite(reward):
        reward = float("-inf")
    return (float(first["success_rate"]), reward)


class TrainingCampaign:
    """Run, test, and export ``config.runs`` training runs in the background."""

    def __init__(
        self,
        scenario: Scenario,
        config: TrainingConfig | None = None,
        *,
        output_root: str | Path = ".",
        session_factory: SessionFactory | None = None,
        test_runner: TestRunner | None = None,
        policy_loader: Callable[[Path], Any] | None = None,
        controller_loader: Callable[[], Any] | None = None,
        live_3d: bool = False,
    ) -> None:
        self.scenario = scenario
        self.config = config if config is not None else TrainingConfig()
        self.output_root = Path(output_root)
        self._session_factory = session_factory or _default_session_factory
        self._test_runner = test_runner or _default_test_runner
        self._policy_loader = policy_loader or _default_policy_loader
        self._controller_loader = controller_loader or nmpc_controller_factory
        self._lock = threading.Lock()
        self._stop_requested = threading.Event()
        self._thread: threading.Thread | None = None
        self._status = STATUS_IDLE
        self._error: str | None = None
        self._campaign_id: str | None = None
        self._results_dir: Path | None = None
        self._artifacts_dir: Path | None = None
        self._started_at: str | None = None
        self._session: Any = None
        self._run_index = 0
        self._phase = ""
        self._finished_timesteps = 0
        self._finished_history: list[EpisodeRecord] = []
        self._last_progress: TrainingProgress | None = None
        self._runs: list[dict[str, Any]] = []
        self._best_checkpoint: Path | None = None
        self._bundle_path: Path | None = None
        self._evaluation_state: VehicleState | None = None
        self._evaluating = False
        # Render 3D episodes so a view can show them; costs training speed.
        self._live_3d = live_3d
        self._evaluation_frame: Any = None

    # ---------------------------------------------------------- public API

    def start(self) -> None:
        """Start the campaign in a daemon thread; it runs at most once."""

        with self._lock:
            if self._thread is not None:
                raise RuntimeError("training campaign was already started")
            self._status = (
                STATUS_STOPPING if self._stop_requested.is_set() else STATUS_RUNNING
            )
            self._thread = threading.Thread(
                target=self._run, name="trailer-docking-campaign", daemon=True
            )
            thread = self._thread
        thread.start()

    def stop(self) -> None:
        """Request a graceful stop and return immediately."""

        with self._lock:
            self._stop_requested.set()
            if self._status == STATUS_RUNNING:
                self._status = STATUS_STOPPING
            session = self._session
        if session is not None:
            session.stop()

    def join(self, timeout: float | None = None) -> None:
        """Wait for the campaign thread; a no-op before :meth:`start`."""

        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(timeout)

    @property
    def progress(self) -> TrainingProgress:
        """Thread-safe snapshot across all runs of the campaign."""

        with self._lock:
            session = self._session
            status = self._status
            error = self._error
            run_index = self._run_index
            phase = self._phase
            finished_timesteps = self._finished_timesteps
            finished_episodes = len(self._finished_history)
            last = self._last_progress
            campaign_id = self._campaign_id
            best = self._best_checkpoint
            bundle = self._bundle_path
        current = session.progress if session is not None else last
        timesteps = finished_timesteps
        episodes = finished_episodes
        if session is not None and current is not None:
            timesteps += current.timesteps
            episodes += current.episodes
        checkpoint = best
        if checkpoint is None and current is not None:
            checkpoint = current.checkpoint_path
        return TrainingProgress(
            status=status,
            timesteps=timesteps,
            total_timesteps=self._training_runs() * self.config.total_timesteps,
            episodes=episodes,
            recent_success_rate=(
                None if current is None else current.recent_success_rate
            ),
            recent_mean_return=None if current is None else current.recent_mean_return,
            last_outcome=None if current is None else current.last_outcome,
            error=error,
            experiment_id=campaign_id,
            checkpoint_path=checkpoint,
            run=max(run_index, 1),
            runs=max(self._training_runs(), 1),
            phase=phase,
            bundle_path=bundle,
        )

    @property
    def spectator_state(self) -> VehicleState | None:
        """Latest vehicle state of the training or test episode in progress."""

        with self._lock:
            if self._evaluating:
                return self._evaluation_state
            session = self._session
            fallback = self._evaluation_state
        if session is None:
            return fallback
        state: VehicleState | None = session.spectator_state
        return state

    @property
    def spectator_frame(self) -> Any:
        """Latest rendered 3D image of the episode in progress, if any."""

        with self._lock:
            if self._evaluating:
                return self._evaluation_frame
            session = self._session
            fallback = self._evaluation_frame
        if session is None:
            return fallback
        return getattr(session, "spectator_frame", None)

    @property
    def spectator_route(self) -> tuple[Point2D, ...] | None:
        """Reference route of the training episode in progress, if any."""

        with self._lock:
            if self._evaluating:
                return None
            session = self._session
        if session is None:
            return None
        route = getattr(session, "spectator_route", None)
        return None if route is None else tuple(route)

    @property
    def episode_history(self) -> tuple[EpisodeRecord, ...]:
        """Every finished training episode of every run, oldest first."""

        with self._lock:
            finished = tuple(self._finished_history)
            session = self._session
        if session is None:
            return finished
        return finished + tuple(session.episode_history)

    @property
    def run_summaries(self) -> tuple[dict[str, Any], ...]:
        """One record per finished run, with its evaluation summaries."""

        with self._lock:
            return tuple(dict(run) for run in self._runs)

    # --------------------------------------------------------------- output

    def _reserve_directories(self) -> None:
        base = new_experiment_id(
            f"{self.config.purpose} campaign", method=self.config.method
        )
        for attempt in count(1):
            campaign_id = base if attempt == 1 else f"{base}-{attempt}"
            results = self.output_root / "results" / campaign_id
            artifacts = self.output_root / "artifacts" / campaign_id
            if results.exists() or artifacts.exists():
                continue
            results.parent.mkdir(parents=True, exist_ok=True)
            try:
                results.mkdir()
            except FileExistsError:
                continue
            artifacts.mkdir(parents=True, exist_ok=True)
            self._results_dir = results
            self._artifacts_dir = artifacts
            with self._lock:
                self._campaign_id = campaign_id
            return

    def _relative(self, path: Path | None) -> str | None:
        if path is None:
            return None
        try:
            return path.relative_to(self.output_root).as_posix()
        except ValueError:
            return path.as_posix()

    def _record(
        self,
        status: str,
        error: str | None,
        best: dict[str, Any] | None,
        *,
        finished: bool,
    ) -> None:
        """Write ``manifest.yaml`` and ``summary.json`` for the campaign.

        Called when the campaign starts, after every run, and at the end, so
        a campaign that is killed still leaves a record naming its runs.
        """

        assert self._results_dir is not None
        with self._lock:
            runs = [dict(run) for run in self._runs]
            bundle = self._bundle_path
        # Recorded paths are relative to the output root, like a run's own.
        for run in runs:
            if run["checkpoint_path"]:
                run["checkpoint_path"] = self._relative(Path(run["checkpoint_path"]))
        if best is not None:
            best = next(run for run in runs if run["run"] == best["run"])
        with self._lock:
            campaign_id = self._campaign_id
        manifest: dict[str, Any] = {
            "experiment_id": campaign_id,
            "kind": "campaign",
            "purpose": self.config.purpose,
            "method": self.config.method,
            "status": status,
            "error": error,
            "started": self._started_at,
            "finished": _timestamp() if finished else None,
            "scenario": self.scenario.to_dict(),
            "scenario_fingerprint": self.scenario.fingerprint,
            "config": self.config.to_dict(),
            "simulator_backend": self.config.backend,
            # Needed whenever any phase ran in MuJoCo, including tests of a
            # policy trained in 2D: no YAML fingerprint covers these values.
            "mujoco_tuning": _mujoco_tuning(self._mujoco_backend()),
            "run_experiment_ids": [run["experiment_id"] for run in runs],
            "best_run_experiment_id": None if best is None else best["experiment_id"],
            "bundle_path": self._relative(bundle),
            "bundle_sha256": file_sha256(bundle),
            "evaluation_files": sorted(
                path.name for path in self._results_dir.glob("evaluation_*.csv")
            ),
            "versions": _package_versions(),
            "runtime": runtime_summary(),
            "git": _git_revision(),
            "note": REWARD_NOTE,
        }
        (self._results_dir / "manifest.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
        )
        summary = {
            "experiment_id": campaign_id,
            "status": status,
            "runs_requested": self.config.runs,
            "runs_recorded": len(runs),
            "runs_finished": sum(1 for run in runs if run["status"] == STATUS_FINISHED),
            "best_run": best,
            "runs": runs,
        }
        (self._results_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )

    # --------------------------------------------------------------- phases

    def _set_phase(self, phase: str) -> None:
        with self._lock:
            self._phase = phase

    def _train(self, index: int) -> TrainingProgress:
        """Train run *index* (1-based) and return its final progress."""

        runs = self.config.runs
        config = replace(
            self.config,
            seed=self.config.seed + index - 1,
            purpose=f"{self.config.purpose} run {index} of {runs}",
            runs=1,
        )
        options: dict[str, Any] = {"output_root": self.output_root}
        if self._live_3d:
            options["live_3d"] = True
        with self._lock:
            options["campaign_id"] = self._campaign_id
        session = self._session_factory(self.scenario, config, **options)
        with self._lock:
            self._session = session
            self._run_index = index
            self._phase = f"training run {index} of {runs} ({config.backend})"
        if self._stop_requested.is_set():
            session.stop()
        session.start()
        while session.progress.status in _SESSION_ACTIVE:
            session.join(POLL_PERIOD)
            thread = getattr(session, "_thread", None)
            if thread is not None and not thread.is_alive():
                break
        final: TrainingProgress = session.progress
        history = tuple(session.episode_history)
        with self._lock:
            self._finished_timesteps += final.timesteps
            self._finished_history.extend(history)
            self._last_progress = final
            self._session = None
        return final

    def _on_evaluation_state(self, state: VehicleState) -> None:
        with self._lock:
            self._evaluation_state = state

    def _on_evaluation_frame(self, frame: Any) -> None:
        with self._lock:
            self._evaluation_frame = frame

    def _evaluate(self, index: int, checkpoint: Path | None) -> list[dict[str, Any]]:
        """Test one run's final policy on every evaluation backend.

        Without a checkpoint the model-based controller is tested instead.
        """

        config = self.config
        if config.evaluation_episodes == 0:
            return []
        policy: Any = None
        controller: dict[str, Any] = {}
        if checkpoint is not None:
            policy = self._policy_loader(checkpoint)
        else:
            controller["controller_factory"] = self._controller_loader()
        summaries: list[dict[str, Any]] = []
        with self._lock:
            self._evaluating = True
        try:
            for backend in config.evaluation_backends:
                if self._stop_requested.is_set():
                    break
                self._set_phase(
                    f"testing run {index} of {config.runs} in {backend}"
                    f" ({config.evaluation_episodes} episodes)"
                )
                options: dict[str, Any] = {}
                with self._lock:
                    self._evaluation_frame = None
                if self._live_3d and backend == "3d":
                    options["on_frame"] = self._on_evaluation_frame
                results = self._test_runner(
                    self.scenario,
                    policy,
                    episodes=config.evaluation_episodes,
                    seed=config.evaluation_seed,
                    max_episode_steps=config.max_episode_steps,
                    backend=backend,
                    on_state=self._on_evaluation_state,
                    should_stop=self._stop_requested.is_set,
                    **options,
                    **controller,
                )
                self._write_evaluation(index, backend, results)
                summary = _summarize(results)
                summary["backend"] = backend
                summary["seed"] = config.evaluation_seed
                summary["complete"] = summary["episodes"] == config.evaluation_episodes
                summaries.append(summary)
        finally:
            with self._lock:
                self._evaluating = False
        return summaries

    def _write_evaluation(self, index: int, backend: str, results: Any) -> None:
        """Keep every test episode, so spread can be computed without a rerun."""

        assert self._results_dir is not None
        rows = [
            result.to_dict() if hasattr(result, "to_dict") else dict(vars(result))
            for result in results
        ]
        if not rows:
            return
        path = self._results_dir / f"evaluation_run{index}_{backend}.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["episode", "seed", *rows[0]])
            writer.writeheader()
            for episode, row in enumerate(rows):
                writer.writerow(
                    {
                        "episode": episode,
                        "seed": self.config.evaluation_seed + episode,
                        **row,
                    }
                )

    def _export(self, best: dict[str, Any]) -> None:
        assert self._artifacts_dir is not None
        checkpoint = Path(best["checkpoint_path"])
        with self._lock:
            campaign_id = self._campaign_id
        path = export_bundle(
            checkpoint,
            self.scenario,
            self._artifacts_dir / BUNDLE_NAME,
            training=replace(self.config, seed=int(best["seed"])),
            source={
                "campaign_id": campaign_id,
                "experiment_id": best["experiment_id"],
                "run": best["run"],
                "seed": best["seed"],
                # A stopped run trained for fewer steps than its budget.
                "status": best["status"],
                "timesteps": best["timesteps"],
            },
            evaluation=best["evaluation"],
            provenance={
                "versions": _package_versions(),
                "git": _git_revision(),
                "mujoco_tuning": _mujoco_tuning(self._mujoco_backend()),
            },
        )
        with self._lock:
            self._bundle_path = path

    def _mujoco_backend(self) -> str:
        """``"3d"`` if training or any test of this campaign uses MuJoCo."""

        config = self.config
        trains_3d = config.method == METHOD_PPO and config.backend == "3d"
        tests_3d = config.evaluation_episodes > 0 and (
            "3d" in config.evaluation_backends
        )
        return "3d" if trains_3d or tests_3d else "2d"

    def _training_runs(self) -> int:
        return self.config.runs if self.config.method == METHOD_PPO else 0

    def _run_model_based(self) -> None:
        """Test the model-based controller; there is nothing to train."""

        with self._lock:
            self._run_index = 1
            campaign_id = self._campaign_id
        run: dict[str, Any] = {
            "run": 1,
            "seed": self.config.seed,
            "experiment_id": campaign_id,
            "status": STATUS_FINISHED,
            "error": None,
            "timesteps": 0,
            "episodes": 0,
            "checkpoint_path": None,
            "evaluation": self._evaluate(1, None),
            "evaluation_error": None,
        }
        with self._lock:
            self._runs.append(run)

    # -------------------------------------------------------- thread target

    def _run(self) -> None:
        status = STATUS_FAILED
        error: str | None = None
        best: dict[str, Any] | None = None
        failures: list[str] = []
        try:
            self._started_at = _timestamp()
            self._reserve_directories()
            self._record(STATUS_RUNNING, None, None, finished=False)
            if self.config.method == METHOD_NMPC:
                self._run_model_based()
            for index in range(1, self._training_runs() + 1):
                if self._stop_requested.is_set():
                    break
                final = self._train(index)
                run: dict[str, Any] = {
                    "run": index,
                    "seed": self.config.seed + index - 1,
                    "experiment_id": final.experiment_id,
                    "status": final.status,
                    "error": final.error,
                    "timesteps": final.timesteps,
                    "episodes": final.episodes,
                    "checkpoint_path": (
                        None
                        if final.checkpoint_path is None
                        else str(final.checkpoint_path)
                    ),
                    "evaluation": [],
                    "evaluation_error": None,
                }
                # The run is recorded before it is tested, so a failing test
                # cannot lose a completed training run.
                with self._lock:
                    self._runs.append(run)
                if final.status == STATUS_FAILED:
                    failures.append(
                        f"run {index} failed: {final.error or 'unknown error'}"
                    )
                elif final.checkpoint_path is not None:
                    try:
                        evaluation = self._evaluate(index, final.checkpoint_path)
                        with self._lock:
                            run["evaluation"] = evaluation
                    except Exception as exc:
                        message = str(exc) or type(exc).__name__
                        with self._lock:
                            run["evaluation_error"] = message
                        failures.append(f"testing run {index} failed: {message}")
                self._record(STATUS_RUNNING, None, None, finished=False)
            with self._lock:
                usable = [
                    run
                    for run in self._runs
                    if run["checkpoint_path"] and run["status"] != STATUS_FAILED
                ]
            if self.config.method == METHOD_NMPC:
                with self._lock:
                    best = self._runs[0] if self._runs else None
            elif usable:
                best = max(usable, key=run_ranking)
                with self._lock:
                    self._best_checkpoint = Path(best["checkpoint_path"])
                self._set_phase("exporting the best policy")
                self._export(best)
            if failures:
                # Whatever did finish is still selected and exported above.
                error = "; ".join(failures)
            elif self._stop_requested.is_set():
                status = STATUS_STOPPED
            else:
                status = STATUS_FINISHED
        except BaseException as exc:
            # Nothing may escape the worker thread: a status left at
            # "running" would make every waiting caller wait forever.
            error = str(exc) or type(exc).__name__
        try:
            if self._results_dir is not None:
                self._record(status, error, best, finished=True)
        except Exception as exc:
            status = STATUS_FAILED
            error = error or f"could not write campaign results: {exc}"
        with self._lock:
            self._session = None
            self._error = error
            self._status = status
            self._phase = ""
