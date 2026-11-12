"""Background PPO training session with live, thread-safe progress.

A :class:`TrainingSession` trains one policy on one scenario in a daemon
thread so a UI (or the CLI) can poll :attr:`TrainingSession.progress`, watch
the vehicle through :attr:`TrainingSession.spectator_state` and request a
graceful stop.  Outputs follow the experiment layout of ``AGENTS.md`` 6.1::

    <output_root>/results/<experiment_id>/manifest.yaml
    <output_root>/results/<experiment_id>/metrics.csv
    <output_root>/artifacts/<experiment_id>/checkpoints/policy_<timesteps>.zip
    <output_root>/artifacts/<experiment_id>/checkpoints/policy_final.zip

Stable-Baselines3, PyTorch and Gymnasium are imported inside the training
thread, so importing this module and constructing a session need none of them.
"""

from __future__ import annotations

import csv
import re
import subprocess
import threading
from collections import deque
from dataclasses import dataclass, fields
from datetime import datetime
from importlib import metadata
from itertools import count
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from trailer_docking.core.state import VehicleState
from trailer_docking.evaluation.scenarios import Scenario
from trailer_docking.training.config import TrainingConfig

if TYPE_CHECKING:
    from stable_baselines3.common.base_class import BaseAlgorithm
    from stable_baselines3.common.callbacks import BaseCallback

    from trailer_docking.environments.docking_env import SimulatorFactory

RECENT_EPISODES = 50
METHOD = "ppo"
FINAL_POLICY_NAME = "policy_final.zip"
VERSIONED_PACKAGES = ("stable_baselines3", "gymnasium", "torch", "mujoco")
REWARD_NOTE = (
    "The DockingEnv reward is a placeholder (progress, step penalty and "
    "terminal bonus/penalty). Do not interpret returns or success rates from "
    "this run as research results."
)

STATUS_IDLE = "idle"
STATUS_RUNNING = "running"
STATUS_STOPPING = "stopping"
STATUS_FINISHED = "finished"
STATUS_STOPPED = "stopped"
STATUS_FAILED = "failed"


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    """One finished training episode; also one row of ``metrics.csv``."""

    index: int
    timesteps: int
    outcome: str
    success: bool
    steps: int
    total_reward: float


@dataclass(frozen=True, slots=True)
class TrainingProgress:
    """Immutable snapshot of a training session.

    ``status`` is one of ``"idle"``, ``"running"``, ``"stopping"``,
    ``"finished"``, ``"stopped"`` or ``"failed"``.  The ``recent_*`` values
    cover the last 50 episodes and are ``None`` before the first one ends.
    """

    status: str
    timesteps: int
    total_timesteps: int
    episodes: int
    recent_success_rate: float | None
    recent_mean_return: float | None
    last_outcome: str | None
    error: str | None
    experiment_id: str | None
    checkpoint_path: Path | None


def new_experiment_id(purpose: str, now: datetime | None = None) -> str:
    """Return ``YYYYMMDD-HHMM_ppo_<purpose-slug>`` for a new run."""

    slug = re.sub(r"[^a-z0-9]+", "-", purpose.lower()).strip("-") or "run"
    moment = now if now is not None else datetime.now()
    return f"{moment:%Y%m%d-%H%M}_{METHOD}_{slug}"


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in VERSIONED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _git_revision() -> dict[str, Any]:
    """Best-effort commit and dirty flag of the checkout providing this code."""

    def git(*arguments: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *arguments],
                cwd=Path(__file__).resolve().parent,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip() if completed.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {"commit": commit, "dirty": None if status is None else bool(status)}


class TrainingSession:
    """Train PPO on one scenario in a background thread."""

    def __init__(
        self,
        scenario: Scenario,
        config: TrainingConfig | None = None,
        *,
        output_root: str | Path = ".",
        simulator_factory: SimulatorFactory | None = None,
    ) -> None:
        self.scenario = scenario
        self.config = config if config is not None else TrainingConfig()
        self.output_root = Path(output_root)
        self._simulator_factory = simulator_factory
        self._lock = threading.Lock()
        self._stop_requested = threading.Event()
        self._thread: threading.Thread | None = None
        self._status = STATUS_IDLE
        self._timesteps = 0
        self._error: str | None = None
        self._experiment_id: str | None = None
        self._checkpoint_path: Path | None = None
        self._spectator_state: VehicleState | None = None
        self._history: list[EpisodeRecord] = []
        self._recent: deque[EpisodeRecord] = deque(maxlen=RECENT_EPISODES)
        self._stopped_early = False
        self._started_at: str | None = None
        self._results_dir: Path | None = None
        self._checkpoint_dir: Path | None = None

    # ---------------------------------------------------------- public API

    def start(self) -> None:
        """Start training in a daemon thread; a session runs at most once."""

        with self._lock:
            if self._thread is not None:
                raise RuntimeError("training session was already started")
            self._status = (
                STATUS_STOPPING if self._stop_requested.is_set() else STATUS_RUNNING
            )
            self._thread = threading.Thread(
                target=self._run, name="trailer-docking-training", daemon=True
            )
            thread = self._thread
        thread.start()

    def stop(self) -> None:
        """Request a graceful stop and return immediately."""

        with self._lock:
            self._stop_requested.set()
            if self._status == STATUS_RUNNING:
                self._status = STATUS_STOPPING

    def join(self, timeout: float | None = None) -> None:
        """Wait for the training thread; a no-op before :meth:`start`."""

        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(timeout)

    @property
    def progress(self) -> TrainingProgress:
        """Thread-safe snapshot of the session."""

        with self._lock:
            recent = tuple(self._recent)
            success_rate: float | None = None
            mean_return: float | None = None
            if recent:
                success_rate = sum(1 for r in recent if r.success) / len(recent)
                mean_return = sum(r.total_reward for r in recent) / len(recent)
            return TrainingProgress(
                status=self._status,
                timesteps=self._timesteps,
                total_timesteps=self.config.total_timesteps,
                episodes=len(self._history),
                recent_success_rate=success_rate,
                recent_mean_return=mean_return,
                last_outcome=recent[-1].outcome if recent else None,
                error=self._error,
                experiment_id=self._experiment_id,
                checkpoint_path=self._checkpoint_path,
            )

    @property
    def spectator_state(self) -> VehicleState | None:
        """Latest vehicle state seen by the training environment."""

        with self._lock:
            return self._spectator_state

    @property
    def episode_history(self) -> tuple[EpisodeRecord, ...]:
        """Every finished training episode so far, oldest first."""

        with self._lock:
            return tuple(self._history)

    # ------------------------------------------- training-thread callbacks

    def _on_state(self, state: VehicleState) -> None:
        with self._lock:
            self._spectator_state = state

    def _set_timesteps(self, timesteps: int) -> None:
        with self._lock:
            self._timesteps = timesteps

    def _record_episode(
        self, timesteps: int, info: dict[str, Any], steps: int, total_reward: float
    ) -> None:
        with self._lock:
            record = EpisodeRecord(
                index=len(self._history),
                timesteps=timesteps,
                outcome=str(info.get("outcome", "unknown")),
                success=bool(info.get("is_success", False)),
                steps=steps,
                total_reward=total_reward,
            )
            self._history.append(record)
            self._recent.append(record)

    def _save_policy(self, model: BaseAlgorithm, name: str) -> None:
        assert self._checkpoint_dir is not None
        path = self._checkpoint_dir / name
        model.save(str(path))
        with self._lock:
            self._checkpoint_path = path

    def _make_callback(self) -> BaseCallback:
        from stable_baselines3.common.callbacks import BaseCallback

        session = self
        interval = self.config.checkpoint_interval

        class SessionCallback(BaseCallback):
            """Bridges the SB3 training loop to the session."""

            def __init__(self) -> None:
                super().__init__(verbose=0)
                self._episode_reward = 0.0
                self._episode_steps = 0
                self._last_checkpoint = 0

            def _on_step(self) -> bool:
                timesteps = int(self.num_timesteps)
                session._set_timesteps(timesteps)
                # One non-vectorized environment: index 0 is the only one.
                self._episode_reward += float(self.locals["rewards"][0])
                self._episode_steps += 1
                if bool(self.locals["dones"][0]):
                    session._record_episode(
                        timesteps,
                        self.locals["infos"][0],
                        self._episode_steps,
                        self._episode_reward,
                    )
                    self._episode_reward = 0.0
                    self._episode_steps = 0
                if timesteps - self._last_checkpoint >= interval:
                    self._last_checkpoint = timesteps
                    session._save_policy(self.model, f"policy_{timesteps}.zip")
                if session._stop_requested.is_set():
                    session._stopped_early = True
                    return False
                return True

        return SessionCallback()

    # ------------------------------------------------------ output on disk

    def _reserve_directories(self) -> None:
        """Claim an unused experiment ID; never reuse a results directory."""

        base = new_experiment_id(self.config.purpose)
        for attempt in count(1):
            experiment_id = base if attempt == 1 else f"{base}-{attempt}"
            results = self.output_root / "results" / experiment_id
            artifacts = self.output_root / "artifacts" / experiment_id
            if results.exists() or artifacts.exists():
                continue
            results.parent.mkdir(parents=True, exist_ok=True)
            try:
                results.mkdir()
            except FileExistsError:
                continue
            checkpoints = artifacts / "checkpoints"
            checkpoints.mkdir(parents=True, exist_ok=True)
            self._results_dir = results
            self._checkpoint_dir = checkpoints
            with self._lock:
                self._experiment_id = experiment_id
            return

    def _relative(self, path: Path | None) -> str | None:
        if path is None:
            return None
        try:
            return path.relative_to(self.output_root).as_posix()
        except ValueError:
            return path.as_posix()

    def _write_manifest(self, status: str, error: str | None, finished: bool) -> None:
        assert self._results_dir is not None
        with self._lock:
            timesteps = self._timesteps
            episodes = len(self._history)
            checkpoint = self._checkpoint_path
            experiment_id = self._experiment_id
        manifest: dict[str, Any] = {
            "experiment_id": experiment_id,
            "purpose": self.config.purpose,
            "method": METHOD,
            "status": status,
            "error": error,
            "started": self._started_at,
            "finished": _timestamp() if finished else None,
            "scenario": self.scenario.to_dict(),
            "scenario_fingerprint": self.scenario.fingerprint,
            "config": self.config.to_dict(),
            "timesteps": timesteps,
            "episodes": episodes,
            "versions": _package_versions(),
            "git": _git_revision(),
            "checkpoint_path": self._relative(checkpoint),
            "checkpoint_dir": self._relative(self._checkpoint_dir),
            "metrics_path": "metrics.csv" if finished else None,
            "note": REWARD_NOTE,
        }
        (self._results_dir / "manifest.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
        )

    def _write_metrics(self) -> None:
        assert self._results_dir is not None
        names = [item.name for item in fields(EpisodeRecord)]
        with (self._results_dir / "metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(names)
            for record in self.episode_history:
                writer.writerow([getattr(record, name) for name in names])

    # ------------------------------------------------------- thread target

    def _run(self) -> None:
        status = STATUS_FAILED
        error: str | None = None
        model: BaseAlgorithm | None = None
        env: Any = None
        try:
            self._started_at = _timestamp()
            self._reserve_directories()
            self._write_manifest(STATUS_RUNNING, None, finished=False)

            from stable_baselines3 import PPO

            from trailer_docking.environments.docking_env import DockingEnv

            env = DockingEnv(
                self.scenario,
                simulator_factory=self._simulator_factory,
                max_episode_steps=self.config.max_episode_steps,
                on_state=self._on_state,
            )
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=self.config.learning_rate,
                n_steps=self.config.n_steps,
                batch_size=self.config.batch_size,
                seed=self.config.seed,
                device="cpu",
                verbose=0,
            )
            model.learn(
                total_timesteps=self.config.total_timesteps,
                callback=self._make_callback(),
            )
            status = STATUS_STOPPED if self._stopped_early else STATUS_FINISHED
        except Exception as exc:
            error = str(exc) or type(exc).__name__

        # Finalize: keep whatever was learned and always leave a manifest.
        try:
            if model is not None and (
                status == STATUS_FINISHED or getattr(model, "_n_updates", 0) > 0
            ):
                self._save_policy(model, FINAL_POLICY_NAME)
        except Exception as exc:
            status = STATUS_FAILED
            error = error or f"could not save the final policy: {exc}"
        try:
            if env is not None:
                env.close()
        except Exception as exc:
            error = error or f"could not close the environment: {exc}"
        try:
            if self._results_dir is not None:
                self._write_metrics()
                self._write_manifest(status, error, finished=True)
        except Exception as exc:
            status = STATUS_FAILED
            error = error or f"could not write results: {exc}"
        with self._lock:
            self._error = error
            self._status = status
