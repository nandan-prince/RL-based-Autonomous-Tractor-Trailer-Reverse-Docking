"""Docking tests: roll a policy out on a scenario and score the episodes.

A *policy* here is any callable mapping one observation to one action, so a
trained PPO checkpoint, a hand-written baseline and a classical controller
wrapper are all tested through the same function.
"""

from __future__ import annotations

import io
import time
import zipfile
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from math import hypot, sqrt
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from trailer_docking.core.state import VehicleState
from trailer_docking.evaluation.scenarios import Scenario
from trailer_docking.training.config import validate_backend
from trailer_docking.training.methods import ControllerFactory

if TYPE_CHECKING:
    from trailer_docking.training.environment import EnvironmentFactory

Policy = Callable[[np.ndarray], np.ndarray]
"""Observation -> action shaped ``(1,)`` (normalized steering rate)."""


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """Outcome, final errors, and maneuver metrics of one test episode.

    ``maneuver_time`` is simulated seconds, ``path_length`` the distance the
    docking point travelled in metres, ``steering_rate_rms`` the root mean
    square of the measured steering rate in rad/s (lower is smoother), and
    ``mean_policy_time_ms`` the mean wall-clock time of one policy call.
    """

    outcome: str
    success: bool
    steps: int
    total_reward: float
    final_longitudinal_error: float
    final_lateral_error: float
    final_yaw_error: float
    max_abs_articulation: float
    maneuver_time: float = 0.0
    path_length: float = 0.0
    steering_rate_rms: float = 0.0
    # Wall-clock time differs between identical rollouts.
    mean_policy_time_ms: float = field(default=0.0, compare=False)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-compatible representation."""

        return asdict(self)


def straight_reverse_policy(observation: np.ndarray) -> np.ndarray:
    """Never steer: a baseline that needs no training."""

    return np.zeros((1,), dtype=np.float32)


def load_policy(path: str | Path) -> Policy:
    """Load a Stable-Baselines3 PPO ``.zip`` as a deterministic policy."""

    source = Path(path)
    if not source.is_file():
        zipped = source.with_name(source.name + ".zip")
        if not zipped.is_file():
            raise FileNotFoundError(f"policy file not found: {source}")
        source = zipped
    return policy_from_bytes(source.read_bytes())


class PolicyFormatError(ValueError):
    """Raised when a file is not a policy this framework can load."""


FIRST_LAYER = "mlp_extractor.policy_net.0.weight"


def _policy_weights(content: bytes) -> dict[str, Any]:
    """Read the network weights of a PPO ``.zip`` without unpickling objects.

    Only ``policy.pth`` is read, with ``weights_only=True``.  The archive's
    ``data`` member is never parsed: Stable-Baselines3 stores pickled Python
    objects there, and unpickling a file from someone else can run arbitrary
    code.
    """

    import pickle

    import torch

    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            weights = torch.load(
                io.BytesIO(archive.read("policy.pth")),
                map_location="cpu",
                weights_only=True,
            )
    except (
        zipfile.BadZipFile,
        KeyError,
        EOFError,
        OSError,
        RuntimeError,
        pickle.UnpicklingError,
    ) as error:
        raise PolicyFormatError(f"not a saved policy: {error}") from error
    if not isinstance(weights, dict) or FIRST_LAYER not in weights:
        raise PolicyFormatError(
            "not a saved policy: the file does not hold the default PPO network"
        )
    return weights


def policy_observation_size(content: bytes) -> int:
    """Return how many observation values a saved policy expects."""

    return int(_policy_weights(content)[FIRST_LAYER].shape[1])


def policy_from_bytes(content: bytes) -> Policy:
    """Build a deterministic policy from the bytes of a PPO ``.zip``.

    The network is rebuilt as the default ``MlpPolicy`` this framework trains
    (bounded observation of the size the weights imply, one bounded action)
    and the weights are loaded into it.  This avoids ``PPO.load``, which
    unpickles the archive's metadata and which some PyTorch releases cannot
    read from inside a zip (torch 2.15 with SB3 2.9).  A checkpoint trained
    with a different network architecture is rejected.
    """

    from gymnasium import spaces
    from stable_baselines3.common.policies import ActorCriticPolicy

    weights = _policy_weights(content)
    size = int(weights[FIRST_LAYER].shape[1])
    model = ActorCriticPolicy(
        spaces.Box(-1.0, 1.0, (size,), dtype=np.float32),
        spaces.Box(-1.0, 1.0, (1,), dtype=np.float32),
        lambda _: 0.0,
    )
    try:
        model.load_state_dict(weights)
    except RuntimeError as error:
        raise PolicyFormatError(
            f"not a policy trained by this framework: {error}"
        ) from error
    model.set_training_mode(False)

    def policy(observation: np.ndarray) -> np.ndarray:
        action, _ = model.predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.float32).reshape((1,))

    return policy


def run_docking_test(
    scenario: Scenario,
    policy: Policy | None,
    *,
    episodes: int = 1,
    seed: int = 0,
    max_episode_steps: int | None = 900,
    backend: str = "2d",
    environment_factory: EnvironmentFactory | None = None,
    on_state: Callable[[VehicleState], None] | None = None,
    on_frame: Callable[[Any], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    controller_factory: ControllerFactory | None = None,
) -> list[EpisodeResult]:
    """Run ``episodes`` episodes of ``policy`` and return their results.

    Episode ``i`` is reset with ``seed + i``, so a test is reproducible and
    two policies tested with the same seed face the same start poses.
    ``max_episode_steps=None`` uses the task configuration's own step limit.
    ``backend`` chooses the simulator: ``"2d"`` (kinematic) or ``"3d"``
    (MuJoCo); each call builds a fresh one.  ``on_frame`` receives rendered
    images of 3D episodes.  A ``controller_factory`` replaces ``policy`` with
    a controller built from the environment at the start of every episode,
    which is how a model-based controller such as NMPC is tested.  If
    ``should_stop()`` becomes true the test ends immediately and only the
    episodes that already finished are returned.  Outcomes are the task's
    termination reasons: ``success``, ``collision``, ``jackknife``,
    ``out_of_bounds`` or ``time_limit``.
    """

    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    if (policy is None) == (controller_factory is None):
        raise ValueError("give either a policy or a controller_factory")

    from trailer_docking.training.environment import (
        ScenarioDockingEnv,
        episode_outcome,
    )

    def stopping() -> bool:
        return should_stop is not None and should_stop()

    factory = environment_factory or ScenarioDockingEnv
    env = factory(
        scenario,
        max_episode_steps=max_episode_steps,
        backend=validate_backend(backend),
        on_state=on_state,
        on_frame=on_frame,
    )
    results: list[EpisodeResult] = []
    try:
        for index in range(episodes):
            if stopping():
                break
            observation, info = env.reset(seed=seed + index)
            act = policy if policy is not None else controller_factory(env)  # type: ignore[misc]
            max_articulation = abs(env.state.articulation_angle)
            total_reward = 0.0
            steps = 0
            finished = False
            path_length = 0.0
            rate_squares = 0.0
            policy_seconds = 0.0
            previous = env.built.vehicle.trailer.rear_face(env.state.trailer_pose)
            while not finished:
                if stopping():
                    return results
                called = time.perf_counter()
                action = np.asarray(act(observation), dtype=np.float32).reshape((1,))
                policy_seconds += time.perf_counter() - called
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                steps += 1
                state = env.state
                point = env.built.vehicle.trailer.rear_face(state.trailer_pose)
                path_length += hypot(point.x - previous.x, point.y - previous.y)
                previous = point
                rate_squares += state.steering_rate**2
                max_articulation = max(
                    max_articulation, abs(env.state.articulation_angle)
                )
                finished = terminated or truncated
            results.append(
                EpisodeResult(
                    outcome=episode_outcome(info),
                    success=bool(info["is_success"]),
                    steps=steps,
                    total_reward=total_reward,
                    final_longitudinal_error=float(info["docking_error_longitudinal"]),
                    final_lateral_error=float(info["docking_error_lateral"]),
                    final_yaw_error=float(info["docking_error_yaw"]),
                    max_abs_articulation=max_articulation,
                    maneuver_time=env.state.simulation_time,
                    path_length=path_length,
                    steering_rate_rms=sqrt(rate_squares / steps),
                    mean_policy_time_ms=1000.0 * policy_seconds / steps,
                )
            )
    finally:
        env.close()
    return results


def summarize(results: Sequence[EpisodeResult]) -> dict[str, float | int]:
    """Aggregate episode results into headline numbers.

    Keys: ``episodes``, ``success_rate``, ``collision_rate``,
    ``jackknife_rate``, ``mean_total_reward``, ``mean_abs_lateral_error``,
    ``mean_abs_longitudinal_error``, ``mean_abs_yaw_error``,
    ``max_abs_articulation``, ``mean_maneuver_time``, ``mean_path_length``,
    ``mean_steering_rate_rms``, ``mean_policy_time_ms`` and one
    ``outcome_<name>`` count per outcome that occurred.  An empty input
    yields zero episodes and zero rates.
    """

    count = len(results)
    summary: dict[str, float | int] = {
        "episodes": count,
        "success_rate": 0.0,
        "collision_rate": 0.0,
        "jackknife_rate": 0.0,
        "mean_total_reward": 0.0,
        "mean_abs_lateral_error": 0.0,
        "mean_abs_longitudinal_error": 0.0,
        "mean_abs_yaw_error": 0.0,
        "max_abs_articulation": 0.0,
        "mean_maneuver_time": 0.0,
        "mean_path_length": 0.0,
        "mean_steering_rate_rms": 0.0,
        "mean_policy_time_ms": 0.0,
    }
    if count == 0:
        return summary

    def mean(values: Iterable[float]) -> float:
        return sum(values) / count

    summary["success_rate"] = mean(1.0 for r in results if r.success)
    summary["collision_rate"] = mean(1.0 for r in results if r.outcome == "collision")
    summary["jackknife_rate"] = mean(1.0 for r in results if r.outcome == "jackknife")
    summary["mean_total_reward"] = mean(r.total_reward for r in results)
    summary["mean_abs_lateral_error"] = mean(
        abs(r.final_lateral_error) for r in results
    )
    summary["mean_abs_longitudinal_error"] = mean(
        abs(r.final_longitudinal_error) for r in results
    )
    summary["mean_abs_yaw_error"] = mean(abs(r.final_yaw_error) for r in results)
    summary["max_abs_articulation"] = max(r.max_abs_articulation for r in results)
    summary["mean_maneuver_time"] = mean(r.maneuver_time for r in results)
    summary["mean_path_length"] = mean(r.path_length for r in results)
    summary["mean_steering_rate_rms"] = mean(r.steering_rate_rms for r in results)
    summary["mean_policy_time_ms"] = mean(r.mean_policy_time_ms for r in results)
    outcomes = Counter(r.outcome for r in results)
    for name in sorted(outcomes):
        summary[f"outcome_{name}"] = outcomes[name]
    return summary
