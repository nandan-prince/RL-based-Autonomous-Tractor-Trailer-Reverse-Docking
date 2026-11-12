"""Docking tests: roll a policy out on a scenario and score the episodes.

A *policy* here is any callable mapping one observation to one action, so a
trained PPO checkpoint, a hand-written baseline and a classical controller
wrapper are all tested through the same function.
"""

from __future__ import annotations

import io
import zipfile
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from trailer_docking.core.state import VehicleState
from trailer_docking.evaluation.scenarios import Scenario

if TYPE_CHECKING:
    from trailer_docking.environments.docking_env import SimulatorFactory

Policy = Callable[[np.ndarray], np.ndarray]
"""Observation -> action shaped ``(1,)`` (normalized steering rate)."""

UNKNOWN_OUTCOME = "unknown"


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """Outcome and final errors of one docking-test episode."""

    outcome: str
    success: bool
    steps: int
    total_reward: float
    final_longitudinal_error: float
    final_lateral_error: float
    final_yaw_error: float
    max_abs_articulation: float

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

    import torch
    from stable_baselines3.common.save_util import json_to_data

    # Rebuild only the policy network instead of calling ``PPO.load``:
    # ``PPO.load`` hands ``torch.load`` a stream inside the zip archive, which
    # some PyTorch releases cannot read (torch 2.15 with SB3 2.9 fails with a
    # "PytorchStreamReader failed reading file" error).  Reading each member
    # into memory first works with every release, and inference needs no more
    # than the policy.
    with zipfile.ZipFile(source) as archive:
        data = json_to_data(archive.read("data").decode())
        weights = torch.load(
            io.BytesIO(archive.read("policy.pth")),
            map_location="cpu",
            weights_only=True,
        )
    model = data["policy_class"](
        data["observation_space"],
        data["action_space"],
        lambda _: 0.0,
        use_sde=data.get("use_sde", False),
        **(data.get("policy_kwargs") or {}),
    )
    model.load_state_dict(weights)
    model.set_training_mode(False)

    def policy(observation: np.ndarray) -> np.ndarray:
        action, _ = model.predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.float32).reshape((1,))

    return policy


def run_docking_test(
    scenario: Scenario,
    policy: Policy,
    *,
    episodes: int = 1,
    seed: int = 0,
    max_episode_steps: int = 900,
    simulator_factory: SimulatorFactory | None = None,
    on_state: Callable[[VehicleState], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> list[EpisodeResult]:
    """Run ``episodes`` episodes of ``policy`` and return their results.

    Episode ``i`` is reset with ``seed + i``, so a test is reproducible and
    two policies tested with the same seed face the same start poses.  If
    ``should_stop()`` becomes true the test ends immediately and only the
    episodes that already finished are returned.
    """

    if episodes < 1:
        raise ValueError("episodes must be at least 1")

    from trailer_docking.environments.docking_env import DockingEnv

    def stopping() -> bool:
        return should_stop is not None and should_stop()

    env = DockingEnv(
        scenario,
        simulator_factory=simulator_factory,
        max_episode_steps=max_episode_steps,
        on_state=on_state,
    )
    results: list[EpisodeResult] = []
    try:
        for index in range(episodes):
            if stopping():
                break
            observation, info = env.reset(seed=seed + index)
            max_articulation = abs(float(info["articulation_angle"]))
            total_reward = 0.0
            steps = 0
            finished = False
            while not finished:
                if stopping():
                    return results
                action = np.asarray(policy(observation), dtype=np.float32)
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                steps += 1
                max_articulation = max(
                    max_articulation, abs(float(info["articulation_angle"]))
                )
                finished = terminated or truncated
            results.append(
                EpisodeResult(
                    outcome=str(info.get("outcome", UNKNOWN_OUTCOME)),
                    success=bool(info["is_success"]),
                    steps=steps,
                    total_reward=total_reward,
                    final_longitudinal_error=float(info["longitudinal_error"]),
                    final_lateral_error=float(info["lateral_error"]),
                    final_yaw_error=float(info["yaw_error"]),
                    max_abs_articulation=max_articulation,
                )
            )
    finally:
        env.close()
    return results


def summarize(results: Sequence[EpisodeResult]) -> dict[str, float | int]:
    """Aggregate episode results into headline numbers.

    Keys: ``episodes``, ``success_rate``, ``mean_total_reward``,
    ``mean_abs_lateral_error`` and one ``outcome_<name>`` count per outcome
    that occurred.  An empty input yields zero episodes and zero rates.
    """

    count = len(results)
    summary: dict[str, float | int] = {
        "episodes": count,
        "success_rate": 0.0,
        "mean_total_reward": 0.0,
        "mean_abs_lateral_error": 0.0,
    }
    if count == 0:
        return summary
    summary["success_rate"] = sum(1 for r in results if r.success) / count
    summary["mean_total_reward"] = sum(r.total_reward for r in results) / count
    summary["mean_abs_lateral_error"] = (
        sum(abs(r.final_lateral_error) for r in results) / count
    )
    outcomes = Counter(r.outcome for r in results)
    for name in sorted(outcomes):
        summary[f"outcome_{name}"] = outcomes[name]
    return summary
