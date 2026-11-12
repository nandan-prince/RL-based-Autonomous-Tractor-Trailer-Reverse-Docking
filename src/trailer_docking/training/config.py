"""Hyperparameters and budget for one PPO training run."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Everything a :class:`~trailer_docking.training.session.TrainingSession`
    needs besides the scenario.

    ``total_timesteps`` is the requested budget; PPO collects whole rollouts,
    so the run ends at the first multiple of ``n_steps`` that reaches it.
    ``purpose`` becomes the short-purpose part of the experiment ID.
    """

    total_timesteps: int = 200_000
    seed: int = 0
    learning_rate: float = 3e-4
    n_steps: int = 1024
    batch_size: int = 256
    max_episode_steps: int = 900
    checkpoint_interval: int = 20_000
    purpose: str = "docking"

    def __post_init__(self) -> None:
        for name in (
            "total_timesteps",
            "n_steps",
            "batch_size",
            "max_episode_steps",
            "checkpoint_interval",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if self.seed < 0:
            raise ValueError("seed cannot be negative")
        rate = self.learning_rate
        if (
            isinstance(rate, bool)
            or not isinstance(rate, (int, float))
            or not isfinite(rate)
            or rate <= 0.0
        ):
            raise ValueError("learning_rate must be a positive finite number")
        if not isinstance(self.purpose, str) or not self.purpose.strip():
            raise ValueError("purpose cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-compatible representation."""

        return asdict(self)
