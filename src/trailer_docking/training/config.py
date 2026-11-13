"""Hyperparameters, budget, and repetition settings for PPO training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any

from trailer_docking.training.methods import validate_method

BACKENDS = ("2d", "3d")
"""``"2d"`` is the kinematic simulator; ``"3d"`` is MuJoCo."""


def validate_backend(value: object, name: str = "backend") -> str:
    """Return *value* if it names a simulator backend, else raise."""

    if value not in BACKENDS:
        raise ValueError(f"{name} must be one of {', '.join(BACKENDS)}")
    return str(value)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Everything a training run or campaign needs besides the scenario.

    ``total_timesteps`` is the requested budget of *one* run; PPO collects
    whole rollouts, so a run ends at the first multiple of ``n_steps`` that
    reaches it.  ``purpose`` becomes the short-purpose part of the experiment
    ID.

    A campaign repeats the run ``runs`` times with seeds ``seed``,
    ``seed + 1``, ... on the ``backend`` simulator.  After each run the final
    policy is tested for ``evaluation_episodes`` episodes on every backend in
    ``evaluation_backends``, starting from ``evaluation_seed`` so every run
    faces the same start poses.  Zero evaluation episodes skips the tests.

    ``method`` is ``"ppo"`` or ``"nmpc"``.  NMPC is not trained: a campaign
    with that method skips training and only runs the tests, so the PPO
    hyperparameters and ``runs`` are then unused.
    """

    total_timesteps: int = 200_000
    seed: int = 0
    learning_rate: float = 3e-4
    n_steps: int = 1024
    batch_size: int = 256
    max_episode_steps: int = 900
    checkpoint_interval: int = 20_000
    purpose: str = "docking"
    runs: int = 1
    backend: str = "2d"
    evaluation_episodes: int = 20
    evaluation_backends: tuple[str, ...] = ("2d",)
    evaluation_seed: int = 100_000
    method: str = "ppo"

    def __post_init__(self) -> None:
        validate_method(self.method)
        for name in (
            "total_timesteps",
            "n_steps",
            "batch_size",
            "max_episode_steps",
            "checkpoint_interval",
            "runs",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("seed", "evaluation_seed", "evaluation_episodes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
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
        validate_backend(self.backend)
        backends = self.evaluation_backends
        if isinstance(backends, str):
            raise ValueError("evaluation_backends must be a sequence of backends")
        chosen = tuple(
            validate_backend(item, "evaluation_backends") for item in backends
        )
        if len(set(chosen)) != len(chosen):
            raise ValueError("evaluation_backends cannot repeat a backend")
        object.__setattr__(self, "evaluation_backends", chosen)
        if self.evaluation_episodes > 0 and not chosen:
            raise ValueError(
                "evaluation_backends cannot be empty when evaluation_episodes > 0"
            )
        if self.batch_size > self.n_steps:
            raise ValueError("batch_size cannot exceed n_steps")
        if self.method == "nmpc" and self.evaluation_episodes == 0:
            raise ValueError(
                "nmpc is only tested, so evaluation_episodes must be at least 1"
            )
        # Tests must not start from the poses a run trained on first.
        first_test = self.evaluation_seed
        last_test = first_test + self.evaluation_episodes - 1
        if (
            self.method == "ppo"
            and self.evaluation_episodes > 0
            and first_test <= self.seed + self.runs - 1
            and self.seed <= last_test
        ):
            raise ValueError(
                "training seeds and evaluation seeds overlap; change seed or"
                " evaluation_seed"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-compatible representation."""

        data = asdict(self)
        data["evaluation_backends"] = list(self.evaluation_backends)
        return data

    @classmethod
    def from_dict(cls, data: Any) -> TrainingConfig:
        """Rebuild a configuration saved by :meth:`to_dict`."""

        if not isinstance(data, dict):
            raise ValueError("training configuration must be a mapping")
        values = dict(data)
        try:
            if "evaluation_backends" in values:
                values["evaluation_backends"] = tuple(values["evaluation_backends"])
            return cls(**values)
        except TypeError as error:
            raise ValueError(f"invalid training configuration: {error}") from error
