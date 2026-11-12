"""PPO training sessions and docking tests for the reverse-docking environment.

This package is a framework skeleton: it trains and tests policies end to end
and records provenance, but the environment reward is still a placeholder, so
nothing it produces is a research result yet.
"""

from trailer_docking.training.config import TrainingConfig
from trailer_docking.training.evaluate import (
    EpisodeResult,
    Policy,
    load_policy,
    run_docking_test,
    straight_reverse_policy,
    summarize,
)
from trailer_docking.training.session import (
    EpisodeRecord,
    TrainingProgress,
    TrainingSession,
    new_experiment_id,
)

__all__ = [
    "EpisodeRecord",
    "EpisodeResult",
    "Policy",
    "TrainingConfig",
    "TrainingProgress",
    "TrainingSession",
    "load_policy",
    "new_experiment_id",
    "run_docking_test",
    "straight_reverse_policy",
    "summarize",
]
