"""PPO training sessions and docking tests for the reverse-docking environment.

This package is a framework skeleton: it trains and tests policies end to end
on the verified 2D Gymnasium task and records provenance.  The reward weights
and start distribution are untuned, so nothing it produces is a research
result until a run is registered as an experiment.
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
