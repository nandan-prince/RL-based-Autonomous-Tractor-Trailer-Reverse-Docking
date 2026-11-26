"""PPO training sessions and docking tests for the reverse-docking environment.

This package trains and tests policies end to end on the verified Gymnasium
task and records provenance.  The reward weights
and start distribution are untuned, so nothing it produces is a research
result until a run is registered as an experiment.
"""

from trailer_docking.training.comparison import (
    ComparisonEvidence,
    ComparisonMethod,
    ComparisonRun,
    run_controlled_comparison,
    save_comparison_evidence,
)
from trailer_docking.training.config import TrainingConfig
from trailer_docking.training.curriculum import (
    CurriculumConfiguration,
    CurriculumResult,
    CurriculumRunner,
    CurriculumStage,
)
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
from trailer_docking.training.study import (
    PPORewardStudyConfiguration,
    PPORewardStudyResult,
    PPORewardStudyRunner,
    RewardCondition,
)

__all__ = [
    "ComparisonEvidence",
    "ComparisonMethod",
    "ComparisonRun",
    "CurriculumConfiguration",
    "CurriculumResult",
    "CurriculumRunner",
    "CurriculumStage",
    "EpisodeRecord",
    "EpisodeResult",
    "PPORewardStudyConfiguration",
    "PPORewardStudyResult",
    "PPORewardStudyRunner",
    "Policy",
    "RewardCondition",
    "TrainingConfig",
    "TrainingProgress",
    "TrainingSession",
    "load_policy",
    "new_experiment_id",
    "run_controlled_comparison",
    "run_docking_test",
    "save_comparison_evidence",
    "straight_reverse_policy",
    "summarize",
]
