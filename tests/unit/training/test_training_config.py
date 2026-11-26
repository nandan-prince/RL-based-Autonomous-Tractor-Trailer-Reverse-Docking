from dataclasses import FrozenInstanceError
from datetime import datetime

import pytest

from trailer_docking.training.config import TrainingConfig
from trailer_docking.training.session import (
    TrainingProgress,
    TrainingSession,
    new_experiment_id,
)


def test_defaults_match_the_contract() -> None:
    config = TrainingConfig()
    assert config.to_dict() == {
        "total_timesteps": 200_000,
        "seed": 0,
        "learning_rate": 3e-4,
        "n_steps": 1024,
        "batch_size": 256,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "clip_range_vf": None,
        "normalize_advantage": True,
        "entropy_coefficient": 0.0,
        "value_coefficient": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": None,
        "device": "cpu",
        "max_episode_steps": 900,
        "checkpoint_interval": 20_000,
        "purpose": "docking",
        "runs": 1,
        "backend": "2d",
        "evaluation_episodes": 20,
        "evaluation_backends": ["2d"],
        "evaluation_seed": 100_000,
        "method": "ppo",
    }
    assert TrainingConfig.from_dict(config.to_dict()) == config
    with pytest.raises(FrozenInstanceError):
        config.seed = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    "field",
    [
        "total_timesteps",
        "n_steps",
        "batch_size",
        "n_epochs",
        "max_episode_steps",
        "checkpoint_interval",
        "runs",
    ],
)
@pytest.mark.parametrize("value", [0, -5])
def test_counts_must_be_positive(field: str, value: int) -> None:
    with pytest.raises(ValueError, match=field):
        TrainingConfig(**{field: value})


@pytest.mark.parametrize("rate", [0.0, -1e-3, float("nan"), float("inf")])
def test_learning_rate_must_be_positive_and_finite(rate: float) -> None:
    with pytest.raises(ValueError, match="learning_rate"):
        TrainingConfig(learning_rate=rate)


@pytest.mark.parametrize("field", ["gamma", "gae_lambda"])
@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_discount_parameters_are_probabilities(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        TrainingConfig(**{field: value})


@pytest.mark.parametrize("field", ["clip_range", "max_grad_norm"])
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan")])
def test_positive_ppo_parameters(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        TrainingConfig(**{field: value})


def test_optional_ppo_parameters_and_device_are_validated() -> None:
    with pytest.raises(ValueError, match="target_kl"):
        TrainingConfig(target_kl=0.0)
    with pytest.raises(ValueError, match="clip_range_vf"):
        TrainingConfig(clip_range_vf=-0.1)
    with pytest.raises(ValueError, match="entropy_coefficient"):
        TrainingConfig(entropy_coefficient=-0.1)
    with pytest.raises(ValueError, match="normalize_advantage"):
        TrainingConfig(normalize_advantage=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="device"):
        TrainingConfig(device="quantum")


def test_seed_and_purpose_are_validated() -> None:
    with pytest.raises(ValueError, match="seed"):
        TrainingConfig(seed=-1)
    with pytest.raises(ValueError, match="purpose"):
        TrainingConfig(purpose="  ")


def test_new_experiment_id_format() -> None:
    now = datetime(2026, 3, 9, 7, 5)
    assert new_experiment_id("docking", now) == "20260309-0705_ppo_docking"
    assert (
        new_experiment_id("Wide Bay / seed_3!", now)
        == "20260309-0705_ppo_wide-bay-seed-3"
    )
    assert new_experiment_id("???", now) == "20260309-0705_ppo_run"
    assert new_experiment_id("docking").endswith("_ppo_docking")


def test_idle_session_needs_no_training_libraries(tmp_path) -> None:
    from trailer_docking.evaluation.scenarios import Scenario

    session = TrainingSession(Scenario(), output_root=tmp_path)
    session.stop()
    session.join(0.1)
    assert session.progress == TrainingProgress(
        status="idle",
        timesteps=0,
        total_timesteps=200_000,
        episodes=0,
        recent_success_rate=None,
        recent_mean_return=None,
        last_outcome=None,
        error=None,
        experiment_id=None,
        checkpoint_path=None,
    )
    assert session.spectator_state is None
    assert session.episode_history == ()
    assert list(tmp_path.iterdir()) == []
