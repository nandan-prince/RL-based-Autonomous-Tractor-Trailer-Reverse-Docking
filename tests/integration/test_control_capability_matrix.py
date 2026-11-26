"""Executable controller/backend capability matrix used by the control UI."""

from __future__ import annotations

from pathlib import Path

import pytest

from trailer_docking.evaluation.scenarios import Scenario, StartDistribution
from trailer_docking.training.config import TrainingConfig
from trailer_docking.training.evaluate import (
    run_docking_test,
    straight_reverse_policy,
)
from trailer_docking.training.methods import (
    lqr_controller_factory,
    nmpc_controller_factory,
)
from trailer_docking.training.session import STATUS_FINISHED, TrainingSession

pytest.importorskip("mujoco")

CAPABILITY_SCENARIO = Scenario(
    name="control-capability-smoke",
    start=StartDistribution((4, 4), (0.2, 0.2), (0, 0), (0, 0)),
)


@pytest.mark.parametrize("backend", ["2d", "3d"])
@pytest.mark.parametrize(
    ("method", "policy", "factory", "controller_name"),
    [
        ("ppo", straight_reverse_policy, None, "ppo-policy"),
        ("lqr", None, lqr_controller_factory(), "trajectory-lqr-v1"),
        ("nmpc", None, nmpc_controller_factory(), "basic-nmpc-v1"),
    ],
)
def test_every_evaluation_method_executes_on_every_backend(
    backend: str, method: str, policy, factory, controller_name: str
) -> None:
    """Execution is the capability claim; this smoke test is not performance."""

    results = run_docking_test(
        CAPABILITY_SCENARIO,
        policy,
        episodes=1,
        seed=7,
        backend=backend,
        max_episode_steps=3,
        controller_factory=factory,
    )

    assert len(results) == 1
    assert results[0].steps == 3
    assert results[0].controller == controller_name
    assert results[0].solve_failures == 0


@pytest.mark.parametrize("backend", ["2d", "3d"])
def test_ppo_training_executes_on_every_backend(
    backend: str, tmp_path: Path
) -> None:
    pytest.importorskip("stable_baselines3")
    scenario = Scenario(
        name=f"ppo-{backend}-capability",
        start=StartDistribution((4, 4), (0, 0), (0, 0), (0, 0)),
    )
    config = TrainingConfig(
        total_timesteps=64,
        n_steps=64,
        batch_size=32,
        n_epochs=1,
        max_episode_steps=10,
        checkpoint_interval=64,
        evaluation_episodes=0,
        backend=backend,
        purpose=f"{backend}-capability-smoke",
    )
    session = TrainingSession(scenario, config, output_root=tmp_path)

    session.start()
    session.join(120)

    assert session.progress.status == STATUS_FINISHED
    assert session.progress.timesteps == 64
    assert session.progress.checkpoint_path is not None
    assert session.progress.checkpoint_path.is_file()
