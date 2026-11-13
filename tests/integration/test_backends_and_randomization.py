"""The scenario environment on both simulators, with domain randomization."""

from __future__ import annotations

import numpy as np
import pytest

from trailer_docking.environments import TrailerDockingEnv
from trailer_docking.evaluation.scenarios import (
    DomainRandomization,
    Scenario,
    ScenarioOverrides,
    StartDistribution,
)
from trailer_docking.simulators.base import SimulatorBackend
from trailer_docking.training.environment import ScenarioDockingEnv
from trailer_docking.training.evaluate import (
    run_docking_test,
    straight_reverse_policy,
)

pytest.importorskip("mujoco")

ALIGNED = StartDistribution((4, 4), (0, 0), (0, 0), (0, 0))
ZERO = np.zeros(1, np.float32)


def _rollout(env: ScenarioDockingEnv, seed: int = 0) -> tuple[int, dict]:
    env.reset(seed=seed)
    steps = 0
    while True:
        _, _, terminated, truncated, info = env.step(ZERO)
        steps += 1
        if terminated or truncated:
            return steps, info


def test_task_environment_backend_selection() -> None:
    assert TrailerDockingEnv().backend == "2d"
    env = TrailerDockingEnv(backend="3d")
    assert isinstance(env.simulator, SimulatorBackend)
    assert type(env.simulator).__name__ == "MujocoSimulator"
    _, info = env.reset(seed=0)
    assert info["simulator_backend"] == "3d"
    assert env.observation_space.shape == TrailerDockingEnv().observation_space.shape
    env.close()
    with pytest.raises(ValueError):
        TrailerDockingEnv(backend="4d")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ScenarioDockingEnv(backend="mujoco")


@pytest.mark.parametrize("backend", ["2d", "3d"])
def test_straight_reverse_docks_on_both_backends(backend: str) -> None:
    env = ScenarioDockingEnv(Scenario(start=ALIGNED), backend=backend)
    steps, info = _rollout(env)
    env.close()
    assert info["termination_reason"] == "success"
    assert info["simulator_backend"] == backend
    assert not info["collision"]
    # 9.75 m at 0.2 m/s, plus acceleration: the same task in both simulators.
    assert 480 <= steps <= 495
    assert abs(info["docking_error_lateral"]) < 1e-3


@pytest.mark.parametrize("backend", ["2d", "3d"])
def test_custom_truck_runs_on_both_backends(backend: str) -> None:
    scenario = Scenario(
        overrides=ScenarioOverrides(
            tractor_wheelbase=2.4,
            trailer_hitch_to_axle=3.2,
            trailer_body_width=1.2,
            dock_interior_width=1.9,
            reverse_speed=0.15,
        ),
        start=ALIGNED,
    )
    env = ScenarioDockingEnv(scenario, backend=backend, max_episode_steps=900)
    steps, info = _rollout(env)
    assert info["termination_reason"] == "success"
    assert env.state.longitudinal_speed == pytest.approx(-0.15, abs=0.02)
    assert steps > 600  # slower than the 0.2 m/s baseline
    env.close()


def test_physics_overrides_reach_the_3d_model() -> None:
    scenario = Scenario(
        overrides=ScenarioOverrides(tractor_chassis_mass=1234.0, ground_friction=0.6)
    )
    env = ScenarioDockingEnv(scenario, backend="3d")
    xml = env.task_env.simulator.xml  # type: ignore[attr-defined]
    assert "1234" in xml
    assert env.built.vehicle.mujoco.contact_friction[0] == pytest.approx(0.6)
    env.close()


@pytest.mark.parametrize("backend", ["2d", "3d"])
def test_trailer_length_is_randomized_per_episode(backend: str) -> None:
    scenario = Scenario(
        start=ALIGNED,
        randomization=DomainRandomization(trailer_hitch_to_axle=(3.0, 4.2)),
    )
    env = ScenarioDockingEnv(scenario, backend=backend, max_episode_steps=5)
    lengths = []
    for seed in (1, 2, 3):
        _, info = env.reset(seed=seed)
        lengths.append(info["trailer_hitch_to_axle"])
        assert env.built.vehicle.trailer.hitch_to_axle == lengths[-1]
        assert env.task_env.vehicle.trailer.hitch_to_axle == lengths[-1]
        # The simulated trailer really has that length.
        state = env.state
        assert state.hitch_position.x - state.trailer_pose.x == pytest.approx(
            lengths[-1], abs=1e-6
        )
        _, _, _, _, step_info = env.step(ZERO)
        assert step_info["trailer_hitch_to_axle"] == lengths[-1]
        assert step_info["scenario_fingerprint"] == scenario.fingerprint
    assert len(set(lengths)) == 3
    assert all(3.0 <= length <= 4.2 for length in lengths)
    # Seeded resets reproduce the draw; unseeded resets continue the sequence.
    _, info = env.reset(seed=1)
    assert info["trailer_hitch_to_axle"] == lengths[0]
    env.close()


def test_observation_noise_is_seeded_bounded_and_off_by_default() -> None:
    clean_env = ScenarioDockingEnv(Scenario(start=ALIGNED))
    clean, _ = clean_env.reset(seed=3)
    again, _ = clean_env.reset(seed=3)
    np.testing.assert_array_equal(clean, again)

    noisy_scenario = Scenario(
        start=ALIGNED, randomization=DomainRandomization(observation_noise_std=0.05)
    )
    env = ScenarioDockingEnv(noisy_scenario)
    first, _ = env.reset(seed=3)
    stepped, *_ = env.step(ZERO)
    second, _ = env.reset(seed=3)
    np.testing.assert_array_equal(first, second)
    assert first.dtype == np.float32
    assert not np.array_equal(first, clean)
    assert np.all(np.abs(first) <= 1.0) and np.all(np.abs(stepped) <= 1.0)
    assert env.observation_space.contains(first)
    # The noise changes what the policy sees, never the simulated state.
    assert env.state == clean_env.state
    clean_env.close()
    env.close()


def test_3d_frames_are_delivered_only_for_the_3d_backend(
    offscreen_rendering: None,
) -> None:
    frames: list[np.ndarray] = []
    results = run_docking_test(
        Scenario(start=ALIGNED),
        straight_reverse_policy,
        backend="3d",
        max_episode_steps=7,
        on_frame=frames.append,
    )
    assert results[0].outcome == "time_limit"
    # One frame at reset, one every third step, one at the end.
    assert len(frames) == 4
    assert frames[0].shape == (480, 640, 3) and frames[0].dtype == np.uint8
    assert frames[0].std() > 1.0  # an actual picture, not a blank buffer

    flat: list[np.ndarray] = []
    run_docking_test(
        Scenario(start=ALIGNED),
        straight_reverse_policy,
        max_episode_steps=7,
        on_frame=flat.append,
    )
    assert flat == []
