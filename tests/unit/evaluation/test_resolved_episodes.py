"""Frozen episode resolution and replay for paired controller experiments."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from trailer_docking.evaluation.episodes import (
    ResolvedEpisode,
    resolve_episode,
    resolve_episodes,
)
from trailer_docking.evaluation.scenarios import DomainRandomization, Scenario
from trailer_docking.training.environment import ScenarioDockingEnv


def test_resolution_is_deterministic_and_seeded() -> None:
    scenario = Scenario()
    first = resolve_episode(scenario, 17)
    repeated = resolve_episode(scenario, 17)
    other = resolve_episode(scenario, 18)

    assert first == repeated
    assert first.to_dict() == repeated.to_dict()
    assert first.episode_fingerprint == repeated.episode_fingerprint
    assert first.episode_fingerprint != other.episode_fingerprint
    assert first.scenario_fingerprint == scenario.fingerprint
    assert len(first.episode_fingerprint) == 64
    assert ResolvedEpisode.from_dict(first.to_dict()) == first

    changed = first.to_dict()
    changed["trailer_yaw"] += 0.01
    with pytest.raises(ValueError, match="fingerprint"):
        ResolvedEpisode.from_dict(changed)


def test_resolution_freezes_geometry_before_start_and_path() -> None:
    scenario = Scenario(
        randomization=DomainRandomization(trailer_hitch_to_axle=(3.0, 4.0))
    )
    episodes = resolve_episodes(scenario, episodes=3, seed=4)

    assert len({item.episode_fingerprint for item in episodes}) == 3
    assert all(3.0 <= item.trailer_hitch_to_axle <= 4.0 for item in episodes)
    assert all(len(item.reference_path.points) >= 2 for item in episodes)


def test_resolution_freezes_a_sampled_predefined_curve() -> None:
    scenario = Scenario(
        randomization=DomainRandomization(
            reference_planners=("curve_gentle", "curve_early", "curve_late")
        )
    )
    episodes = resolve_episodes(scenario, episodes=12, seed=4)

    assert {item.reference_planner for item in episodes} <= {
        "curve_gentle",
        "curve_early",
        "curve_late",
    }
    assert len({item.reference_planner for item in episodes}) > 1
    env = ScenarioDockingEnv(scenario)
    try:
        for episode in episodes:
            _, info = env.reset(
                seed=episode.seed, options={"resolved_episode": episode}
            )
            assert env.reference_path == episode.reference_path
            assert info["planner"] == episode.reference_planner
    finally:
        env.close()


def test_exact_episode_replays_without_resampling() -> None:
    scenario = Scenario(
        randomization=DomainRandomization(trailer_hitch_to_axle=(3.0, 4.0))
    )
    episode = resolve_episode(scenario, 9)
    env = ScenarioDockingEnv(scenario, max_episode_steps=3)
    try:
        first, info = env.reset(
            seed=episode.seed, options={"resolved_episode": episode}
        )
        first_state = env.state
        first_path = env.reference_path
        env.step(np.zeros((1,), dtype=np.float32))
        *_, step_info = env.step(np.zeros((1,), dtype=np.float32))
        repeated, repeated_info = env.reset(
            seed=episode.seed, options={"resolved_episode": episode}
        )

        assert np.array_equal(first, repeated)
        assert env.state == first_state
        assert env.reference_path == first_path == episode.reference_path
        assert env.built.vehicle.trailer.hitch_to_axle == pytest.approx(
            episode.trailer_hitch_to_axle
        )
        assert info["episode_fingerprint"] == episode.episode_fingerprint
        assert step_info["episode_fingerprint"] == episode.episode_fingerprint
        assert repeated_info["episode_seed"] == episode.seed
    finally:
        env.close()


def test_replay_rejects_scenario_or_seed_mismatch() -> None:
    scenario = Scenario()
    episode = resolve_episode(scenario, 3)
    env = ScenarioDockingEnv(scenario)
    try:
        with pytest.raises(ValueError, match="seed does not match"):
            env.reset(seed=4, options={"resolved_episode": episode})
        changed = replace(episode, scenario_fingerprint="0" * 64)
        with pytest.raises(ValueError, match="different scenario"):
            env.reset(seed=3, options={"resolved_episode": changed})
        with pytest.raises(ValueError, match="must be ResolvedEpisode"):
            env.reset(options={"resolved_episode": {}})
    finally:
        env.close()
