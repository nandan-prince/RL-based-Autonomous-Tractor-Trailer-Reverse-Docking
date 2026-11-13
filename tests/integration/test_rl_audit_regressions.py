"""Regression tests for defects found by the audit of the RL/UI skeleton."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trailer_docking.core.config import ConfigurationError
from trailer_docking.evaluation.scenarios import Scenario, StartDistribution
from trailer_docking.paths.reference import ReferencePath
from trailer_docking.training import cli
from trailer_docking.training.environment import ScenarioDockingEnv


def test_path_progress_per_step_is_bounded_by_distance_travelled() -> None:
    offset = Scenario(
        start=StartDistribution(
            standoff=(3, 6),
            lateral=(-1, 1),
            trailer_yaw=(-0.3, 0.3),
            articulation=(-0.2, 0.2),
        )
    )
    env = ScenarioDockingEnv(offset, max_episode_steps=400)
    for seed in (4, 31, 98):
        _, info = env.reset(seed=seed)
        previous = info["path_progress"]
        for _ in range(400):
            _, _, terminated, truncated, info = env.step(np.zeros(1, np.float32))
            step = info["path_progress"] - previous
            previous = info["path_progress"]
            # 0.2 m/s for 0.1 s is 0.02 m of travel per step.
            assert abs(step) < 0.3
            if terminated or truncated:
                break
    env.close()


@pytest.mark.parametrize(
    "start",
    [
        {"articulation": [2.0, 2.0]},
        {"articulation": [-0.9, 0.0]},
        {"standoff": [float("inf"), float("inf")]},
        {"lateral": [0.0, float("nan")]},
        {"trailer_yaw": [-4.0, 4.0]},
    ],
)
def test_unusable_start_ranges_are_rejected_before_an_episode(start: dict) -> None:
    with pytest.raises(ConfigurationError):
        Scenario.from_dict({"start": start}).build()


def test_environment_reports_each_reference_path() -> None:
    routes: list[ReferencePath] = []
    env = ScenarioDockingEnv(on_route=routes.append)
    env.reset(seed=1)
    env.reset(seed=2)
    assert len(routes) == 2 and routes[-1] is env.reference_path
    # Paths run goal-to-start, so different starts differ at the last point.
    assert routes[0].points[0] == routes[1].points[0]
    assert routes[0].points[-1] != routes[1].points[-1]
    env.close()


def test_seeded_resets_reproduce_the_start_sequence() -> None:
    env = ScenarioDockingEnv()
    env.reset(seed=7)
    first = env.state
    env.reset()
    second = env.state
    env.reset(seed=7)
    assert env.state == first
    env.reset()
    assert env.state == second
    assert first != second
    env.close()


def test_scenario_environment_is_the_fixed_reverse_task() -> None:
    scenario = Scenario()
    env = ScenarioDockingEnv(scenario, max_episode_steps=3)
    assert env.action_space.shape == (1,)
    assert env.task_env.settings.max_episode_steps == 3
    _, info = env.reset(seed=0)
    assert info["task_fingerprint"] == scenario.build().task.source_fingerprint
    for _ in range(3):
        _, _, terminated, truncated, info = env.step(np.zeros(1, np.float32))
    assert truncated and not terminated
    assert info["termination_reason"] == "time_limit"
    with pytest.raises(ValueError):
        ScenarioDockingEnv(max_episode_steps=0)
    env.close()


def test_cli_test_reports_bad_input_as_usage_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["test", "--seed", "-1"]) == 2
    assert "error:" in capsys.readouterr().err
    not_a_zip = tmp_path / "bad.zip"
    not_a_zip.write_text("not a policy", encoding="utf-8")
    assert cli.main(["test", "--policy", str(not_a_zip)]) == 2
    assert "error:" in capsys.readouterr().err
    assert cli.main(["test", "--policy", str(tmp_path / "missing.zip")]) == 2
