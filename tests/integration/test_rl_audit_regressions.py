"""Regression tests for defects found by the audit of the RL/UI skeleton."""

from __future__ import annotations

from pathlib import Path

import pytest

from trailer_docking.core.config import ConfigurationError
from trailer_docking.core.state import Pose2D
from trailer_docking.evaluation.scenarios import Scenario, StartDistribution
from trailer_docking.paths.reference import ReferencePath
from trailer_docking.paths.tracking import closest_arc_length

np = pytest.importorskip("numpy")
pytest.importorskip("gymnasium")
pytest.importorskip("mujoco")

from trailer_docking.environments.docking_env import DockingEnv  # noqa: E402
from trailer_docking.training import cli  # noqa: E402


def _hairpin() -> ReferencePath:
    """Two parallel legs 1 m apart joined by a tight turn."""

    poses = [Pose2D(10.0 - 0.1 * i, 0.0, 0.0) for i in range(101)]
    poses += [Pose2D(0.0, 0.5, -1.5708)]
    poses += [Pose2D(0.1 * i, 1.0, 3.14159) for i in range(101)]
    return ReferencePath.from_poses(poses)


def test_projection_window_prevents_jumping_to_a_far_part_of_the_path() -> None:
    path = _hairpin()
    # A point between the legs, slightly nearer the second one.
    pose = Pose2D(5.0, 0.55, 0.0)
    unconstrained = closest_arc_length(path, pose)
    assert unconstrained > path.length / 2.0
    # Tracking continuously along the first leg must stay on the first leg.
    tracked = closest_arc_length(path, pose, previous=4.9)
    assert tracked == pytest.approx(5.0, abs=0.05)
    assert closest_arc_length(path, pose, previous=-5.0) < 1.5
    assert closest_arc_length(path, pose, previous=1e9) > path.length - 0.3


def test_path_progress_per_step_is_bounded_by_distance_travelled() -> None:
    wide = Scenario(
        start=StartDistribution(
            standoff=(3, 9),
            lateral=(-3, 3),
            trailer_yaw=(-1.2, 1.2),
            articulation=(-0.5, 0.5),
        )
    )
    env = DockingEnv(wide, max_episode_steps=400)
    for seed in (4, 31, 98):
        _, info = env.reset(seed=seed)
        previous = info["path_progress"]
        for _ in range(400):
            _, reward, terminated, truncated, info = env.step(np.zeros(1, np.float32))
            step = info["path_progress"] - previous
            previous = info["path_progress"]
            # 0.2 m/s for 0.1 s is 0.02 m; the projection window bounds the rest.
            assert abs(step) < 0.3
            if terminated or truncated:
                break
            assert reward < 0.5


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


def test_environment_reports_each_planned_route() -> None:
    routes: list[ReferencePath] = []
    env = DockingEnv(on_route=routes.append)
    env.reset(seed=1)
    env.reset(seed=2)
    assert len(routes) == 2 and routes[-1] is env.reference_path
    assert routes[0].poses[0] != routes[1].poses[0]


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
