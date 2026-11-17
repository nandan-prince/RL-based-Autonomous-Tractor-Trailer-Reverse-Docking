"""Swappable route planners and the reward for staying on the planned route."""

from __future__ import annotations

import itertools
from math import hypot

import numpy as np
import pytest

from trailer_docking.core.config import ConfigurationError
from trailer_docking.core.state import Point2D
from trailer_docking.environments import TrailerDockingEnv
from trailer_docking.evaluation.scenarios import (
    Scenario,
    ScenarioOverrides,
    StartDistribution,
    assembled_start,
)
from trailer_docking.paths.planners import (
    DEFAULT_PLANNER,
    PLANNER_LABELS,
    PLANNERS,
    WALL_CLEARANCE,
    _grown_walls,
    _segment_clear,
    plan_reference,
    validate_planner,
)
from trailer_docking.training.environment import ScenarioDockingEnv

BUILT = Scenario().build()
GEOMETRY = BUILT.vehicle.geometry
BAY = BUILT.dock.geometry


def _route(planner: str, x: float, y: float, yaw: float = 0.0) -> tuple[Point2D, ...]:
    pose, trailer_yaw = assembled_start(GEOMETRY, Point2D(x, y), yaw, 0.0)
    state = GEOMETRY.assembled_state(pose, trailer_yaw)
    return plan_reference(planner, GEOMETRY, BAY, state).points


def _length(points: tuple[Point2D, ...]) -> float:
    return sum(hypot(b.x - a.x, b.y - a.y) for a, b in itertools.pairwise(points))


def test_registry() -> None:
    assert DEFAULT_PLANNER == "entry"
    assert set(PLANNERS) == set(PLANNER_LABELS) == {"entry", "astar"}
    assert validate_planner("astar") == "astar"
    with pytest.raises(ValueError, match="planner must be one of"):
        validate_planner("rrt")
    with pytest.raises(ValueError):
        TrailerDockingEnv(planner="rrt")
    with pytest.raises(ConfigurationError):
        Scenario(planner="rrt").build()


@pytest.mark.parametrize("planner", sorted(PLANNERS))
@pytest.mark.parametrize(
    "start", [(10.0, 0.0), (10.0, 1.0), (13.0, -2.5), (9.0, 4.0), (6.5, 0.1)]
)
def test_every_planner_goes_from_the_goal_to_the_start(
    planner: str, start: tuple[float, float]
) -> None:
    points = _route(planner, *start)
    assert points[0] == BAY.target_position  # ordered goal to start
    assert points[-1].x == pytest.approx(start[0])
    assert points[-1].y == pytest.approx(start[1])
    # No route is shorter than the straight line or absurdly longer.
    direct = hypot(start[0] - points[0].x, start[1] - points[0].y)
    assert direct - 1e-9 <= _length(points) <= direct * 1.5 + 1.0


def test_astar_route_keeps_the_trailer_body_clear_of_the_walls() -> None:
    clearance = GEOMETRY.trailer.body_width / 2.0 + WALL_CLEARANCE
    grown = _grown_walls(BAY, clearance)
    # A start beside the bay: the straight line to the goal crosses a wall.
    points = _route("astar", 5.0, 3.0)
    assert not _segment_clear(Point2D(5.0, 3.0), BAY.target_position, grown)
    outside = points[1:]  # everything but the final straight leg into the bay
    for first, second in itertools.pairwise(outside):
        assert _segment_clear(first, second, grown)
    # The final leg runs straight along the bay centreline.
    assert points[1].y == pytest.approx(0.0) and points[1].x > BAY.interior_depth
    # The default planner has no such guarantee from there.
    assert len(_route("entry", 5.0, 3.0)) == 2


def test_astar_matches_the_default_planner_on_an_aligned_start() -> None:
    default = _route("entry", 10.0, 0.0)
    searched = _route("astar", 10.0, 0.0)
    assert _length(searched) == pytest.approx(_length(default))
    assert all(point.y == pytest.approx(0.0) for point in searched)


def test_planner_is_part_of_the_scenario() -> None:
    scenario = Scenario(planner="astar")
    assert scenario.to_dict()["planner"] == "astar"
    assert Scenario.from_dict(scenario.to_dict()) == scenario
    assert Scenario.from_dict({}).planner == "entry"
    assert scenario.fingerprint != Scenario().fingerprint

    start = StartDistribution((5, 5), (1.2, 1.2), (0, 0), (0, 0))
    routes = {}
    for planner in ("entry", "astar"):
        env = ScenarioDockingEnv(Scenario(start=start, planner=planner))
        _, info = env.reset(seed=0)
        assert info["planner"] == planner == env.task_env.planner
        routes[planner] = env.reference_path.points
        # The task tracks the planned route: the start is on it.
        assert abs(env.task_env.tracking.lateral_error) < 1e-6
        env.close()
    assert routes["entry"] != routes["astar"]


def test_reward_penalizes_leaving_the_planned_route() -> None:
    """The lateral and heading terms are the reward for staying on the route."""

    start = StartDistribution((4, 4), (0, 0), (0, 0), (0, 0))
    steer = np.array([1.0], np.float32)

    def penalties(scenario: Scenario) -> tuple[float, float, float]:
        env = ScenarioDockingEnv(scenario)
        env.reset(seed=0)
        lateral = heading = 0.0
        for _ in range(120):
            _, _, terminated, truncated, info = env.step(steer)
            lateral += info["reward_terms"]["lateral"]
            heading += info["reward_terms"]["heading"]
            if terminated or truncated:
                break
        error = abs(env.task_env.tracking.lateral_error)
        env.close()
        return lateral, heading, error

    base_lateral, base_heading, drift = penalties(Scenario(start=start))
    assert drift > 0.05  # steering hard does leave the route
    assert base_lateral < 0.0 and base_heading < 0.0

    # On the route there is nothing to pay.
    env = ScenarioDockingEnv(Scenario(start=start))
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.zeros(1, np.float32))
    assert info["reward_terms"]["lateral"] == pytest.approx(0.0, abs=1e-12)
    assert info["reward_terms"]["heading"] == pytest.approx(0.0, abs=1e-12)
    env.close()

    # The two weights scale exactly those terms and nothing else.
    weighted = Scenario(
        start=start,
        overrides=ScenarioOverrides(path_lateral_weight=5.0, path_heading_weight=0.4),
    )
    built = weighted.build()
    assert built.task.reward_lateral_error == 5.0
    assert built.task.reward_heading_error == 0.4
    assert built.task.reward_progress == BUILT.task.reward_progress
    lateral, heading, _ = penalties(weighted)
    assert lateral == pytest.approx(base_lateral * 5.0 / 0.5)
    assert heading == pytest.approx(base_heading * 0.4 / 0.2)
    assert weighted.fingerprint != Scenario(start=start).fingerprint
    for bad in (-1.0, 0.0, float("nan")):
        with pytest.raises(ValueError):
            Scenario(overrides=ScenarioOverrides(path_lateral_weight=bad)).build()


def test_route_is_drawn_in_3d_frames_only_while_switched_on(
    offscreen_rendering: None,
) -> None:
    pytest.importorskip("mujoco")
    from trailer_docking.training.evaluate import (
        run_docking_test,
        straight_reverse_policy,
    )
    from trailer_docking.visualization.render3d import display

    scenario = Scenario(start=StartDistribution((4, 4), (1, 1), (0, 0), (0, 0)))

    def first_frame() -> np.ndarray:
        frames: list[np.ndarray] = []
        run_docking_test(
            scenario,
            straight_reverse_policy,
            backend="3d",
            max_episode_steps=1,
            on_frame=frames.append,
        )
        return frames[0]

    def green(frame: np.ndarray) -> int:
        red, grn, blue = (frame[..., i].astype(int) for i in range(3))
        return int(((grn > 120) & (grn > red + 60) & (grn > blue + 60)).sum())

    assert display.show_route is True
    try:
        shown = first_frame()
        display.show_route = False
        hidden = first_frame()
    finally:
        display.show_route = True
    assert green(shown) > green(hidden) + 50
    assert green(hidden) < 20
