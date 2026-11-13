"""Static obstacles in the 2D task, the MuJoCo scene, the planner, and the UI."""

from __future__ import annotations

from itertools import pairwise
from math import pi
from pathlib import Path

import numpy as np
import pytest

from trailer_docking.core.config import ConfigurationError
from trailer_docking.core.obstacles import Obstacle, ObstacleError, obstacle_name
from trailer_docking.environments import TrailerDockingEnv
from trailer_docking.evaluation.scenarios import Scenario, StartDistribution
from trailer_docking.paths.planners import WALL_CLEARANCE, _grown_walls, _segment_clear
from trailer_docking.safety.collision import polygons_intersect
from trailer_docking.training.environment import ScenarioDockingEnv
from trailer_docking.ui.controller import (
    AppController,
    ParameterError,
    format_obstacles,
    parse_obstacles,
)

pytest.importorskip("mujoco")

AHEAD = StartDistribution((6, 6), (0, 0), (0, 0), (0, 0))
BLOCK = Obstacle(9.0, 0.0, 1.0, 1.0)
ZERO = np.zeros(1, np.float32)


def _run(env: ScenarioDockingEnv) -> tuple[int, dict]:
    env.reset(seed=0)
    steps = 0
    while True:
        _, _, terminated, truncated, info = env.step(ZERO)
        steps += 1
        if terminated or truncated:
            return steps, info


def test_obstacle_geometry_and_validation() -> None:
    box = Obstacle(2.0, 1.0, 4.0, 2.0)
    assert [(p.x, p.y) for p in box.polygon()] == [
        (0.0, 0.0),
        (4.0, 0.0),
        (4.0, 2.0),
        (0.0, 2.0),
    ]
    turned = Obstacle(0.0, 0.0, 4.0, 2.0, yaw=pi / 2)
    xs = sorted(round(p.x, 9) for p in turned.polygon())
    assert xs == [-1.0, -1.0, 1.0, 1.0]
    assert Obstacle.from_dict(box.to_dict()) == box
    assert obstacle_name(0) == "obstacle_1"
    for bad in (
        {"length": 0.0},
        {"width": -1.0},
        {"height": 0.0},
        {"x": float("nan")},
        {"yaw": float("inf")},
        {"y": True},
    ):
        with pytest.raises(ObstacleError):
            Obstacle(**{"x": 1.0, "y": 1.0, "length": 1.0, "width": 1.0, **bad})
    with pytest.raises(ObstacleError):
        Obstacle.from_dict({"x": 1})
    with pytest.raises(ObstacleError):
        Obstacle.from_dict([1, 2, 3, 4])


@pytest.mark.parametrize("backend", ["2d", "3d"])
def test_hitting_an_obstacle_is_a_collision_in_both_simulators(backend: str) -> None:
    env = ScenarioDockingEnv(Scenario(start=AHEAD, obstacles=(BLOCK,)), backend=backend)
    steps, info = _run(env)
    assert info["termination_reason"] == "collision"
    assert info["collision_contacts"] == ("trailer:obstacle_1",)
    # The trailer's rear face starts 2.5 m from the box face: about 125 steps.
    assert 120 <= steps <= 135
    if backend == "3d":
        # The box is a real body in the MuJoCo scene and is being touched.
        simulator = env.task_env.simulator
        assert simulator.wall_contacts() == ("trailer_body",)  # type: ignore[attr-defined]
        assert 'name="obstacle_1"' in simulator.xml  # type: ignore[attr-defined]
    env.close()


@pytest.mark.parametrize("backend", ["2d", "3d"])
def test_an_obstacle_off_the_route_changes_nothing(backend: str) -> None:
    start = StartDistribution((4, 4), (0, 0), (0, 0), (0, 0))
    aside = Obstacle(9.0, 3.0, 1.0, 1.0, yaw=0.4, height=0.5)
    clear = ScenarioDockingEnv(Scenario(start=start), backend=backend)
    cluttered = ScenarioDockingEnv(
        Scenario(start=start, obstacles=(aside,)), backend=backend
    )
    assert _run(clear)[0] == _run(cluttered)[0]
    assert cluttered.state == clear.state
    clear.close()
    cluttered.close()


def test_default_task_has_no_obstacles_and_an_unchanged_scene() -> None:
    env = TrailerDockingEnv(backend="3d")
    assert env.obstacles == ()
    assert "obstacle" not in env.simulator.xml  # type: ignore[attr-defined]
    env.close()


def test_astar_routes_around_an_obstacle_and_entry_does_not() -> None:
    routes = {}
    for planner in ("entry", "astar"):
        env = ScenarioDockingEnv(
            Scenario(start=AHEAD, obstacles=(BLOCK,), planner=planner)
        )
        env.reset(seed=0)
        routes[planner] = env.reference_path.points
        env.close()
    box = BLOCK.polygon()

    def crosses(points: tuple) -> bool:
        return any(
            polygons_intersect((a, b, type(a)(b.x, b.y + 1e-6)), box)
            for a, b in pairwise(points)
        )

    assert crosses(routes["entry"])  # the default route ignores obstacles
    assert not crosses(routes["astar"])
    # Away from its start, the searched route keeps the trailer's half width
    # plus the clearance from the box.
    built = Scenario().build()
    clearance = built.vehicle.trailer.body_width / 2.0 + WALL_CLEARANCE
    grown = _grown_walls(built.dock.geometry, clearance, (BLOCK,))
    middle = routes["astar"][1:-1]
    assert len(middle) >= 3
    assert all(_segment_clear(a, b, grown) for a, b in pairwise(middle))


def test_scenarios_store_validate_and_fingerprint_obstacles(tmp_path: Path) -> None:
    scenario = Scenario(start=AHEAD, obstacles=(BLOCK, Obstacle(11.0, -3.0, 2.0, 0.5)))
    assert scenario.build().obstacles == scenario.obstacles
    loaded = Scenario.load(scenario.save(tmp_path / "s.yaml"))
    assert loaded == scenario and loaded.fingerprint == scenario.fingerprint
    assert scenario.fingerprint != Scenario(start=AHEAD).fingerprint
    assert Scenario.from_dict({}).obstacles == ()
    with pytest.raises(ConfigurationError):
        Scenario(obstacles=("box",))  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError):
        Scenario.from_dict({"obstacles": [{"x": 1}]})
    # An obstacle where the docked vehicle stands, or on a start pose, makes
    # the scenario impossible and is refused.
    with pytest.raises(ConfigurationError, match="docked vehicle would overlap"):
        Scenario(obstacles=(Obstacle(3.0, 0.0, 1.0, 1.0),)).build()
    with pytest.raises(ConfigurationError, match="begin already failed"):
        Scenario(start=AHEAD, obstacles=(Obstacle(13.0, 0.0, 1.0, 1.0),)).build()


def test_obstacles_are_entered_as_text_in_the_ui(tmp_path: Path) -> None:
    assert parse_obstacles("") == () and parse_obstacles(" ; ") == ()
    parsed = parse_obstacles("9, 3, 1, 1; 11, -3, 2, 0.5, 0.3, 2")
    assert parsed == (
        Obstacle(9.0, 3.0, 1.0, 1.0),
        Obstacle(11.0, -3.0, 2.0, 0.5, yaw=0.3, height=2.0),
    )
    assert parse_obstacles(format_obstacles(parsed)) == parsed
    assert format_obstacles(parsed[:1]) == "9, 3, 1, 1"
    for bad in ("1, 2, 3", "1, 2, 3, 4, 5, 6, 7", "a, 2, 3, 4", "1, 2, 0, 4"):
        with pytest.raises(ParameterError, match="Obstacle 1"):
            parse_obstacles(bad)

    controller = AppController(output_root=tmp_path)
    assert controller.parameter_values()["obstacles"] == ""
    controller.apply_parameters({"obstacles": "10, 3, 1, 1; 12, -3, 2, 0.5"})
    assert len(controller.built.obstacles) == 2
    assert controller.parameter_values()["obstacles"] == "10, 3, 1, 1; 12, -3, 2, 0.5"
    before = controller.scenario
    with pytest.raises(ParameterError, match="docked vehicle would overlap"):
        controller.apply_parameters({"obstacles": "3, 0, 1, 1"})
    assert controller.scenario == before
    # Other fields leave the obstacles alone; a blank field removes them.
    controller.apply_parameters({"planner": "astar"})
    assert len(controller.scenario.obstacles) == 2
    try:
        cluttered = controller.preview_frame()
        controller.apply_parameters({"obstacles": ""})
        assert controller.scenario.obstacles == ()
        assert not np.array_equal(controller.preview_frame(), cluttered)
    except Exception as error:  # no OpenGL on this machine
        if "3D preview unavailable" not in str(error):
            raise


def test_top_down_view_draws_obstacles() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from trailer_docking.visualization.topdown import TopDownView

    built = Scenario().build()
    view = TopDownView(built.vehicle, built.dock)
    before = len(view.axes.patches)
    view.set_obstacles((BLOCK, Obstacle(11.0, -3.0, 2.0, 0.5)))
    assert len(view.axes.patches) == before + 2
    view.set_obstacles(())
    assert len(view.axes.patches) == before
    plt.close(view.figure)
