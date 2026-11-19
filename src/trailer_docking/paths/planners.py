"""Swappable reference-route planners for the docking task.

A planner turns one start state into the route of the trailer's docking point,
ordered goal to start like every :class:`ReferencePath`.  The task, the
viewers, and the control UI select one by name from :data:`PLANNERS`, so a new
planner only has to be registered here.

``entry``
    The original route: straight along the bay centreline, joined through an
    entry point just outside the opening for an offset start.  It ignores
    obstacles.

``astar``
    A basic path-finding algorithm.  The docking point is planned as a point
    on an occupancy grid with A* search (8-connected, Euclidean heuristic),
    with the dock walls and any obstacles grown by the trailer's half width
    so the body keeps clear of them.  The grid path is shortened to its
    line-of-sight corners.
    The final leg into the bay is always the straight centreline, because the
    bay is narrower than the grown walls allow a grid search to enter.

Neither planner checks that an articulated vehicle can actually follow the
route in reverse: both are geometric.  A feasibility-checked planner
(Reeds-Shepp, Hybrid A*) can be registered under a new name.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable, Sequence
from math import ceil, hypot

from trailer_docking.core.geometry import DockGeometry, VehicleGeometry
from trailer_docking.core.obstacles import Obstacle
from trailer_docking.core.state import Point2D, VehicleState
from trailer_docking.paths.generators import (
    ENTRY_STANDOFF,
    docking_reference,
    manual_reference,
)
from trailer_docking.paths.reference import ReferencePath
from trailer_docking.safety.collision import dock_wall_polygons

Planner = Callable[
    [VehicleGeometry, DockGeometry, VehicleState, Sequence[Obstacle]],
    ReferencePath,
]

DEFAULT_PLANNER = "entry"
GRID_RESOLUTION = 0.25
"""Cell size in metres of the A* occupancy grid."""
WALL_CLEARANCE = 0.15
"""Extra room in metres kept between the trailer body and a wall."""
MARGIN_COST = 8.0
"""Cost multiplier for travelling inside a wall's or obstacle's clearance."""
SEARCH_PADDING = 3.0
"""How far beyond the start and the dock the grid extends, in metres."""

Cell = tuple[int, int]
Box = tuple[float, float, float, float]


class PlanningError(ValueError):
    """Raised when a planner cannot produce a route."""


def _grown_walls(
    dock: DockGeometry, clearance: float, obstacles: Sequence[Obstacle] = ()
) -> tuple[Box, ...]:
    """Axis-aligned bounds of every wall and obstacle, grown by *clearance*.

    A rotated obstacle is covered by its bounding box, which is conservative.
    """

    boxes = []
    polygons = [
        *dock_wall_polygons(dock).values(),
        *(obstacle.polygon() for obstacle in obstacles),
    ]
    for polygon in polygons:
        xs = [point.x for point in polygon]
        ys = [point.y for point in polygon]
        boxes.append(
            (
                min(xs) - clearance,
                max(xs) + clearance,
                min(ys) - clearance,
                max(ys) + clearance,
            )
        )
    return tuple(boxes)


def _blocked(point: Point2D, walls: tuple[Box, ...]) -> bool:
    return any(
        x_min <= point.x <= x_max and y_min <= point.y <= y_max
        for x_min, x_max, y_min, y_max in walls
    )


def _segment_clear(first: Point2D, second: Point2D, walls: tuple[Box, ...]) -> bool:
    """Whether the straight segment stays out of every grown wall."""

    steps = max(ceil(hypot(second.x - first.x, second.y - first.y) / 0.05), 1)
    return not any(
        _blocked(
            Point2D(
                first.x + (second.x - first.x) * index / steps,
                first.y + (second.y - first.y) * index / steps,
            ),
            walls,
        )
        for index in range(steps + 1)
    )


def _search(
    start: Point2D, goal: Point2D, solid: tuple[Box, ...], margin: tuple[Box, ...]
) -> tuple[Point2D, ...]:
    """A* from *start* to *goal*; returns the cell-centre path, start first.

    *solid* boxes are the real walls and obstacles and are never entered.
    *margin* boxes are the same grown by the clearance; a cell inside one can
    be crossed, at ``MARGIN_COST`` times the distance, so a route keeps its
    clearance wherever it can but a start that already stands inside a margin
    can still leave it.
    """

    x_min = min(start.x, goal.x) - SEARCH_PADDING
    y_min = min(start.y, goal.y) - SEARCH_PADDING
    columns = ceil((max(start.x, goal.x) + SEARCH_PADDING - x_min) / GRID_RESOLUTION)
    rows = ceil((max(start.y, goal.y) + SEARCH_PADDING - y_min) / GRID_RESOLUTION)

    def cell(point: Point2D) -> Cell:
        return (
            round((point.x - x_min) / GRID_RESOLUTION),
            round((point.y - y_min) / GRID_RESOLUTION),
        )

    def centre(index: Cell) -> Point2D:
        return Point2D(
            x_min + index[0] * GRID_RESOLUTION, y_min + index[1] * GRID_RESOLUTION
        )

    origin, target = cell(start), cell(goal)
    if _blocked(centre(target), solid):
        raise PlanningError("the route's end point lies inside a wall or obstacle")
    frontier: list[tuple[float, float, Cell]] = [(0.0, 0.0, origin)]
    cost: dict[Cell, float] = {origin: 0.0}
    parent: dict[Cell, Cell] = {}
    while frontier:
        _, travelled, current = heapq.heappop(frontier)
        if current == target:
            break
        if travelled > cost[current]:
            continue
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neighbour = (current[0] + dx, current[1] + dy)
                if (dx == 0 and dy == 0) or not (
                    0 <= neighbour[0] <= columns and 0 <= neighbour[1] <= rows
                ):
                    continue
                point = centre(neighbour)
                if _blocked(point, solid):
                    continue
                weight = MARGIN_COST if _blocked(point, margin) else 1.0
                step = travelled + weight * GRID_RESOLUTION * hypot(dx, dy)
                if step < cost.get(neighbour, float("inf")):
                    cost[neighbour] = step
                    parent[neighbour] = current
                    remaining = GRID_RESOLUTION * hypot(
                        target[0] - neighbour[0], target[1] - neighbour[1]
                    )
                    heapq.heappush(frontier, (step + remaining, step, neighbour))
    if target not in cost:
        raise PlanningError("no route to the dock avoids the walls and obstacles")
    cells = [target]
    while cells[-1] != origin:
        cells.append(parent[cells[-1]])
    interior = [centre(index) for index in reversed(cells)][1:-1]
    return (start, *interior, goal)


def _shortened(
    points: tuple[Point2D, ...], walls: tuple[Box, ...]
) -> tuple[Point2D, ...]:
    """Keep only the corners needed for a wall-free polyline."""

    kept = [points[0]]
    anchor = 0
    while anchor < len(points) - 1:
        reach = len(points) - 1
        while reach > anchor + 1 and not _segment_clear(
            points[anchor], points[reach], walls
        ):
            reach -= 1
        kept.append(points[reach])
        anchor = reach
    return tuple(kept)


def entry_reference(
    vehicle: VehicleGeometry,
    dock: DockGeometry,
    state: VehicleState,
    obstacles: Sequence[Obstacle] = (),
) -> ReferencePath:
    """The default route; it does not look at obstacles."""

    return docking_reference(vehicle, dock, state)


def astar_reference(
    vehicle: VehicleGeometry,
    dock: DockGeometry,
    state: VehicleState,
    obstacles: Sequence[Obstacle] = (),
) -> ReferencePath:
    """Docking-point route found by A* around grown walls and obstacles."""

    goal = dock.target_position
    current = vehicle.trailer.rear_face(state.trailer_pose)
    clearance = vehicle.trailer.body_width / 2.0 + WALL_CLEARANCE
    entry = Point2D(
        dock.interior_depth + max(ENTRY_STANDOFF, clearance + GRID_RESOLUTION), 0.0
    )
    # Already at or inside the entry point with a clear line to the goal:
    # only the straight final leg is left, as the default planner draws it.
    # A start beside the bay has a wall in the way and is searched instead.
    near = current.x <= entry.x or hypot(current.x - entry.x, current.y) < 1e-9
    if near and _segment_clear(goal, current, _grown_walls(dock, 0.0, obstacles)):
        return docking_reference(vehicle, dock, state)
    walls = _grown_walls(dock, clearance, obstacles)
    solid = _grown_walls(dock, 0.0, obstacles)
    route = _shortened(_search(current, entry, solid, walls), walls)
    ordered = [goal, *reversed(route)]
    distinct = [ordered[0]]
    for point in ordered[1:]:
        if hypot(point.x - distinct[-1].x, point.y - distinct[-1].y) > 1e-9:
            distinct.append(point)
    return manual_reference(tuple(distinct))


PLANNERS: dict[str, Planner] = {
    "entry": entry_reference,
    "astar": astar_reference,
}
PLANNER_LABELS: dict[str, str] = {
    "entry": "straight in through an entry point",
    "astar": "A* grid search around the walls",
}


def validate_planner(name: object) -> str:
    """Return *name* if it is a registered planner, else raise."""

    if name not in PLANNERS:
        raise ValueError(f"planner must be one of {', '.join(sorted(PLANNERS))}")
    return str(name)


def plan_reference(
    name: str,
    vehicle: VehicleGeometry,
    dock: DockGeometry,
    state: VehicleState,
    obstacles: Sequence[Obstacle] = (),
) -> ReferencePath:
    """Plan with the registered planner called *name*."""

    return PLANNERS[validate_planner(name)](vehicle, dock, state, tuple(obstacles))
