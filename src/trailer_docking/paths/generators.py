"""Straight and manually defined curved reference-path generators."""

from __future__ import annotations

from trailer_docking.core.geometry import DockGeometry, VehicleGeometry
from trailer_docking.core.state import Point2D, VehicleState
from trailer_docking.paths.reference import ReferencePath

# Distance outside the bay opening at which an offset approach joins the
# bay centreline.
ENTRY_STANDOFF = 0.75
CURVE_SAMPLES = 41


def straight_reference(goal: Point2D, start: Point2D) -> ReferencePath:
    """Return a goal-to-start line suitable for aligned reverse traversal."""

    return ReferencePath((goal, start))


def manual_reference(points_goal_to_start: tuple[Point2D, ...]) -> ReferencePath:
    """Create a piecewise-linear path from explicitly reviewed waypoints."""

    return ReferencePath(points_goal_to_start)


def smooth_docking_reference(
    goal: Point2D,
    start: Point2D,
    *,
    goal_control_fraction: float = 0.35,
    start_control_fraction: float = 0.35,
    samples: int = CURVE_SAMPLES,
) -> ReferencePath:
    """Return a sampled cubic docking curve with aligned endpoint tangents.

    The path is ordered from the dock goal outward to the episode start, like
    every project reference path. Both endpoint tangents point along ``+x`` so
    an aligned tractor--trailer can begin and finish the reverse maneuver
    without an artificial heading discontinuity. The two fractions define a
    reproducible curve template rather than an online path optimization.
    """

    if samples < 2:
        raise ValueError("a smooth docking reference needs at least two samples")
    for name, fraction in (
        ("goal_control_fraction", goal_control_fraction),
        ("start_control_fraction", start_control_fraction),
    ):
        if not 0.0 < fraction < 1.0:
            raise ValueError(f"{name} must be between zero and one")
    span = start.x - goal.x
    if span <= 1e-9 or abs(start.y - goal.y) <= 1e-12:
        return straight_reference(goal, start)
    first_control = Point2D(goal.x + goal_control_fraction * span, goal.y)
    second_control = Point2D(start.x - start_control_fraction * span, start.y)
    points: list[Point2D] = []
    for index in range(samples):
        parameter = index / (samples - 1)
        complement = 1.0 - parameter
        points.append(
            Point2D(
                complement**3 * goal.x
                + 3.0 * complement**2 * parameter * first_control.x
                + 3.0 * complement * parameter**2 * second_control.x
                + parameter**3 * start.x,
                complement**3 * goal.y
                + 3.0 * complement**2 * parameter * first_control.y
                + 3.0 * complement * parameter**2 * second_control.y
                + parameter**3 * start.y,
            )
        )
    return ReferencePath(tuple(points))


def docking_reference(
    vehicle: VehicleGeometry, dock: DockGeometry, state: VehicleState
) -> ReferencePath:
    """Goal-to-start docking-point reference for one start state.

    An aligned start reverses along the bay centreline.  An offset start first
    joins the centreline at an entry point just outside the bay opening, unless
    it already is nearer the bay than that point: a reversing vehicle cannot
    drive back out to it, so the reference then goes straight to the goal.
    This is the reference the Gymnasium task tracks, so viewers that call it
    draw the route a controller is asked to follow.
    """

    goal = dock.target_position
    current = vehicle.trailer.rear_face(state.trailer_pose)
    entry = Point2D(dock.interior_depth + ENTRY_STANDOFF, 0.0)
    if abs(current.x - goal.x) < 1e-12 and abs(current.y - goal.y) < 1e-12:
        return manual_reference((goal, entry))
    if abs(current.y) < 1e-9 or current.x <= entry.x:
        return manual_reference((goal, current))
    return manual_reference((goal, entry, current))
