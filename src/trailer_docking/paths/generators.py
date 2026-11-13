"""Straight and manually defined curved reference-path generators."""

from __future__ import annotations

from trailer_docking.core.geometry import DockGeometry, VehicleGeometry
from trailer_docking.core.state import Point2D, VehicleState
from trailer_docking.paths.reference import ReferencePath

# Distance outside the bay opening at which an offset approach joins the
# bay centreline.
ENTRY_STANDOFF = 0.75


def straight_reference(goal: Point2D, start: Point2D) -> ReferencePath:
    """Return a goal-to-start line suitable for aligned reverse traversal."""

    return ReferencePath((goal, start))


def manual_reference(points_goal_to_start: tuple[Point2D, ...]) -> ReferencePath:
    """Create a piecewise-linear path from explicitly reviewed waypoints."""

    return ReferencePath(points_goal_to_start)


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
