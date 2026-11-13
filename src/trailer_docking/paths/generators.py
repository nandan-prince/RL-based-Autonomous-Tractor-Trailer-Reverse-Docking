"""Straight and manually defined curved reference-path generators."""

from __future__ import annotations

from trailer_docking.core.state import Point2D
from trailer_docking.paths.reference import ReferencePath


def straight_reference(goal: Point2D, start: Point2D) -> ReferencePath:
    """Return a goal-to-start line suitable for aligned reverse traversal."""

    return ReferencePath((goal, start))


def manual_reference(points_goal_to_start: tuple[Point2D, ...]) -> ReferencePath:
    """Create a piecewise-linear path from explicitly reviewed waypoints."""

    return ReferencePath(points_goal_to_start)
