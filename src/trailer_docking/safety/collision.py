"""Planar collision between vehicle footprints and dock walls.

Collision is a property of the shared planar contract: a chassis footprint
either overlaps a wall rectangle or it does not.  Both simulator backends are
judged by this one function, so an episode ends for the same geometric reason
in 2D and in MuJoCo.
"""

from __future__ import annotations

from collections.abc import Sequence

from trailer_docking.core.geometry import DockGeometry, VehicleGeometry
from trailer_docking.core.state import Point2D, VehicleState

Polygon = Sequence[Point2D]


def convex_polygons_overlap(
    first: Polygon, second: Polygon, *, tolerance: float = 1e-9
) -> bool:
    """Return whether two convex polygons overlap by more than ``tolerance``.

    Uses the separating-axis test.  Polygons that merely touch are not in
    collision.
    """

    for polygon in (first, second):
        count = len(polygon)
        for index in range(count):
            start = polygon[index]
            end = polygon[(index + 1) % count]
            normal_x = -(end.y - start.y)
            normal_y = end.x - start.x
            length = (normal_x**2 + normal_y**2) ** 0.5
            if length == 0.0:
                continue
            normal_x /= length
            normal_y /= length
            first_projection = [p.x * normal_x + p.y * normal_y for p in first]
            second_projection = [p.x * normal_x + p.y * normal_y for p in second]
            if (
                max(first_projection) <= min(second_projection) + tolerance
                or max(second_projection) <= min(first_projection) + tolerance
            ):
                return False
    return True


def _rectangle(
    x_min: float, x_max: float, y_min: float, y_max: float
) -> tuple[Point2D, ...]:
    return (
        Point2D(x_min, y_min),
        Point2D(x_max, y_min),
        Point2D(x_max, y_max),
        Point2D(x_min, y_max),
    )


def dock_wall_polygons(dock: DockGeometry) -> dict[str, tuple[Point2D, ...]]:
    """Return the back and side wall rectangles in the dock frame."""

    half_width = dock.interior_width / 2.0
    thickness = dock.wall_thickness
    depth = dock.interior_depth
    return {
        "back_wall": _rectangle(
            -thickness, 0.0, -half_width - thickness, half_width + thickness
        ),
        "left_wall": _rectangle(0.0, depth, half_width, half_width + thickness),
        "right_wall": _rectangle(0.0, depth, -half_width - thickness, -half_width),
    }


def wall_collisions(
    state: VehicleState, vehicle: VehicleGeometry, dock: DockGeometry
) -> tuple[str, ...]:
    """Return sorted ``"<body>:<wall>"`` pairs whose footprints overlap."""

    footprints = {
        "tractor": vehicle.tractor.footprint(state.tractor_pose),
        "trailer": vehicle.trailer.footprint(state.trailer_pose),
    }
    walls = dock_wall_polygons(dock)
    return tuple(
        sorted(
            f"{body}:{wall}"
            for body, footprint in footprints.items()
            for wall, rectangle in walls.items()
            if convex_polygons_overlap(footprint, rectangle)
        )
    )
