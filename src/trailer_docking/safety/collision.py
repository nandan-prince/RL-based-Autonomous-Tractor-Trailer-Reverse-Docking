"""Dependency-free convex polygon and loading-dock collision checks."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from trailer_docking.core.geometry import Bounds1D, DockGeometry, VehicleGeometry
from trailer_docking.core.state import Point2D, VehicleState

Polygon2D = tuple[Point2D, ...]


@dataclass(frozen=True, slots=True)
class CollisionReport:
    """Stable names of dock objects touched by each vehicle body."""

    contacts: tuple[str, ...]

    @property
    def collided(self) -> bool:
        return bool(self.contacts)


def _rectangle(x: Bounds1D, y: Bounds1D) -> Polygon2D:
    return (
        Point2D(x.minimum, y.minimum),
        Point2D(x.maximum, y.minimum),
        Point2D(x.maximum, y.maximum),
        Point2D(x.minimum, y.maximum),
    )


def dock_wall_polygons(dock: DockGeometry) -> dict[str, Polygon2D]:
    """Return walls inflated inward by the configured collision margin."""

    if not isinstance(dock, DockGeometry):
        raise TypeError("dock must be DockGeometry")
    half = dock.interior_width / 2.0
    thickness = dock.wall_thickness
    margin = dock.collision_margin
    rear_wall = _rectangle(
        Bounds1D(-thickness, margin),
        Bounds1D(-half - thickness, half + thickness),
    )
    if dock.specification_id == "service-center-dock-v1":
        # The service-center target is an open, painted loading zone. Its two
        # longitudinal edges are intentionally not physical collision walls.
        building_half_length = 100.0
        building_depth = 5.0
        return {
            "rear_wall": rear_wall,
            "building_left_facade": _rectangle(
                Bounds1D(-building_depth, margin),
                Bounds1D(half + thickness, building_half_length),
            ),
            "building_right_facade": _rectangle(
                Bounds1D(-building_depth, margin),
                Bounds1D(-building_half_length, -half - thickness),
            ),
        }
    return {
        "rear_wall": _rectangle(
            Bounds1D(-thickness, margin),
            Bounds1D(-half - thickness, half + thickness),
        ),
        "left_wall": _rectangle(
            Bounds1D(-thickness, dock.interior_depth),
            Bounds1D(half - margin, half + thickness),
        ),
        "right_wall": _rectangle(
            Bounds1D(-thickness, dock.interior_depth),
            Bounds1D(-half - thickness, -half + margin),
        ),
    }


def polygons_intersect(first: Polygon2D, second: Polygon2D) -> bool:
    """Separating-axis test; touching boundaries count as collision."""

    if len(first) < 3 or len(second) < 3:
        raise ValueError("collision polygons need at least three points")
    for polygon in (first, second):
        for start, end in zip(polygon, polygon[1:] + polygon[:1], strict=True):
            edge_x, edge_y = end.x - start.x, end.y - start.y
            length = hypot(edge_x, edge_y)
            if length == 0.0:
                raise ValueError("collision polygons cannot contain zero-length edges")
            axis_x, axis_y = -edge_y / length, edge_x / length
            first_projection = tuple(
                point.x * axis_x + point.y * axis_y for point in first
            )
            second_projection = tuple(
                point.x * axis_x + point.y * axis_y for point in second
            )
            if max(first_projection) < min(second_projection) or max(
                second_projection
            ) < min(first_projection):
                return False
    return True


def vehicle_dock_collision(
    state: VehicleState, vehicle: VehicleGeometry, dock: DockGeometry
) -> CollisionReport:
    """Check tractor and trailer chassis footprints against all dock walls."""

    if not isinstance(state, VehicleState):
        raise TypeError("state must be VehicleState")
    bodies = {
        "tractor": vehicle.tractor.footprint(state.tractor_pose),
        "trailer": vehicle.trailer.footprint(state.trailer_pose),
    }
    contacts = tuple(
        f"{body}:{wall}"
        for body, footprint in bodies.items()
        for wall, wall_polygon in dock_wall_polygons(dock).items()
        if polygons_intersect(footprint, wall_polygon)
    )
    return CollisionReport(contacts)


def vehicle_inside_workspace(
    state: VehicleState,
    vehicle: VehicleGeometry,
    x_bounds: Bounds1D,
    y_bounds: Bounds1D,
) -> bool:
    """Return true only when both complete chassis footprints are in bounds."""

    points = vehicle.tractor.footprint(
        state.tractor_pose
    ) + vehicle.trailer.footprint(state.trailer_pose)
    return all(
        x_bounds.minimum <= point.x <= x_bounds.maximum
        and y_bounds.minimum <= point.y <= y_bounds.maximum
        for point in points
    )


def trailer_inside_bay(
    state: VehicleState, vehicle: VehicleGeometry, dock: DockGeometry
) -> bool:
    """Check the complete trailer footprint against the usable bay interior."""

    margin = dock.collision_margin
    half = dock.interior_width / 2.0
    return all(
        margin <= point.x <= dock.interior_depth
        and -half + margin <= point.y <= half - margin
        for point in vehicle.trailer.footprint(state.trailer_pose)
    )
