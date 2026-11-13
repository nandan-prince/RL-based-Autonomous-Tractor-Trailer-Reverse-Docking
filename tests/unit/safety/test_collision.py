"""Planar collision, bay containment, and workspace tests."""

from trailer_docking.core.geometry import Bounds1D
from trailer_docking.core.state import Point2D, Pose2D
from trailer_docking.safety.collision import (
    polygons_intersect,
    trailer_inside_bay,
    vehicle_dock_collision,
    vehicle_inside_workspace,
)


def _box(x0: float, x1: float, y0: float, y1: float):
    return (
        Point2D(x0, y0),
        Point2D(x1, y0),
        Point2D(x1, y1),
        Point2D(x0, y1),
    )


def test_polygon_separation_overlap_and_touch() -> None:
    first = _box(0, 1, 0, 1)
    assert not polygons_intersect(first, _box(2, 3, 0, 1))
    assert polygons_intersect(first, _box(0.5, 1.5, 0.5, 1.5))
    assert polygons_intersect(first, _box(1, 2, 0, 1))


def test_nominal_target_is_collision_free_and_inside_bay(
    vehicle_config, dock_config
) -> None:
    target = dock_config.geometry.target_position
    tractor_x = (
        target.x
        + vehicle_config.trailer.rear_overhang
        + vehicle_config.trailer.hitch_to_axle
    )
    state = vehicle_config.geometry.assembled_state(
        Pose2D(tractor_x, target.y, 0), 0
    )
    assert not vehicle_dock_collision(
        state, vehicle_config.geometry, dock_config.geometry
    ).collided
    assert trailer_inside_bay(state, vehicle_config.geometry, dock_config.geometry)


def test_rear_and_side_wall_contacts_are_named(vehicle_config, dock_config) -> None:
    rear = vehicle_config.geometry.assembled_state(Pose2D(5.0, 0, 0), 0)
    rear_contacts = vehicle_dock_collision(
        rear, vehicle_config.geometry, dock_config.geometry
    ).contacts
    assert "trailer:rear_wall" in rear_contacts

    side = vehicle_config.geometry.assembled_state(Pose2D(8.0, 0.3, 0), 0)
    side_contacts = vehicle_dock_collision(
        side, vehicle_config.geometry, dock_config.geometry
    ).contacts
    assert "trailer:left_wall" in side_contacts


def test_complete_footprints_must_fit_workspace(vehicle_config) -> None:
    state = vehicle_config.geometry.assembled_state(Pose2D(5, 0, 0), 0)
    assert vehicle_inside_workspace(
        state, vehicle_config.geometry, Bounds1D(-2, 10), Bounds1D(-2, 2)
    )
    assert not vehicle_inside_workspace(
        state, vehicle_config.geometry, Bounds1D(0.1, 10), Bounds1D(-2, 2)
    )
