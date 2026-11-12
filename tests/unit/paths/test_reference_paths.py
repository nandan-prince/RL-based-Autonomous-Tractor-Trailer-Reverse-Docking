"""Reference paths, the fixed docking planner, and path-relative tracking."""

from math import cos, pi, sin

import pytest

from trailer_docking.core.config import DockConfiguration, VehicleConfiguration
from trailer_docking.core.state import Point2D, Pose2D, VehicleState
from trailer_docking.evaluation.scenarios import assembled_start
from trailer_docking.observations.builder import (
    BASE_FIELDS,
    ObservationLayout,
    build_observation,
)
from trailer_docking.paths.generators import (
    FixedDockingPlanner,
    goal_trailer_pose,
    plan_docking_route,
)
from trailer_docking.paths.reference import ReferencePath, ReferencePathError
from trailer_docking.paths.tracking import closest_arc_length, tracking_errors


def _state(
    vehicle: VehicleConfiguration,
    docking_point: Point2D,
    trailer_yaw: float = 0.0,
    articulation: float = 0.0,
) -> VehicleState:
    pose, yaw = assembled_start(
        vehicle.geometry, docking_point, trailer_yaw, articulation
    )
    return vehicle.geometry.assembled_state(pose, yaw)


def test_reference_path_derives_arc_length_and_curvature() -> None:
    # A reverse quarter circle of radius 5: heading is opposite to travel.
    radius, count = 5.0, 51
    poses = []
    for index in range(count):
        angle = (pi / 2.0) * index / (count - 1)
        poses.append(Pose2D(radius * cos(angle), radius * sin(angle), angle - pi / 2.0))
    path = ReferencePath.from_poses(poses)
    assert path.length == pytest.approx(radius * pi / 2.0, rel=1e-3)
    assert path.arc_lengths[0] == 0.0
    assert all(
        value == pytest.approx(1.0 / radius, rel=1e-2) for value in path.curvatures
    )
    assert path.max_abs_curvature == pytest.approx(0.2, rel=1e-2)
    assert len(path.points) == count
    middle = path.pose_at(path.length / 2.0)
    assert (middle.x**2 + middle.y**2) ** 0.5 == pytest.approx(radius, abs=1e-3)
    assert path.pose_at(-1.0) == path.poses[0]
    assert path.pose_at(1e9) == path.goal


def test_reference_path_rejects_malformed_input() -> None:
    with pytest.raises(ReferencePathError):
        ReferencePath.from_poses([Pose2D(0, 0, 0)])
    with pytest.raises(ReferencePathError):
        ReferencePath.from_poses([Pose2D(0, 0, 0), Pose2D(0, 0, 0.1)])
    with pytest.raises(ReferencePathError):
        ReferencePath((Pose2D(0, 0, 0), Pose2D(1, 0, 0)), (0.0, 1.0), (0.0,))
    with pytest.raises(ReferencePathError):
        ReferencePath((Pose2D(0, 0, 0), Pose2D(1, 0, 0)), (0.0, 0.0), (0.0, 0.0))
    # Duplicate consecutive positions are dropped rather than rejected.
    path = ReferencePath.from_poses([Pose2D(0, 0, 0), Pose2D(0, 0, 0), Pose2D(1, 0, 0)])
    assert len(path.poses) == 2


def test_planner_ends_on_the_dock_target(
    vehicle_config: VehicleConfiguration, dock_config: DockConfiguration
) -> None:
    geometry, bay = vehicle_config.geometry, dock_config.geometry
    start = _state(vehicle_config, Point2D(13.0, 0.4), 0.1)
    path = plan_docking_route(geometry, bay, start)
    goal = goal_trailer_pose(geometry, bay)
    assert path.reverse
    assert path.goal.to_dict() == pytest.approx(goal.to_dict(), abs=1e-9)
    # The goal puts the docking point exactly on the dock target.
    docking = geometry.trailer.rear_face(path.goal)
    assert (docking.x, docking.y) == pytest.approx(
        (bay.target_position.x, bay.target_position.y)
    )
    # The route starts at the trailer axle with the trailer's heading.
    assert path.poses[0].to_dict() == pytest.approx(
        start.trailer_pose.to_dict(), abs=1e-9
    )
    # It is straight on the bay centreline for the final approach.
    inside = [pose for pose in path.poses if pose.x < bay.interior_depth]
    assert inside and all(abs(pose.y) < 1e-9 for pose in inside)
    assert all(abs(pose.yaw) < 1e-9 for pose in inside)
    # Heading opposes the direction of travel everywhere (a reverse path).
    for earlier, later in zip(path.poses, path.poses[1:], strict=False):
        travel_x, travel_y = later.x - earlier.x, later.y - earlier.y
        assert travel_x * cos(earlier.yaw) + travel_y * sin(earlier.yaw) < 0.0
    assert path.max_abs_curvature < 0.2


def test_planner_handles_aligned_and_close_starts(
    vehicle_config: VehicleConfiguration, dock_config: DockConfiguration
) -> None:
    geometry, bay = vehicle_config.geometry, dock_config.geometry
    aligned = plan_docking_route(
        geometry, bay, _state(vehicle_config, Point2D(10.0, 0.0))
    )
    assert aligned.max_abs_curvature == pytest.approx(0.0, abs=1e-9)
    assert aligned.length == pytest.approx(10.0 - bay.target_position.x, abs=1e-6)
    # A start already inside the alignment zone still reaches the goal.
    close = plan_docking_route(
        geometry, bay, _state(vehicle_config, Point2D(6.2, 0.05))
    )
    assert close.goal.to_dict() == pytest.approx(
        goal_trailer_pose(geometry, bay).to_dict(), abs=1e-9
    )
    with pytest.raises(ValueError):
        FixedDockingPlanner(spacing=0.0)


def test_tracking_errors_use_the_documented_signs(
    vehicle_config: VehicleConfiguration, dock_config: DockConfiguration
) -> None:
    geometry, bay = vehicle_config.geometry, dock_config.geometry
    path = plan_docking_route(geometry, bay, _state(vehicle_config, Point2D(10.0, 0.0)))

    on_path = tracking_errors(path, _state(vehicle_config, Point2D(8.0, 0.0)))
    assert on_path.trailer_lateral_error == pytest.approx(0.0, abs=1e-9)
    assert on_path.trailer_heading_error == pytest.approx(0.0, abs=1e-9)
    assert on_path.arc_length == pytest.approx(2.0, abs=1e-6)
    assert on_path.remaining_distance == pytest.approx(path.length - 2.0, abs=1e-6)
    assert all(
        preview.lateral_offset == pytest.approx(0.0, abs=1e-9)
        for preview in on_path.previews
    )

    # Trailer displaced to the left (+y) and rotated counterclockwise, with
    # the tractor articulated further counterclockwise.
    off = tracking_errors(
        path, _state(vehicle_config, Point2D(8.0, 0.3), 0.1, 0.2), (1.0, 2.0)
    )
    assert off.trailer_lateral_error > 0.25
    assert off.trailer_heading_error == pytest.approx(0.1, abs=1e-9)
    assert off.tractor_heading_error == pytest.approx(0.3, abs=1e-9)
    assert len(off.previews) == 2
    # The path lies to the trailer's right, so preview points are at -y.
    assert all(preview.lateral_offset < 0.0 for preview in off.previews)
    assert off.previews[0].trailer_heading_error == pytest.approx(0.1, abs=1e-9)
    assert off.previews[0].tractor_heading_error == pytest.approx(0.3, abs=1e-9)

    # Projection clamps to the ends of the path.
    assert closest_arc_length(path, Pose2D(50.0, 0.0, 0.0)) == 0.0
    assert closest_arc_length(path, Pose2D(-50.0, 0.0, 0.0)) == pytest.approx(
        path.length
    )


def test_observation_layout_and_normalization(
    vehicle_config: VehicleConfiguration, dock_config: DockConfiguration
) -> None:
    geometry, bay = vehicle_config.geometry, dock_config.geometry
    path = plan_docking_route(geometry, bay, _state(vehicle_config, Point2D(10.0, 0.0)))
    state = _state(vehicle_config, Point2D(8.0, 0.3), 0.1, 0.2)
    errors = tracking_errors(path, state, (1.0, 2.0, 4.0))
    layout = ObservationLayout(preview_count=3)
    values = build_observation(state, errors, vehicle_config.limits)
    assert len(values) == layout.size == len(layout.names) == len(BASE_FIELDS) + 9
    named = dict(zip(layout.names, values, strict=True))
    assert named["trailer_heading_error"] == pytest.approx(0.1)
    assert named["tractor_heading_error"] == pytest.approx(0.3)
    assert named["articulation_angle"] == pytest.approx(
        0.2 / vehicle_config.limits.articulation_hard_abs_max
    )
    assert named["trailer_lateral_error"] > 0.0
    assert named["preview_0_lateral_offset"] < 0.0
    assert all(abs(value) <= 2.0 for value in values)
