"""Path-relative state used by rewards, observations, and controllers."""

from __future__ import annotations

from dataclasses import dataclass

from trailer_docking.core.geometry import world_to_local
from trailer_docking.core.state import Point2D, Pose2D, VehicleState, wrap_angle
from trailer_docking.paths.reference import PathSample, ReferencePath


@dataclass(frozen=True, slots=True)
class TrackingState:
    """Errors at the trailer docking point and local preview coordinates."""

    progress: float
    distance_to_goal: float
    lateral_error: float
    trailer_heading_error: float
    tractor_heading_error: float
    preview_samples: tuple[PathSample, ...]
    preview_points_trailer_frame: tuple[Point2D, ...]


def tracking_state(
    state: VehicleState,
    path: ReferencePath,
    docking_point: Point2D,
    preview_distances: tuple[float, ...],
) -> TrackingState:
    """Calculate the complete path-relative state for one vehicle state."""

    if not isinstance(state, VehicleState):
        raise TypeError("state must be VehicleState")
    if not isinstance(path, ReferencePath):
        raise TypeError("path must be ReferencePath")
    if not isinstance(docking_point, Point2D):
        raise TypeError("docking_point must be Point2D")
    nearest = path.nearest(docking_point)
    previews = path.previews(nearest.sample.progress, preview_distances)
    docking_frame = Pose2D(docking_point.x, docking_point.y, state.trailer_pose.yaw)
    local = tuple(world_to_local(sample.point, docking_frame) for sample in previews)
    heading = nearest.sample.heading
    return TrackingState(
        progress=nearest.sample.progress,
        distance_to_goal=nearest.sample.progress,
        lateral_error=nearest.signed_lateral_error,
        trailer_heading_error=wrap_angle(state.trailer_pose.yaw - heading),
        tractor_heading_error=wrap_angle(state.tractor_pose.yaw - heading),
        preview_samples=previews,
        preview_points_trailer_frame=local,
    )
