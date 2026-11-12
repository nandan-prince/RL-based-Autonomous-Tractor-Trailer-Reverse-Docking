"""Path-relative tracking errors and preview points.

These are the "path-relative errors + preview points" of the proposal: the
quantities a trajectory-aware controller observes instead of global
coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

from trailer_docking.core.geometry import world_to_local
from trailer_docking.core.state import Pose2D, VehicleState, wrap_angle
from trailer_docking.paths.reference import ReferencePath


@dataclass(frozen=True, slots=True)
class PreviewPoint:
    """A path pose ahead of the vehicle, expressed in the trailer frame.

    This is the per-preview error triple of the trajectory-state formulation
    (Yan et al., 2024): ``lateral_offset`` is the preview point's sideways
    position in the trailer frame (positive left), and the two heading errors
    are vehicle heading minus the path heading at the preview point.
    """

    lateral_offset: float
    trailer_heading_error: float
    tractor_heading_error: float


@dataclass(frozen=True, slots=True)
class TrackingErrors:
    """Trailer and tractor errors relative to the closest path pose."""

    arc_length: float
    remaining_distance: float
    trailer_lateral_error: float
    trailer_heading_error: float
    tractor_heading_error: float
    path_curvature: float
    previews: tuple[PreviewPoint, ...]


# Search window around the previous projection, in metres of arc length.
# One control step moves the vehicle about 0.02 m.  On the inside of a curve
# the projection moves faster than the vehicle, so the window allows several
# times that, while still preventing a jump to a far part of the path.
SEARCH_BEHIND = 0.15
SEARCH_AHEAD = 0.15


def closest_arc_length(
    path: ReferencePath, pose: Pose2D, previous: float | None = None
) -> float:
    """Return the travelled distance of the path point closest to ``pose``.

    With ``previous`` (the projection from the last control step) only a
    window around it is searched.  On the inside of a curve the globally
    closest point can lie far ahead; without the window the projection would
    jump there and report metres of progress the vehicle never made.
    """

    best_distance = float("inf")
    if previous is not None:
        previous = min(max(previous, 0.0), path.length)
    best_arc = 0.0 if previous is None else previous
    poses, arcs = path.poses, path.arc_lengths
    for index in range(len(poses) - 1):
        if previous is not None and (
            arcs[index + 1] < previous - SEARCH_BEHIND
            or arcs[index] > previous + SEARCH_AHEAD
        ):
            continue
        start, end = poses[index], poses[index + 1]
        segment_x, segment_y = end.x - start.x, end.y - start.y
        length_squared = segment_x**2 + segment_y**2
        fraction = (
            (pose.x - start.x) * segment_x + (pose.y - start.y) * segment_y
        ) / length_squared
        fraction = min(max(fraction, 0.0), 1.0)
        distance = (pose.x - start.x - fraction * segment_x) ** 2 + (
            pose.y - start.y - fraction * segment_y
        ) ** 2
        if distance < best_distance:
            best_distance = distance
            best_arc = arcs[index] + fraction * (arcs[index + 1] - arcs[index])
    return best_arc


def tracking_errors(
    path: ReferencePath,
    state: VehicleState,
    preview_distances: tuple[float, ...] = (1.0, 2.0, 4.0),
    *,
    previous_arc_length: float | None = None,
) -> TrackingErrors:
    """Measure the vehicle against a trailer-axle reference path.

    Pass the previous step's ``arc_length`` as ``previous_arc_length`` when
    tracking continuously; omit it for a one-off or first measurement.
    """

    trailer = state.trailer_pose
    arc = closest_arc_length(path, trailer, previous_arc_length)
    reference = path.pose_at(arc)
    # Lateral error is the trailer axle's sideways position in the path frame.
    lateral = world_to_local(trailer.position, reference).y
    previews = []
    for distance in preview_distances:
        ahead = path.pose_at(arc + distance)
        previews.append(
            PreviewPoint(
                lateral_offset=world_to_local(ahead.position, trailer).y,
                trailer_heading_error=wrap_angle(trailer.yaw - ahead.yaw),
                tractor_heading_error=wrap_angle(state.tractor_pose.yaw - ahead.yaw),
            )
        )
    low = path.pose_at(max(arc - 0.05, 0.0))
    high = path.pose_at(min(arc + 0.05, path.length))
    span = min(arc + 0.05, path.length) - max(arc - 0.05, 0.0)
    return TrackingErrors(
        arc_length=arc,
        remaining_distance=path.length - arc,
        trailer_lateral_error=lateral,
        trailer_heading_error=wrap_angle(trailer.yaw - reference.yaw),
        tractor_heading_error=wrap_angle(state.tractor_pose.yaw - reference.yaw),
        path_curvature=wrap_angle(high.yaw - low.yaw) / span,
        previews=tuple(previews),
    )
