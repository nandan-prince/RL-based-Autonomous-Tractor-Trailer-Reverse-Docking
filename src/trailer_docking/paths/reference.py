"""Reference path produced by a planner and tracked by a controller.

A reference path is the route of the **trailer-axle midpoint**, the same
point as ``VehicleState.trailer_pose``.  Each waypoint carries the desired
trailer heading, so a vehicle tracking the path perfectly has zero lateral and
zero heading error at every waypoint.  The path is geometric (not
time-parameterised); arc length increases in the direction of travel, toward
the dock.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from math import hypot

from trailer_docking.core.state import Point2D, Pose2D, wrap_angle


class ReferencePathError(ValueError):
    """Raised when a reference path is malformed."""


@dataclass(frozen=True, slots=True)
class ReferencePath:
    """An ordered sequence of trailer-axle poses from start to goal.

    ``poses[i].yaw`` is the desired trailer heading (the vehicle's forward
    axis), which points *against* the direction of travel on a reverse
    segment.  ``arc_lengths[i]`` is the travelled distance from the first
    waypoint and ``curvatures[i]`` the rate of heading change per metre
    travelled.
    """

    poses: tuple[Pose2D, ...]
    arc_lengths: tuple[float, ...]
    curvatures: tuple[float, ...]
    reverse: bool = True

    def __post_init__(self) -> None:
        count = len(self.poses)
        if count < 2:
            raise ReferencePathError("a reference path needs at least two poses")
        if len(self.arc_lengths) != count or len(self.curvatures) != count:
            raise ReferencePathError("path arrays must have equal length")
        if self.arc_lengths[0] != 0.0 or any(
            later <= earlier
            for earlier, later in zip(
                self.arc_lengths, self.arc_lengths[1:], strict=False
            )
        ):
            raise ReferencePathError("arc lengths must start at 0 and increase")

    @classmethod
    def from_poses(
        cls, poses: Sequence[Pose2D], *, reverse: bool = True
    ) -> ReferencePath:
        """Build a path from poses, deriving arc length and curvature.

        Consecutive duplicate positions are dropped.
        """

        kept: list[Pose2D] = []
        for pose in poses:
            if kept and hypot(pose.x - kept[-1].x, pose.y - kept[-1].y) < 1e-9:
                continue
            kept.append(pose)
        if len(kept) < 2:
            raise ReferencePathError("a reference path needs at least two poses")
        arc = [0.0]
        for earlier, later in itertools.pairwise(kept):
            arc.append(arc[-1] + hypot(later.x - earlier.x, later.y - earlier.y))
        curvature = []
        last = len(kept) - 1
        for index in range(len(kept)):
            low, high = max(index - 1, 0), min(index + 1, last)
            span = arc[high] - arc[low]
            curvature.append(wrap_angle(kept[high].yaw - kept[low].yaw) / span)
        return cls(tuple(kept), tuple(arc), tuple(curvature), reverse)

    @property
    def length(self) -> float:
        return self.arc_lengths[-1]

    @property
    def points(self) -> tuple[Point2D, ...]:
        """Waypoint positions, for drawing."""

        return tuple(pose.position for pose in self.poses)

    @property
    def goal(self) -> Pose2D:
        return self.poses[-1]

    @property
    def max_abs_curvature(self) -> float:
        return max(abs(value) for value in self.curvatures)

    def pose_at(self, arc_length: float) -> Pose2D:
        """Interpolate the pose at a travelled distance, clamped to the path."""

        if arc_length <= 0.0:
            return self.poses[0]
        if arc_length >= self.length:
            return self.poses[-1]
        low, high = 0, len(self.arc_lengths) - 1
        while high - low > 1:
            middle = (low + high) // 2
            if self.arc_lengths[middle] <= arc_length:
                low = middle
            else:
                high = middle
        start, end = self.poses[low], self.poses[high]
        fraction = (arc_length - self.arc_lengths[low]) / (
            self.arc_lengths[high] - self.arc_lengths[low]
        )
        return Pose2D(
            start.x + fraction * (end.x - start.x),
            start.y + fraction * (end.y - start.y),
            start.yaw + fraction * wrap_angle(end.yaw - start.yaw),
        )
