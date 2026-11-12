"""Reference-path planners for the docking task.

The proposal's pipeline is *map + dock pose -> path planner -> controller*.
This module provides the planner interface and the first, deliberately simple
planner: a predefined curve-then-straight docking path.  Reeds-Shepp or
Hybrid A* planners can replace it behind the same :class:`Planner` protocol.

.. warning::
   :class:`FixedDockingPlanner` is geometric only.  It does not check trailer
   articulation feasibility or collisions along the path; the proposal
   requires every generated path to be simulated and collision-checked before
   it is used in an experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, ceil, cos, hypot, sin
from typing import Protocol

from trailer_docking.core.geometry import DockGeometry, VehicleGeometry
from trailer_docking.core.state import Pose2D, VehicleState
from trailer_docking.paths.reference import ReferencePath


class Planner(Protocol):
    """Anything that turns a start state and a dock into a reference path."""

    def plan(
        self, vehicle: VehicleGeometry, dock: DockGeometry, start: VehicleState
    ) -> ReferencePath:
        """Return the trailer-axle route from ``start`` to the dock target."""


def goal_trailer_pose(vehicle: VehicleGeometry, dock: DockGeometry) -> Pose2D:
    """Trailer-axle pose that puts the docking point on the dock target."""

    yaw = dock.target_trailer_yaw
    overhang = vehicle.trailer.rear_overhang
    return Pose2D(
        dock.target_position.x + overhang * cos(yaw),
        dock.target_position.y + overhang * sin(yaw),
        yaw,
    )


@dataclass(frozen=True, slots=True)
class FixedDockingPlanner:
    """Predefined docking path: a smooth curve onto the bay axis, then straight.

    The trailer axle is brought onto the bay centreline ``alignment_distance``
    metres before the docking point reaches the bay opening, and then reverses
    straight to the target.
    """

    alignment_distance: float = 1.0
    spacing: float = 0.1

    def __post_init__(self) -> None:
        if self.alignment_distance < 0.0 or self.spacing <= 0.0:
            raise ValueError("planner distances must be positive")

    def plan(
        self, vehicle: VehicleGeometry, dock: DockGeometry, start: VehicleState
    ) -> ReferencePath:
        goal = goal_trailer_pose(vehicle, dock)
        forward_x, forward_y = cos(goal.yaw), sin(goal.yaw)
        # Distance of the trailer axle from the target axle pose, measured
        # outward along the bay axis, at the alignment point.
        target_to_opening = dock.interior_depth - (
            (dock.target_position.x) * cos(dock.target_trailer_yaw)
            + (dock.target_position.y) * sin(dock.target_trailer_yaw)
        )
        aligned_offset = max(target_to_opening + self.alignment_distance, 0.0)
        aligned = Pose2D(
            goal.x + aligned_offset * forward_x,
            goal.y + aligned_offset * forward_y,
            goal.yaw,
        )
        begin = start.trailer_pose
        # If the start is already past the alignment point, curve directly to
        # the goal instead.
        start_offset = (begin.x - goal.x) * forward_x + (begin.y - goal.y) * forward_y
        poses: list[Pose2D]
        if start_offset > aligned_offset + self.spacing:
            poses = self._curve(begin, aligned)
            poses += self._straight(aligned, goal)[1:]
        else:
            poses = self._curve(begin, goal)
        return ReferencePath.from_poses(poses, reverse=True)

    def _straight(self, begin: Pose2D, end: Pose2D) -> list[Pose2D]:
        distance = hypot(end.x - begin.x, end.y - begin.y)
        steps = max(ceil(distance / self.spacing), 1)
        return [
            Pose2D(
                begin.x + (end.x - begin.x) * index / steps,
                begin.y + (end.y - begin.y) * index / steps,
                end.yaw,
            )
            for index in range(steps + 1)
        ]

    def _curve(self, begin: Pose2D, end: Pose2D) -> list[Pose2D]:
        """Cubic Hermite curve whose end tangents follow the trailer headings.

        Travel is in reverse, so the direction of travel at each end is the
        opposite of the trailer heading there.
        """

        distance = hypot(end.x - begin.x, end.y - begin.y)
        if distance < 1e-9:
            return [begin, end]
        scale = distance
        begin_tangent = (-cos(begin.yaw) * scale, -sin(begin.yaw) * scale)
        end_tangent = (-cos(end.yaw) * scale, -sin(end.yaw) * scale)
        steps = max(ceil(1.5 * distance / self.spacing), 2)
        poses: list[Pose2D] = []
        for index in range(steps + 1):
            t = index / steps
            h00 = 2 * t**3 - 3 * t**2 + 1
            h10 = t**3 - 2 * t**2 + t
            h01 = -2 * t**3 + 3 * t**2
            h11 = t**3 - t**2
            x = (
                h00 * begin.x
                + h10 * begin_tangent[0]
                + h01 * end.x
                + h11 * end_tangent[0]
            )
            y = (
                h00 * begin.y
                + h10 * begin_tangent[1]
                + h01 * end.y
                + h11 * end_tangent[1]
            )
            d00 = 6 * t**2 - 6 * t
            d10 = 3 * t**2 - 4 * t + 1
            d01 = -6 * t**2 + 6 * t
            d11 = 3 * t**2 - 2 * t
            dx = (
                d00 * begin.x
                + d10 * begin_tangent[0]
                + d01 * end.x
                + d11 * end_tangent[0]
            )
            dy = (
                d00 * begin.y
                + d10 * begin_tangent[1]
                + d01 * end.y
                + d11 * end_tangent[1]
            )
            if index == 0:
                yaw = begin.yaw
            elif index == steps:
                yaw = end.yaw
            else:
                # Heading is opposite to the direction of travel.
                yaw = atan2(-dy, -dx)
            poses.append(Pose2D(x, y, yaw))
        return poses


DEFAULT_PLANNER = FixedDockingPlanner()


def plan_docking_route(
    vehicle: VehicleGeometry, dock: DockGeometry, start: VehicleState
) -> ReferencePath:
    """Plan with the default planner; the route every viewer and env shows."""

    return DEFAULT_PLANNER.plan(vehicle, dock, start)
