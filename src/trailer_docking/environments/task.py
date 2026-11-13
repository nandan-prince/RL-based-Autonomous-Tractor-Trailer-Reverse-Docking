"""Shared reverse-docking scenario, goal, and terminal-event definitions."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin

from trailer_docking.core.config import (
    DockingTaskConfiguration,
    VehicleConfiguration,
)
from trailer_docking.core.geometry import Bounds1D, DockGeometry
from trailer_docking.core.state import Point2D, Pose2D, VehicleState, wrap_angle
from trailer_docking.safety.collision import (
    CollisionReport,
    trailer_inside_bay,
    vehicle_dock_collision,
    vehicle_inside_workspace,
)
from trailer_docking.safety.jackknife import ArticulationStatus, articulation_status


@dataclass(frozen=True, slots=True)
class DockingErrors:
    """Trailer docking-point and heading error in the dock frame."""

    longitudinal: float
    lateral: float
    trailer_yaw: float


@dataclass(frozen=True, slots=True)
class TaskStatus:
    """All independent success and terminal conditions for one state."""

    success: bool
    collision: CollisionReport
    articulation: ArticulationStatus
    out_of_bounds: bool
    trailer_completely_inside: bool
    errors: DockingErrors

    @property
    def terminated(self) -> bool:
        return (
            self.success
            or self.collision.collided
            or self.articulation.jackknifed
            or self.out_of_bounds
        )

    @property
    def reason(self) -> str | None:
        if self.success:
            return "success"
        if self.collision.collided:
            return "collision"
        if self.articulation.jackknifed:
            return "jackknife"
        if self.out_of_bounds:
            return "out_of_bounds"
        return None


def aligned_pose_for_docking_point(
    vehicle: VehicleConfiguration, docking_point: Point2D, yaw: float = 0.0
) -> Pose2D:
    """Tractor pose for an aligned assembly with a requested trailer rear point."""

    heading = wrap_angle(yaw)
    trailer_axle_x = docking_point.x + vehicle.trailer.rear_overhang * cos(heading)
    trailer_axle_y = docking_point.y + vehicle.trailer.rear_overhang * sin(heading)
    hitch_x = trailer_axle_x + vehicle.trailer.hitch_to_axle * cos(heading)
    hitch_y = trailer_axle_y + vehicle.trailer.hitch_to_axle * sin(heading)
    tractor = vehicle.tractor
    offset_x = tractor.hitch_longitudinal * cos(heading) - tractor.hitch_lateral * sin(
        heading
    )
    offset_y = tractor.hitch_longitudinal * sin(heading) + tractor.hitch_lateral * cos(
        heading
    )
    return Pose2D(hitch_x - offset_x, hitch_y - offset_y, heading)


def default_start_pose(
    vehicle: VehicleConfiguration,
    dock: DockGeometry,
    settings: DockingTaskConfiguration,
) -> Pose2D:
    """Straight collision-free start outside the dock opening."""

    rear_point = Point2D(dock.interior_depth + settings.start_standoff, 0.0)
    return aligned_pose_for_docking_point(vehicle, rear_point)


def workspace_bounds(
    start_pose: Pose2D,
    dock: DockGeometry,
    settings: DockingTaskConfiguration,
) -> tuple[Bounds1D, Bounds1D]:
    """Finite operating region used for out-of-bounds termination."""

    outward_maximum = (
        settings.workspace_outward_limit
        if settings.workspace_outward_limit is not None
        else max(start_pose.x, dock.interior_depth) + settings.workspace_front_padding
    )
    return (
        Bounds1D(
            -settings.workspace_rear_padding,
            outward_maximum,
        ),
        Bounds1D(
            -settings.workspace_lateral_half_width,
            settings.workspace_lateral_half_width,
        ),
    )


def docking_errors(
    state: VehicleState, vehicle: VehicleConfiguration, dock: DockGeometry
) -> DockingErrors:
    """Evaluate the frozen target at the trailer rear-face docking point."""

    point = vehicle.trailer.rear_face(state.trailer_pose)
    return DockingErrors(
        longitudinal=point.x - dock.target_position.x,
        lateral=point.y - dock.target_position.y,
        trailer_yaw=wrap_angle(state.trailer_pose.yaw - dock.target_trailer_yaw),
    )


def evaluate_task(
    state: VehicleState,
    vehicle: VehicleConfiguration,
    dock: DockGeometry,
    x_bounds: Bounds1D,
    y_bounds: Bounds1D,
) -> TaskStatus:
    """Evaluate collision, jackknife, boundary, and docking success separately."""

    errors = docking_errors(state, vehicle, dock)
    collision = vehicle_dock_collision(state, vehicle.geometry, dock)
    articulation = articulation_status(state, vehicle.limits)
    inside = trailer_inside_bay(state, vehicle.geometry, dock)
    out = not vehicle_inside_workspace(
        state, vehicle.geometry, x_bounds, y_bounds
    )
    within_pose = (
        abs(errors.longitudinal) <= dock.longitudinal_tolerance
        and abs(errors.lateral) <= dock.lateral_tolerance
        and abs(errors.trailer_yaw) <= dock.trailer_yaw_tolerance
    )
    success = (
        within_pose
        and (not dock.require_no_collision or not collision.collided)
        and (
            not dock.require_articulation_below_hard_limit
            or not articulation.jackknifed
        )
        and (not dock.require_complete_trailer_inside_bay or inside)
    )
    return TaskStatus(success, collision, articulation, out, inside, errors)
