"""Docking error and success evaluation on the shared planar state.

These functions implement Section 11.1 of the vehicle and environment
specification.  They take only shared types, so 2D and MuJoCo rollouts are
scored identically.
"""

from __future__ import annotations

from dataclasses import dataclass

from trailer_docking.core.config import VehicleLimits
from trailer_docking.core.geometry import DockGeometry, VehicleGeometry, world_to_local
from trailer_docking.core.state import Pose2D, VehicleState, wrap_angle
from trailer_docking.safety.collision import wall_collisions
from trailer_docking.safety.jackknife import is_jackknifed


@dataclass(frozen=True, slots=True)
class DockingError:
    """Docking-point error expressed in the target frame.

    ``longitudinal`` is along the target trailer heading (positive means the
    docking point is still outside the target), ``lateral`` is to its left, and
    ``yaw`` is trailer heading minus target heading, wrapped.
    """

    longitudinal: float
    lateral: float
    yaw: float

    @property
    def distance(self) -> float:
        return (self.longitudinal**2 + self.lateral**2) ** 0.5


@dataclass(frozen=True, slots=True)
class DockingOutcome:
    """Every success condition of the specification, evaluated separately."""

    error: DockingError
    within_tolerance: bool
    trailer_inside_bay: bool
    collisions: tuple[str, ...]
    jackknifed: bool
    success: bool


def docking_error(
    state: VehicleState, vehicle: VehicleGeometry, dock: DockGeometry
) -> DockingError:
    """Measure the docking point against the dock target."""

    docking_point = vehicle.trailer.rear_face(state.trailer_pose)
    target = Pose2D(
        dock.target_position.x, dock.target_position.y, dock.target_trailer_yaw
    )
    local = world_to_local(docking_point, target)
    return DockingError(
        longitudinal=local.x,
        lateral=local.y,
        yaw=wrap_angle(state.trailer_pose.yaw - dock.target_trailer_yaw),
    )


def trailer_inside_bay(
    state: VehicleState, vehicle: VehicleGeometry, dock: DockGeometry
) -> bool:
    """Return whether the whole trailer footprint is inside the bay interior.

    The configured collision margin is excluded from the usable interior.
    """

    margin = dock.collision_margin
    half_width = dock.interior_width / 2.0
    return all(
        margin <= corner.x <= dock.interior_depth - margin
        and -half_width + margin <= corner.y <= half_width - margin
        for corner in vehicle.trailer.footprint(state.trailer_pose)
    )


def evaluate_docking(
    state: VehicleState,
    vehicle: VehicleGeometry,
    dock: DockGeometry,
    limits: VehicleLimits,
) -> DockingOutcome:
    """Evaluate all docking-success conditions at one control step."""

    error = docking_error(state, vehicle, dock)
    within_tolerance = (
        abs(error.longitudinal) <= dock.longitudinal_tolerance
        and abs(error.lateral) <= dock.lateral_tolerance
        and abs(error.yaw) <= dock.trailer_yaw_tolerance
    )
    inside = trailer_inside_bay(state, vehicle, dock)
    collisions = wall_collisions(state, vehicle, dock)
    jackknifed = is_jackknifed(state, limits)
    success = (
        within_tolerance
        and (inside or not dock.require_complete_trailer_inside_bay)
        and (not collisions or not dock.require_no_collision)
        and (not jackknifed or not dock.require_articulation_below_hard_limit)
    )
    return DockingOutcome(
        error=error,
        within_tolerance=within_tolerance,
        trailer_inside_bay=inside,
        collisions=collisions,
        jackknifed=jackknifed,
        success=success,
    )
