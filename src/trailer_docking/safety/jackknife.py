"""Articulation warning and terminal jackknife classification."""

from __future__ import annotations

from dataclasses import dataclass

from trailer_docking.core.config import VehicleLimits
from trailer_docking.core.state import VehicleState


@dataclass(frozen=True, slots=True)
class ArticulationStatus:
    """Soft warning and hard terminal state for one articulation angle."""

    magnitude: float
    soft_limit_reached: bool
    jackknifed: bool


def articulation_status(
    state: VehicleState, limits: VehicleLimits
) -> ArticulationStatus:
    """Evaluate shared articulation thresholds without backend tolerance."""

    if not isinstance(state, VehicleState):
        raise TypeError("state must be VehicleState")
    magnitude = abs(state.articulation_angle)
    return ArticulationStatus(
        magnitude=magnitude,
        soft_limit_reached=magnitude >= limits.articulation_soft_abs_max,
        jackknifed=magnitude >= limits.articulation_hard_abs_max,
    )
