"""Articulation thresholds shared by every backend and controller."""

from __future__ import annotations

from trailer_docking.core.config import VehicleLimits
from trailer_docking.core.state import VehicleState


def is_jackknifed(state: VehicleState, limits: VehicleLimits) -> bool:
    """Return whether articulation has reached the hard safety threshold."""

    return abs(state.articulation_angle) >= limits.articulation_hard_abs_max


def exceeds_soft_articulation(state: VehicleState, limits: VehicleLimits) -> bool:
    """Return whether articulation has passed the soft warning threshold."""

    return abs(state.articulation_angle) >= limits.articulation_soft_abs_max
