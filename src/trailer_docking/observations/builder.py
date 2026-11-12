"""Trajectory-relative observation for path-tracking controllers.

The vector follows the proposal: trailer lateral error, tractor and trailer
heading errors, articulation angle and rate, steering state, distance to the
goal, and lateral/heading offsets of several preview points.  Each preview
point carries the error triple of the trajectory-state formulation (Yan et
al., 2024): lateral offset in the trailer frame and trailer and tractor
heading errors.  The preview count, spacing, and normalization are open
choices in the source material and are plain constants here.  Every entry is
normalized to roughly ``[-1, 1]`` and contains no global coordinates, so the
policy does not depend on where the dock is.
"""

from __future__ import annotations

from dataclasses import dataclass

from trailer_docking.core.config import VehicleLimits
from trailer_docking.core.state import VehicleState
from trailer_docking.paths.tracking import TrackingErrors

LATERAL_SCALE = 2.0
HEADING_SCALE = 1.0
DISTANCE_SCALE = 15.0
ARTICULATION_RATE_SCALE = 0.5

BASE_FIELDS = (
    "trailer_lateral_error",
    "trailer_heading_error",
    "tractor_heading_error",
    "articulation_angle",
    "articulation_rate",
    "steering_angle",
    "steering_rate",
    "remaining_distance",
)


@dataclass(frozen=True, slots=True)
class ObservationLayout:
    """Names of the observation entries, in order."""

    preview_count: int

    @property
    def names(self) -> tuple[str, ...]:
        previews = tuple(
            f"preview_{index}_{kind}"
            for index in range(self.preview_count)
            for kind in (
                "lateral_offset",
                "trailer_heading_error",
                "tractor_heading_error",
            )
        )
        return BASE_FIELDS + previews

    @property
    def size(self) -> int:
        return len(BASE_FIELDS) + 3 * self.preview_count


def build_observation(
    state: VehicleState, errors: TrackingErrors, limits: VehicleLimits
) -> tuple[float, ...]:
    """Return the normalized trajectory-relative observation."""

    values = [
        errors.trailer_lateral_error / LATERAL_SCALE,
        errors.trailer_heading_error / HEADING_SCALE,
        errors.tractor_heading_error / HEADING_SCALE,
        state.articulation_angle / limits.articulation_hard_abs_max,
        state.articulation_rate / ARTICULATION_RATE_SCALE,
        state.steering_angle / limits.steering_angle_abs_max,
        state.steering_rate / limits.steering_rate_abs_max,
        errors.remaining_distance / DISTANCE_SCALE,
    ]
    for preview in errors.previews:
        values.append(preview.lateral_offset / LATERAL_SCALE)
        values.append(preview.trailer_heading_error / HEADING_SCALE)
        values.append(preview.tractor_heading_error / HEADING_SCALE)
    return tuple(values)
