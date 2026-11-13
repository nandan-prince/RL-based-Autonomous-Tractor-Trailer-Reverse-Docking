"""Build bounded trajectory-relative observations for NMPC and PPO."""

from __future__ import annotations

from math import cos, sin

import numpy as np
from numpy.typing import NDArray

from trailer_docking.core.config import VehicleConfiguration
from trailer_docking.core.geometry import DockGeometry
from trailer_docking.core.state import VehicleState
from trailer_docking.paths.reference import ReferencePath
from trailer_docking.paths.tracking import TrackingState

FloatArray = NDArray[np.float32]


def _clip(value: float) -> float:
    return min(max(value, -1.0), 1.0)


class ObservationBuilder:
    """Stable, normalized observation schema with local preview coordinates."""

    def __init__(
        self,
        vehicle: VehicleConfiguration,
        dock: DockGeometry,
        path: ReferencePath,
        preview_distances: tuple[float, ...],
    ) -> None:
        if not preview_distances:
            raise ValueError("at least one preview distance is required")
        self.vehicle = vehicle
        self.dock = dock
        self.path = path
        self.preview_distances = preview_distances
        preview_names = tuple(
            name
            for index in range(len(preview_distances))
            for name in (f"preview_{index}_x", f"preview_{index}_y")
        )
        self.names = (
            "lateral_error",
            "trailer_heading_sin",
            "trailer_heading_cos",
            "tractor_heading_sin",
            "tractor_heading_cos",
            "articulation",
            "articulation_rate",
            "steering_angle",
            "steering_rate",
            "longitudinal_speed",
            "remaining_distance",
            "goal_longitudinal_error",
            "goal_lateral_error",
            "goal_heading_sin",
            "goal_heading_cos",
            *preview_names,
        )

    @property
    def shape(self) -> tuple[int]:
        return (len(self.names),)

    def build(
        self,
        state: VehicleState,
        tracking: TrackingState,
        goal_longitudinal_error: float,
        goal_lateral_error: float,
        goal_yaw_error: float,
    ) -> FloatArray:
        """Return finite float32 values clipped to the declared ``[-1, 1]`` box."""

        limits = self.vehicle.limits
        lateral_scale = max(self.dock.interior_width, 1.0)
        path_scale = max(self.path.length, 1.0)
        speed_scale = max(limits.forward_speed_max, limits.reverse_speed_abs_max)
        values = [
            _clip(tracking.lateral_error / lateral_scale),
            sin(tracking.trailer_heading_error),
            cos(tracking.trailer_heading_error),
            sin(tracking.tractor_heading_error),
            cos(tracking.tractor_heading_error),
            _clip(state.articulation_angle / limits.articulation_hard_abs_max),
            _clip(state.articulation_rate),
            _clip(state.steering_angle / limits.steering_angle_abs_max),
            _clip(state.steering_rate / limits.steering_rate_abs_max),
            _clip(state.longitudinal_speed / speed_scale),
            _clip(tracking.distance_to_goal / path_scale),
            _clip(goal_longitudinal_error / path_scale),
            _clip(goal_lateral_error / lateral_scale),
            sin(goal_yaw_error),
            cos(goal_yaw_error),
        ]
        for point in tracking.preview_points_trailer_frame:
            values.extend((_clip(point.x / path_scale), _clip(point.y / lateral_scale)))
        result = np.asarray(values, dtype=np.float32)
        if result.shape != self.shape or not np.isfinite(result).all():
            raise RuntimeError("observation builder produced an invalid vector")
        return result
