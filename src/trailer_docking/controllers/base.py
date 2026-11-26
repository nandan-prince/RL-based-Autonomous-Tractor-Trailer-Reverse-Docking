"""Shared controller contract for deterministic, optimization, and RL policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from trailer_docking.core.config import VehicleConfiguration
from trailer_docking.core.state import VehicleState
from trailer_docking.observations import FloatArray
from trailer_docking.paths import ReferencePath, TrackingState

ControllerAction = NDArray[np.float32]


class ControllerError(RuntimeError):
    """Raised when a controller cannot produce a valid environment action."""


@dataclass(frozen=True, slots=True)
class ControllerContext:
    """Identical per-step information available to all controller adapters.

    PPO adapters should normally consume only ``observation``. Model-based
    controllers may additionally use the shared state, path, and tracking
    values. The resulting environment action is identical for every method.
    """

    observation: FloatArray
    state: VehicleState
    path: ReferencePath
    tracking: TrackingState
    vehicle: VehicleConfiguration
    step_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.observation, np.ndarray):
            raise TypeError("observation must be a numpy array")
        if self.observation.dtype != np.float32 or self.observation.ndim != 1:
            raise ValueError("observation must be a one-dimensional float32 array")
        if not np.all(np.isfinite(self.observation)):
            raise ValueError("observation must contain only finite values")
        if not isinstance(self.state, VehicleState):
            raise TypeError("state must be VehicleState")
        if not isinstance(self.path, ReferencePath):
            raise TypeError("path must be ReferencePath")
        if not isinstance(self.tracking, TrackingState):
            raise TypeError("tracking must be TrackingState")
        if not isinstance(self.vehicle, VehicleConfiguration):
            raise TypeError("vehicle must be VehicleConfiguration")
        if not isinstance(self.step_index, int) or self.step_index < 0:
            raise ValueError("step_index must be a non-negative integer")


def normalized_steering_rate(
    steering_rate: float, vehicle: VehicleConfiguration
) -> ControllerAction:
    """Convert a physical steering rate to the shared one-element action."""

    maximum = vehicle.limits.steering_rate_abs_max
    normalized = float(np.clip(steering_rate / maximum, -1.0, 1.0))
    return np.array([normalized], dtype=np.float32)


def validate_controller_action(action: ControllerAction) -> ControllerAction:
    """Validate and defensively copy the fixed-reverse Gymnasium action."""

    candidate = np.asarray(action, dtype=np.float32)
    if candidate.shape != (1,) or not np.all(np.isfinite(candidate)):
        raise ControllerError("controller action must be one finite float32 value")
    if np.any(candidate < -1.0) or np.any(candidate > 1.0):
        raise ControllerError("controller action must lie in [-1, 1]")
    return candidate.copy()


@runtime_checkable
class Controller(Protocol):
    """Controller interface shared by LQR, NMPC, and PPO inference."""

    @property
    def name(self) -> str:
        """Stable method name for diagnostics and experiment manifests."""

    def reset(self) -> None:
        """Clear controller memory before a new episode."""

    def act(self, context: ControllerContext) -> ControllerAction:
        """Return one normalized steering-rate action."""
