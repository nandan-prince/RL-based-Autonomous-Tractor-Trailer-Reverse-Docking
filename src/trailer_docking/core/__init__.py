"""Simulator-independent domain types and mathematical definitions."""

from trailer_docking.core.config import (
    CommandConfiguration,
    ConfigurationError,
    DockConfiguration,
    SimulationTiming,
    VehicleConfiguration,
    VehicleLimits,
    load_dock_configuration,
    load_vehicle_configuration,
)
from trailer_docking.core.geometry import (
    DockGeometry,
    TractorGeometry,
    TrailerGeometry,
    VehicleGeometry,
    WheelGeometry,
)
from trailer_docking.core.state import (
    Point2D,
    Pose2D,
    StateValidationError,
    VehicleCommand,
    VehicleState,
    wrap_angle,
)

__all__ = [
    "CommandConfiguration",
    "ConfigurationError",
    "DockConfiguration",
    "DockGeometry",
    "Point2D",
    "Pose2D",
    "SimulationTiming",
    "StateValidationError",
    "TractorGeometry",
    "TrailerGeometry",
    "VehicleCommand",
    "VehicleConfiguration",
    "VehicleGeometry",
    "VehicleLimits",
    "VehicleState",
    "WheelGeometry",
    "load_dock_configuration",
    "load_vehicle_configuration",
    "wrap_angle",
]
