"""Controller protocols and implemented classical/model-based controllers."""

from trailer_docking.controllers.base import (
    Controller,
    ControllerAction,
    ControllerContext,
    ControllerError,
    normalized_steering_rate,
    validate_controller_action,
)
from trailer_docking.controllers.lqr import LQRConfiguration, TrajectoryLQRController
from trailer_docking.controllers.nmpc import BasicNMPCController, NMPCConfiguration
from trailer_docking.controllers.ppo import PPOController

__all__ = [
    "BasicNMPCController",
    "Controller",
    "ControllerAction",
    "ControllerContext",
    "ControllerError",
    "LQRConfiguration",
    "NMPCConfiguration",
    "PPOController",
    "TrajectoryLQRController",
    "normalized_steering_rate",
    "validate_controller_action",
]
