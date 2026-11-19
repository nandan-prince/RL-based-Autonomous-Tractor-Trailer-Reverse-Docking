"""Trajectory-error LQR used as the first closed-loop control baseline."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np
from numpy.typing import NDArray

from trailer_docking.controllers.base import (
    ControllerAction,
    ControllerContext,
    ControllerError,
    normalized_steering_rate,
)
from trailer_docking.core.config import VehicleConfiguration
from trailer_docking.simulators.kinematic.model import (
    KinematicInput,
    KinematicModel,
    PlanarKinematicState,
)

FloatMatrix = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class LQRConfiguration:
    """Weights for straight-path error stabilization at fixed reverse speed."""

    lateral_weight: float = 4.0
    tractor_heading_weight: float = 0.8
    trailer_heading_weight: float = 6.0
    steering_angle_weight: float = 0.3
    steering_rate_weight: float = 0.4
    linearization_epsilon: float = 1e-5
    riccati_tolerance: float = 1e-11
    riccati_max_iterations: int = 10_000

    def __post_init__(self) -> None:
        for name in (
            "lateral_weight",
            "tractor_heading_weight",
            "trailer_heading_weight",
            "steering_angle_weight",
            "steering_rate_weight",
            "linearization_epsilon",
            "riccati_tolerance",
        ):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.riccati_max_iterations <= 0:
            raise ValueError("riccati_max_iterations must be positive")


def _solve_discrete_lqr(
    system: FloatMatrix,
    control: FloatMatrix,
    state_cost: FloatMatrix,
    input_cost: FloatMatrix,
    *,
    tolerance: float,
    max_iterations: int,
) -> FloatMatrix:
    """Solve the discrete algebraic Riccati equation by fixed-point iteration."""

    value = state_cost.copy()
    for _ in range(max_iterations):
        inverse_term = input_cost + control.T @ value @ control
        gain = np.linalg.solve(inverse_term, control.T @ value @ system)
        updated = (
            state_cost
            + system.T @ value @ system
            - system.T @ value @ control @ gain
        )
        updated = (updated + updated.T) / 2.0
        if np.linalg.norm(updated - value, ord=np.inf) <= tolerance:
            return np.asarray(
                np.linalg.solve(
                    input_cost + control.T @ updated @ control,
                    control.T @ updated @ system,
                ),
                dtype=np.float64,
            )
        value = updated
    raise ControllerError("LQR Riccati iteration did not converge")


class TrajectoryLQRController:
    """Discrete LQR over trailer lateral/heading and steering errors.

    The gain is obtained from a numerical linearization of the same RK4 model
    used by the environment. The error vector is ordered as trailer rear-face
    lateral error, tractor heading error, trailer heading error, and steering
    angle. The optimized input is physical steering rate; the public action is
    the shared normalized one-element Gymnasium action.
    """

    def __init__(
        self,
        vehicle: VehicleConfiguration,
        configuration: LQRConfiguration | None = None,
    ) -> None:
        if not isinstance(vehicle, VehicleConfiguration):
            raise TypeError("vehicle must be VehicleConfiguration")
        self.vehicle = vehicle
        self.configuration = configuration or LQRConfiguration()
        self._model = KinematicModel(vehicle.geometry)
        self._system, self._control = self._linearize()
        config = self.configuration
        state_cost = np.diag(
            [
                config.lateral_weight,
                config.tractor_heading_weight,
                config.trailer_heading_weight,
                config.steering_angle_weight,
            ]
        )
        input_cost = np.array([[config.steering_rate_weight]], dtype=np.float64)
        self._gain = _solve_discrete_lqr(
            self._system,
            self._control,
            state_cost,
            input_cost,
            tolerance=config.riccati_tolerance,
            max_iterations=config.riccati_max_iterations,
        )

    @property
    def name(self) -> str:
        return "trajectory-lqr-v1"

    @property
    def display_name(self) -> str:
        return "LQR TRAJECTORY TRACKER"

    @property
    def gain(self) -> FloatMatrix:
        """Return a defensive copy of the computed steering-rate gain."""

        return self._gain.copy()

    @property
    def linear_model(self) -> tuple[FloatMatrix, FloatMatrix]:
        """Return defensive copies of the discrete error dynamics."""

        return self._system.copy(), self._control.copy()

    def reset(self) -> None:
        """LQR has no episode memory."""

    def _state_from_error(self, error: FloatMatrix) -> PlanarKinematicState:
        trailer_yaw = float(error[2])
        tractor_yaw = float(error[1])
        steering = float(error[3])
        provisional = PlanarKinematicState(
            tractor_x=0.0,
            tractor_y=0.0,
            tractor_yaw=tractor_yaw,
            trailer_yaw=trailer_yaw,
            steering_angle=steering,
            longitudinal_speed=self.vehicle.command.nominal_target_speed,
        )
        shared = self._model.shared_state(provisional, steering_rate=0.0)
        rear = self.vehicle.trailer.rear_face(shared.trailer_pose)
        return PlanarKinematicState(
            tractor_x=0.0,
            tractor_y=float(error[0]) - rear.y,
            tractor_yaw=tractor_yaw,
            trailer_yaw=trailer_yaw,
            steering_angle=steering,
            longitudinal_speed=self.vehicle.command.nominal_target_speed,
        )

    def _transition(self, error: FloatMatrix, steering_rate: float) -> FloatMatrix:
        state = self._state_from_error(error)
        integration_step = self.vehicle.timing.kinematic_integration_step
        control = KinematicInput(steering_rate, 0.0)
        for _ in range(self.vehicle.timing.kinematic_substeps_per_control):
            state = self._model.rk4_step(state, control, integration_step)
        shared = self._model.shared_state(state, steering_rate=steering_rate)
        docking = self.vehicle.trailer.rear_face(shared.trailer_pose)
        return np.array(
            [
                docking.y,
                state.tractor_yaw,
                state.trailer_yaw,
                state.steering_angle,
            ],
            dtype=np.float64,
        )

    def _linearize(self) -> tuple[FloatMatrix, FloatMatrix]:
        epsilon = self.configuration.linearization_epsilon
        origin = np.zeros(4, dtype=np.float64)
        system = np.empty((4, 4), dtype=np.float64)
        for index in range(4):
            offset = np.zeros(4, dtype=np.float64)
            offset[index] = epsilon
            system[:, index] = (
                self._transition(origin + offset, 0.0)
                - self._transition(origin - offset, 0.0)
            ) / (2.0 * epsilon)
        control = (
            self._transition(origin, epsilon)
            - self._transition(origin, -epsilon)
        ).reshape(4, 1) / (2.0 * epsilon)
        return system, control

    def act(self, context: ControllerContext) -> ControllerAction:
        if context.vehicle.effective_fingerprint != self.vehicle.effective_fingerprint:
            raise ControllerError("controller and environment vehicle do not match")
        error = np.array(
            [
                context.tracking.lateral_error,
                context.tracking.tractor_heading_error,
                context.tracking.trailer_heading_error,
                context.state.steering_angle,
            ],
            dtype=np.float64,
        )
        steering_rate = -float((self._gain @ error)[0])
        return normalized_steering_rate(steering_rate, self.vehicle)


def make_docking_controller(env: Any) -> TrajectoryLQRController:
    """Build the LQR tracker for the episode's resolved vehicle geometry."""

    return TrajectoryLQRController(env.built.vehicle)
