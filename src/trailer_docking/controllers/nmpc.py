"""Basic fixed-speed nonlinear model-predictive path tracker."""

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

try:
    import casadi as ca
except ImportError:  # pragma: no cover - depends on optional control installation
    ca = None  # type: ignore[assignment]

FloatMatrix = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class NMPCConfiguration:
    """Horizon, costs, constraints, and solver settings for basic NMPC."""

    horizon_steps: int = 40
    position_weight: float = 20.0
    trailer_heading_weight: float = 15.0
    tractor_heading_weight: float = 0.5
    articulation_weight: float = 0.8
    steering_angle_weight: float = 0.15
    steering_rate_weight: float = 0.03
    terminal_weight: float = 5.0
    solver_tolerance: float = 1e-6
    solver_max_iterations: int = 200

    def __post_init__(self) -> None:
        if self.horizon_steps <= 0:
            raise ValueError("horizon_steps must be positive")
        for name in (
            "position_weight",
            "trailer_heading_weight",
            "tractor_heading_weight",
            "articulation_weight",
            "steering_angle_weight",
            "steering_rate_weight",
            "terminal_weight",
            "solver_tolerance",
        ):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.solver_max_iterations <= 0:
            raise ValueError("solver_max_iterations must be positive")


class BasicNMPCController:
    """Receding-horizon tracker using the shared kinematics and path.

    The nonlinear program optimizes steering rate only. Longitudinal speed is
    fixed to the vehicle configuration's nominal reverse target, matching the
    one-dimensional Gymnasium/PPO action contract. Collision constraints are
    deliberately outside this first controller milestone; the environment
    still evaluates collision and terminates consistently.
    """

    def __init__(
        self,
        vehicle: VehicleConfiguration,
        configuration: NMPCConfiguration | None = None,
    ) -> None:
        if not isinstance(vehicle, VehicleConfiguration):
            raise TypeError("vehicle must be VehicleConfiguration")
        if ca is None:
            raise ModuleNotFoundError(
                "BasicNMPCController requires the 'control' extra: "
                "pip install -e '.[control]'"
            )
        self.vehicle = vehicle
        self.configuration = configuration or NMPCConfiguration()
        self._solve_failures = 0
        self._last_solve_succeeded = False
        self._last_objective: float | None = None
        self._last_controls: FloatMatrix | None = None
        self._last_states: FloatMatrix | None = None
        self._build_problem()

    @property
    def name(self) -> str:
        return "basic-nmpc-v1"

    @property
    def display_name(self) -> str:
        return "NONLINEAR MODEL PREDICTIVE CONTROL (NMPC)"

    @property
    def solve_failures(self) -> int:
        return self._solve_failures

    @property
    def last_solve_succeeded(self) -> bool:
        return self._last_solve_succeeded

    @property
    def last_objective(self) -> float | None:
        return self._last_objective

    @property
    def predicted_states(self) -> FloatMatrix | None:
        return None if self._last_states is None else self._last_states.copy()

    def reset(self) -> None:
        self._last_solve_succeeded = False
        self._last_objective = None
        self._last_controls = None
        self._last_states = None

    def _continuous_dynamics(self, state: Any, steering_rate: Any) -> Any:
        tractor = self.vehicle.tractor
        trailer = self.vehicle.trailer
        speed = self.vehicle.command.nominal_target_speed
        tractor_yaw = state[2]
        trailer_yaw = state[3]
        steering = state[4]
        tractor_rate = speed * ca.tan(steering) / tractor.wheelbase
        articulation = tractor_yaw - trailer_yaw
        hitch_speed = speed - tractor.hitch_lateral * tractor_rate
        trailer_rate = (
            hitch_speed * ca.sin(articulation)
            + tractor.hitch_longitudinal * tractor_rate * ca.cos(articulation)
        ) / trailer.hitch_to_axle
        return ca.vertcat(
            speed * ca.cos(tractor_yaw),
            speed * ca.sin(tractor_yaw),
            tractor_rate,
            trailer_rate,
            steering_rate,
        )

    def _rk4(self, state: Any, steering_rate: Any) -> Any:
        timestep = self.vehicle.timing.controller_period
        first = self._continuous_dynamics(state, steering_rate)
        second = self._continuous_dynamics(
            state + timestep * first / 2.0, steering_rate
        )
        third = self._continuous_dynamics(
            state + timestep * second / 2.0, steering_rate
        )
        fourth = self._continuous_dynamics(state + timestep * third, steering_rate)
        return state + timestep * (
            first + 2.0 * second + 2.0 * third + fourth
        ) / 6.0

    def _docking_point(self, state: Any) -> tuple[Any, Any]:
        tractor = self.vehicle.tractor
        trailer = self.vehicle.trailer
        hitch_x = (
            state[0]
            + tractor.hitch_longitudinal * ca.cos(state[2])
            - tractor.hitch_lateral * ca.sin(state[2])
        )
        hitch_y = (
            state[1]
            + tractor.hitch_longitudinal * ca.sin(state[2])
            + tractor.hitch_lateral * ca.cos(state[2])
        )
        rear_distance = trailer.hitch_to_axle + trailer.rear_overhang
        return (
            hitch_x - rear_distance * ca.cos(state[3]),
            hitch_y - rear_distance * ca.sin(state[3]),
        )

    def _stage_cost(self, state: Any, reference: Any, multiplier: float) -> Any:
        config = self.configuration
        docking_x, docking_y = self._docking_point(state)
        position = (docking_x - reference[0]) ** 2 + (
            docking_y - reference[1]
        ) ** 2
        trailer_heading = 2.0 * (1.0 - ca.cos(state[3] - reference[2]))
        tractor_heading = 2.0 * (1.0 - ca.cos(state[2] - reference[2]))
        articulation = 2.0 * (1.0 - ca.cos(state[2] - state[3]))
        return multiplier * (
            config.position_weight * position
            + config.trailer_heading_weight * trailer_heading
            + config.tractor_heading_weight * tractor_heading
            + config.articulation_weight * articulation
            + config.steering_angle_weight * state[4] ** 2
        )

    def _build_problem(self) -> None:
        config = self.configuration
        horizon = config.horizon_steps
        optimizer = ca.Opti()
        states = optimizer.variable(5, horizon + 1)
        controls = optimizer.variable(1, horizon)
        initial = optimizer.parameter(5)
        references = optimizer.parameter(3, horizon + 1)

        optimizer.subject_to(states[:, 0] == initial)
        steering_limit = self.vehicle.limits.steering_angle_abs_max
        rate_limit = self.vehicle.limits.steering_rate_abs_max
        articulation_limit = self.vehicle.limits.articulation_hard_abs_max
        optimizer.subject_to(
            optimizer.bounded(-steering_limit, states[4, :], steering_limit)
        )
        optimizer.subject_to(optimizer.bounded(-rate_limit, controls, rate_limit))
        optimizer.subject_to(
            optimizer.bounded(
                -articulation_limit,
                states[2, :] - states[3, :],
                articulation_limit,
            )
        )

        objective: Any = 0.0
        for index in range(horizon):
            optimizer.subject_to(
                states[:, index + 1]
                == self._rk4(states[:, index], controls[0, index])
            )
            objective += self._stage_cost(
                states[:, index], references[:, index], 1.0
            )
            objective += config.steering_rate_weight * controls[0, index] ** 2
        objective += self._stage_cost(
            states[:, horizon],
            references[:, horizon],
            config.terminal_weight,
        )
        optimizer.minimize(objective)
        optimizer.solver(
            "ipopt",
            {"expand": True, "print_time": False},
            {
                "print_level": 0,
                "sb": "yes",
                "max_iter": config.solver_max_iterations,
                "tol": config.solver_tolerance,
                "acceptable_tol": config.solver_tolerance * 10.0,
            },
        )
        self._optimizer = optimizer
        self._state_variables = states
        self._control_variables = controls
        self._initial_parameter = initial
        self._reference_parameters = references
        self._objective = objective

    def _references(self, context: ControllerContext) -> FloatMatrix:
        horizon = self.configuration.horizon_steps
        distance_per_step = abs(self.vehicle.command.nominal_target_speed) * (
            self.vehicle.timing.controller_period
        )
        references = np.empty((3, horizon + 1), dtype=np.float64)
        for index in range(horizon + 1):
            progress = max(
                context.tracking.progress - index * distance_per_step,
                0.0,
            )
            sample = context.path.sample(progress)
            references[:, index] = (
                sample.point.x,
                sample.point.y,
                sample.heading,
            )
        return references

    def _initial_guess(self, state: FloatMatrix) -> None:
        horizon = self.configuration.horizon_steps
        if self._last_controls is None or self._last_states is None:
            self._optimizer.set_initial(
                self._control_variables,
                np.zeros((1, horizon), dtype=np.float64),
            )
            self._optimizer.set_initial(
                self._state_variables,
                np.repeat(state.reshape(5, 1), horizon + 1, axis=1),
            )
            return
        shifted_controls = np.hstack(
            (self._last_controls[:, 1:], self._last_controls[:, -1:])
        )
        shifted_states = np.hstack(
            (self._last_states[:, 1:], self._last_states[:, -1:])
        )
        self._optimizer.set_initial(self._control_variables, shifted_controls)
        self._optimizer.set_initial(self._state_variables, shifted_states)

    def act(self, context: ControllerContext) -> ControllerAction:
        if context.vehicle.effective_fingerprint != self.vehicle.effective_fingerprint:
            raise ControllerError("controller and environment vehicle do not match")
        state = np.array(
            [
                context.state.tractor_pose.x,
                context.state.tractor_pose.y,
                context.state.tractor_pose.yaw,
                context.state.trailer_pose.yaw,
                context.state.steering_angle,
            ],
            dtype=np.float64,
        )
        self._optimizer.set_value(self._initial_parameter, state)
        self._optimizer.set_value(self._reference_parameters, self._references(context))
        self._initial_guess(state)
        try:
            solution = self._optimizer.solve()
        except RuntimeError:
            self._solve_failures += 1
            self._last_solve_succeeded = False
            self._last_objective = None
            return normalized_steering_rate(0.0, self.vehicle)
        controls = np.asarray(
            solution.value(self._control_variables), dtype=np.float64
        ).reshape(1, self.configuration.horizon_steps)
        states = np.asarray(
            solution.value(self._state_variables), dtype=np.float64
        ).reshape(5, self.configuration.horizon_steps + 1)
        self._last_controls = controls
        self._last_states = states
        self._last_objective = float(solution.value(self._objective))
        self._last_solve_succeeded = True
        return normalized_steering_rate(float(controls[0, 0]), self.vehicle)
