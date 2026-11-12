"""Time integration and state management for the 2D simulator."""

from __future__ import annotations

from math import isclose

from trailer_docking.core.config import VehicleConfiguration
from trailer_docking.core.state import Pose2D, VehicleCommand, VehicleState, wrap_angle
from trailer_docking.safety.supervisor import CommandAssessment, supervise_command
from trailer_docking.simulators.kinematic.model import (
    KinematicInput,
    KinematicModel,
    KinematicModelError,
    PlanarKinematicState,
)


def _clip(value: float, minimum: float, maximum: float) -> float:
    return min(max(value, minimum), maximum)


class KinematicSimulator:
    """Exact-limit, deterministic 2D backend for control and RL workloads."""

    def __init__(self, vehicle: VehicleConfiguration) -> None:
        if not isinstance(vehicle, VehicleConfiguration):
            raise KinematicModelError("vehicle must be VehicleConfiguration")
        if not isclose(vehicle.timing.nominal_command_delay, 0.0, abs_tol=1e-12):
            raise NotImplementedError(
                "a nonzero command delay is not implemented in the 2D backend"
            )
        self.vehicle = vehicle
        self.model = KinematicModel(vehicle.geometry)
        self._internal = PlanarKinematicState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        stopped = VehicleCommand(0.0, 0.0)
        self._assessment = CommandAssessment(stopped, stopped, ())
        self._state = self.model.shared_state(self._internal, steering_rate=0.0)

    @property
    def controller_period(self) -> float:
        return self.vehicle.timing.controller_period

    @property
    def state(self) -> VehicleState:
        return self._state

    @property
    def applied_command(self) -> VehicleCommand:
        return self._assessment.applied

    @property
    def command_assessment(self) -> CommandAssessment:
        """Requested/applied command and auditable intervention reasons."""

        return self._assessment

    def reset(
        self,
        tractor_pose: Pose2D | None = None,
        trailer_yaw: float | None = None,
        *,
        steering_angle: float = 0.0,
    ) -> VehicleState:
        """Reset at rest in a geometrically consistent articulated pose."""

        pose = Pose2D(0.0, 0.0, 0.0) if tractor_pose is None else tractor_pose
        if not isinstance(pose, Pose2D):
            raise KinematicModelError("tractor_pose must be a Pose2D")
        yaw = pose.yaw if trailer_yaw is None else wrap_angle(trailer_yaw)
        candidate = PlanarKinematicState(
            tractor_x=pose.x,
            tractor_y=pose.y,
            tractor_yaw=pose.yaw,
            trailer_yaw=yaw,
            steering_angle=steering_angle,
            longitudinal_speed=0.0,
            simulation_time=0.0,
        )
        limit = self.vehicle.limits.steering_angle_abs_max
        if not -limit <= candidate.steering_angle <= limit:
            raise KinematicModelError(
                f"steering_angle={candidate.steering_angle} exceeds the"
                f" +-{limit} rad limit"
            )
        self._internal = candidate
        stopped = VehicleCommand(0.0, 0.0)
        self._assessment = CommandAssessment(stopped, stopped, ())
        self._state = self.model.shared_state(self._internal, steering_rate=0.0)
        return self._state

    def step(self, command: VehicleCommand) -> VehicleState:
        """Advance exactly one configured controller period."""

        if not isinstance(command, VehicleCommand):
            raise KinematicModelError("command must be a VehicleCommand")
        assessment = supervise_command(
            command, self._state, self.vehicle.limits, self.vehicle.timing
        )
        self._assessment = assessment
        dt = self.vehicle.timing.kinematic_integration_step
        limits = self.vehicle.limits
        target_speed = assessment.applied.target_speed
        for _ in range(self.vehicle.timing.kinematic_substeps_per_control):
            acceleration = _clip(
                (target_speed - self._internal.longitudinal_speed) / dt,
                -limits.longitudinal_acceleration_abs_max,
                limits.longitudinal_acceleration_abs_max,
            )
            control = KinematicInput(
                steering_rate=assessment.applied.steering_rate,
                longitudinal_acceleration=acceleration,
            )
            self._internal = self.model.rk4_step(self._internal, control, dt)

        # Eliminate last-bit accumulation at exact ideal-plant bounds.
        self._internal = PlanarKinematicState(
            tractor_x=self._internal.tractor_x,
            tractor_y=self._internal.tractor_y,
            tractor_yaw=self._internal.tractor_yaw,
            trailer_yaw=self._internal.trailer_yaw,
            steering_angle=_clip(
                self._internal.steering_angle,
                -limits.steering_angle_abs_max,
                limits.steering_angle_abs_max,
            ),
            longitudinal_speed=target_speed,
            simulation_time=self._internal.simulation_time,
        )
        self._state = self.model.shared_state(
            self._internal, steering_rate=assessment.applied.steering_rate
        )
        return self._state
