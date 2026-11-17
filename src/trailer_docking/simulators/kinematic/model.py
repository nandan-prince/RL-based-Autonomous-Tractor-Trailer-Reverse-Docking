"""Canonical low-speed tractor-semitrailer kinematic equations.

Only independent generalized coordinates are integrated.  Hitch and trailer
axle positions are reconstructed by :class:`VehicleGeometry`, so numerical
integration cannot violate the rigid connection.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, sin, tan
from numbers import Real

from trailer_docking.core.geometry import VehicleGeometry
from trailer_docking.core.state import Pose2D, VehicleState, wrap_angle


class KinematicModelError(ValueError):
    """Raised when a kinematic state, input, or integration step is invalid."""


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise KinematicModelError(f"{name} must be a real number")
    try:
        number = float(value)
    except OverflowError as error:
        raise KinematicModelError(f"{name} must be finite") from error
    if not isfinite(number):
        raise KinematicModelError(f"{name} must be finite")
    return number


@dataclass(frozen=True, slots=True)
class PlanarKinematicState:
    """Independent state integrated by the deterministic 2D plant."""

    tractor_x: float
    tractor_y: float
    tractor_yaw: float
    trailer_yaw: float
    steering_angle: float
    longitudinal_speed: float
    simulation_time: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "tractor_x",
            "tractor_y",
            "steering_angle",
            "longitudinal_speed",
            "simulation_time",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        object.__setattr__(self, "tractor_yaw", wrap_angle(self.tractor_yaw))
        object.__setattr__(self, "trailer_yaw", wrap_angle(self.trailer_yaw))
        if self.simulation_time < 0.0:
            raise KinematicModelError("simulation_time cannot be negative")


@dataclass(frozen=True, slots=True)
class KinematicInput:
    """Input held constant over one numerical integration substep."""

    steering_rate: float
    longitudinal_acceleration: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "steering_rate", _finite(self.steering_rate, "steering_rate")
        )
        object.__setattr__(
            self,
            "longitudinal_acceleration",
            _finite(self.longitudinal_acceleration, "longitudinal_acceleration"),
        )


@dataclass(frozen=True, slots=True)
class KinematicDerivative:
    """Continuous-time derivative of :class:`PlanarKinematicState`."""

    tractor_x: float
    tractor_y: float
    tractor_yaw: float
    trailer_yaw: float
    steering_angle: float
    longitudinal_speed: float


class KinematicModel:
    """General low-speed tractor--single-semitrailer model.

    The tractor rear axle and trailer axle obey no-lateral-slip constraints.
    Both signed tractor hitch offsets are included.  For longitudinal offset
    ``a``, lateral offset ``b``, tractor yaw rate ``omega``, articulation
    ``alpha``, and trailer length ``L_r``::

        trailer_yaw_rate = ((v - b*omega) sin(alpha)
                            + a*omega cos(alpha)) / L_r

    This reduces to the frozen on-axle equation when ``a = b = 0``.
    """

    def __init__(self, geometry: VehicleGeometry) -> None:
        if not isinstance(geometry, VehicleGeometry):
            raise KinematicModelError("geometry must be VehicleGeometry")
        self.geometry = geometry

    def derivative(
        self, state: PlanarKinematicState, control: KinematicInput
    ) -> KinematicDerivative:
        """Evaluate the continuous-time equations at one state and input."""

        if not isinstance(state, PlanarKinematicState):
            raise KinematicModelError("state must be PlanarKinematicState")
        if not isinstance(control, KinematicInput):
            raise KinematicModelError("control must be KinematicInput")
        tractor = self.geometry.tractor
        trailer = self.geometry.trailer
        speed = state.longitudinal_speed
        tractor_rate = speed * tan(state.steering_angle) / tractor.wheelbase
        articulation = state.tractor_yaw - state.trailer_yaw
        hitch_forward_speed = speed - tractor.hitch_lateral * tractor_rate
        trailer_rate = (
            hitch_forward_speed * sin(articulation)
            + tractor.hitch_longitudinal * tractor_rate * cos(articulation)
        ) / trailer.hitch_to_axle
        return KinematicDerivative(
            tractor_x=speed * cos(state.tractor_yaw),
            tractor_y=speed * sin(state.tractor_yaw),
            tractor_yaw=tractor_rate,
            trailer_yaw=trailer_rate,
            steering_angle=control.steering_rate,
            longitudinal_speed=control.longitudinal_acceleration,
        )

    @staticmethod
    def _offset(
        state: PlanarKinematicState,
        derivative: KinematicDerivative,
        scale: float,
    ) -> PlanarKinematicState:
        return PlanarKinematicState(
            tractor_x=state.tractor_x + scale * derivative.tractor_x,
            tractor_y=state.tractor_y + scale * derivative.tractor_y,
            tractor_yaw=state.tractor_yaw + scale * derivative.tractor_yaw,
            trailer_yaw=state.trailer_yaw + scale * derivative.trailer_yaw,
            steering_angle=(state.steering_angle + scale * derivative.steering_angle),
            longitudinal_speed=(
                state.longitudinal_speed + scale * derivative.longitudinal_speed
            ),
            simulation_time=state.simulation_time + scale,
        )

    def rk4_step(
        self,
        state: PlanarKinematicState,
        control: KinematicInput,
        timestep: float,
    ) -> PlanarKinematicState:
        """Advance one fixed step with classical fourth-order Runge--Kutta."""

        dt = _finite(timestep, "timestep")
        if dt <= 0.0:
            raise KinematicModelError("timestep must be greater than zero")
        k1 = self.derivative(state, control)
        k2 = self.derivative(self._offset(state, k1, dt / 2.0), control)
        k3 = self.derivative(self._offset(state, k2, dt / 2.0), control)
        k4 = self.derivative(self._offset(state, k3, dt), control)

        def combine(
            value: float, first: float, second: float, third: float, fourth: float
        ) -> float:
            return value + dt * (first + 2.0 * second + 2.0 * third + fourth) / 6.0

        return PlanarKinematicState(
            tractor_x=combine(
                state.tractor_x,
                k1.tractor_x,
                k2.tractor_x,
                k3.tractor_x,
                k4.tractor_x,
            ),
            tractor_y=combine(
                state.tractor_y,
                k1.tractor_y,
                k2.tractor_y,
                k3.tractor_y,
                k4.tractor_y,
            ),
            tractor_yaw=combine(
                state.tractor_yaw,
                k1.tractor_yaw,
                k2.tractor_yaw,
                k3.tractor_yaw,
                k4.tractor_yaw,
            ),
            trailer_yaw=combine(
                state.trailer_yaw,
                k1.trailer_yaw,
                k2.trailer_yaw,
                k3.trailer_yaw,
                k4.trailer_yaw,
            ),
            steering_angle=combine(
                state.steering_angle,
                k1.steering_angle,
                k2.steering_angle,
                k3.steering_angle,
                k4.steering_angle,
            ),
            longitudinal_speed=combine(
                state.longitudinal_speed,
                k1.longitudinal_speed,
                k2.longitudinal_speed,
                k3.longitudinal_speed,
                k4.longitudinal_speed,
            ),
            simulation_time=state.simulation_time + dt,
        )

    def shared_state(
        self, state: PlanarKinematicState, *, steering_rate: float
    ) -> VehicleState:
        """Reconstruct the redundant shared state from rigid geometry."""

        rate = _finite(steering_rate, "steering_rate")
        derivative = self.derivative(state, KinematicInput(rate, 0.0))
        tractor_pose = Pose2D(state.tractor_x, state.tractor_y, state.tractor_yaw)
        return self.geometry.assembled_state(
            tractor_pose,
            state.trailer_yaw,
            steering_angle=state.steering_angle,
            steering_rate=rate,
            longitudinal_speed=state.longitudinal_speed,
            tractor_yaw_rate=derivative.tractor_yaw,
            trailer_yaw_rate=derivative.trailer_yaw,
            simulation_time=state.simulation_time,
        )
