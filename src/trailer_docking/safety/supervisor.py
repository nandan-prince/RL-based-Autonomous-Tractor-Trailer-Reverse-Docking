"""Hard command limits and auditable safety interventions."""

from __future__ import annotations

from dataclasses import dataclass

from trailer_docking.core.config import SimulationTiming, VehicleLimits
from trailer_docking.core.state import VehicleCommand, VehicleState


def _clip(value: float, minimum: float, maximum: float) -> float:
    return min(max(value, minimum), maximum)


@dataclass(frozen=True, slots=True)
class CommandAssessment:
    """Raw and applied commands plus every reason for an intervention."""

    requested: VehicleCommand
    applied: VehicleCommand
    reasons: tuple[str, ...]

    @property
    def intervened(self) -> bool:
        return bool(self.reasons)


def supervise_command(
    command: VehicleCommand,
    state: VehicleState,
    limits: VehicleLimits,
    timing: SimulationTiming,
) -> CommandAssessment:
    """Apply rate, angle, speed, and acceleration limits without hiding edits."""

    reasons: list[str] = []
    steering_rate = _clip(
        command.steering_rate,
        -limits.steering_rate_abs_max,
        limits.steering_rate_abs_max,
    )
    if steering_rate != command.steering_rate:
        reasons.append("steering_rate_limit")

    # Restrict the rate further so one controller interval cannot cross the
    # steering-angle bound.  Kinematics will later integrate this applied rate.
    minimum_rate = (
        -limits.steering_angle_abs_max - state.steering_angle
    ) / timing.controller_period
    maximum_rate = (
        limits.steering_angle_abs_max - state.steering_angle
    ) / timing.controller_period
    # If the measured angle is already outside its bound (possible with a
    # physical plant), the angle correction alone could demand more than the
    # rate limit.  Return toward the bound at no more than the rate limit.
    angle_safe_rate = _clip(
        _clip(steering_rate, minimum_rate, maximum_rate),
        -limits.steering_rate_abs_max,
        limits.steering_rate_abs_max,
    )
    # Record the reason only when this stage actually changed the command.
    if angle_safe_rate != steering_rate:
        reasons.append("steering_angle_limit")
    steering_rate = angle_safe_rate

    target_speed = _clip(
        command.target_speed,
        -limits.reverse_speed_abs_max,
        limits.forward_speed_max,
    )
    if target_speed != command.target_speed:
        reasons.append("speed_limit")

    speed_delta = limits.longitudinal_acceleration_abs_max * timing.controller_period
    acceleration_safe_speed = _clip(
        target_speed,
        state.longitudinal_speed - speed_delta,
        state.longitudinal_speed + speed_delta,
    )
    # A measured speed beyond the limit (possible with a physical plant) would
    # otherwise let the acceleration window pull the target outside the speed
    # limit.  The speed limit takes precedence.
    acceleration_safe_speed = _clip(
        acceleration_safe_speed,
        -limits.reverse_speed_abs_max,
        limits.forward_speed_max,
    )
    if acceleration_safe_speed != target_speed:
        reasons.append("acceleration_limit")
    target_speed = acceleration_safe_speed

    return CommandAssessment(
        requested=command,
        applied=VehicleCommand(
            steering_rate=steering_rate,
            target_speed=target_speed,
        ),
        reasons=tuple(reasons),
    )


@dataclass(frozen=True, slots=True)
class MeasurementTolerance:
    """Allowed excess of *measured* state over a plant limit.

    The exact 2D plant can be checked at numerical precision, which is the
    default.  A physical plant enforces the same limits on its commands but
    its measured steering angle, steering rate, and speed overshoot slightly
    through actuator and contact dynamics; such a backend publishes the
    tolerance its measurements need.  The jackknife threshold is a safety
    event, not a plant limit, and never receives a tolerance.
    """

    steering_angle: float = 1e-12
    steering_rate: float = 1e-12
    speed: float = 1e-12

    def __post_init__(self) -> None:
        for name in ("steering_angle", "steering_rate", "speed"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not 0.0 <= value < float("inf"):
                raise ValueError(f"{name} tolerance must be finite and non-negative")


EXACT_PLANT_TOLERANCE = MeasurementTolerance()


def state_limit_violations(
    state: VehicleState,
    limits: VehicleLimits,
    tolerance: MeasurementTolerance = EXACT_PLANT_TOLERANCE,
) -> tuple[str, ...]:
    """Return stable identifiers for hard plant/safety limit violations."""

    violations: list[str] = []
    if (
        abs(state.steering_angle)
        > limits.steering_angle_abs_max + tolerance.steering_angle
    ):
        violations.append("steering_angle_limit")
    if (
        abs(state.steering_rate)
        > limits.steering_rate_abs_max + tolerance.steering_rate
    ):
        violations.append("steering_rate_limit")
    if state.longitudinal_speed > limits.forward_speed_max + tolerance.speed:
        violations.append("forward_speed_limit")
    if state.longitudinal_speed < -limits.reverse_speed_abs_max - tolerance.speed:
        violations.append("reverse_speed_limit")
    if abs(state.articulation_angle) >= limits.articulation_hard_abs_max:
        violations.append("jackknife_limit")
    return tuple(violations)
