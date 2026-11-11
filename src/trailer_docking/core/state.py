"""Canonical simulator-independent state and command definitions.

The objects in this module contain no vehicle dynamics.  They define the values
that every simulator backend and controller exchanges, including their units
and sign conventions.  Derived articulation values are properties so callers
cannot construct internally contradictory state.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import isfinite, pi
from typing import Any


class StateValidationError(ValueError):
    """Raised when a state or command contains invalid primitive values."""


def _finite_float(value: float, field_name: str) -> float:
    """Return *value* as a float after rejecting booleans and non-finite data."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StateValidationError(f"{field_name} must be a real number")
    converted = float(value)
    if not isfinite(converted):
        raise StateValidationError(f"{field_name} must be finite")
    return converted


def wrap_angle(angle: float) -> float:
    """Wrap an angle in radians to the half-open interval ``[-pi, pi)``."""

    value = _finite_float(angle, "angle")
    wrapped = (value + pi) % (2.0 * pi) - pi
    # Do not expose negative zero in serialized states or diagnostics.
    return 0.0 if wrapped == 0.0 else wrapped


@dataclass(frozen=True, slots=True)
class Point2D:
    """A finite point in a planar Cartesian frame, measured in metres."""

    x: float
    y: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _finite_float(self.x, "Point2D.x"))
        object.__setattr__(self, "y", _finite_float(self.y, "Point2D.y"))

    def to_dict(self) -> dict[str, float]:
        """Return a JSON/YAML-compatible representation."""

        return {"x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Point2D:
        """Construct a point from an exact ``x``/``y`` mapping."""

        if set(data) != {"x", "y"}:
            raise StateValidationError("Point2D mapping must contain only x and y")
        return cls(x=data["x"], y=data["y"])


@dataclass(frozen=True, slots=True)
class Pose2D:
    """Position and counterclockwise yaw in a planar Cartesian frame."""

    x: float
    y: float
    yaw: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _finite_float(self.x, "Pose2D.x"))
        object.__setattr__(self, "y", _finite_float(self.y, "Pose2D.y"))
        object.__setattr__(self, "yaw", wrap_angle(self.yaw))

    @property
    def position(self) -> Point2D:
        """Return the positional component of this pose."""

        return Point2D(self.x, self.y)

    def to_dict(self) -> dict[str, float]:
        """Return a JSON/YAML-compatible representation."""

        return {"x": self.x, "y": self.y, "yaw": self.yaw}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Pose2D:
        """Construct a pose from an exact ``x``/``y``/``yaw`` mapping."""

        if set(data) != {"x", "y", "yaw"}:
            raise StateValidationError("Pose2D mapping must contain only x, y, and yaw")
        return cls(x=data["x"], y=data["y"], yaw=data["yaw"])


@dataclass(frozen=True, slots=True)
class VehicleCommand:
    """High-level plant command shared by 2D and MuJoCo backends.

    This type validates representation only.  It deliberately does not clip
    controller output; the safety supervisor will compare raw commands with
    vehicle limits and record any intervention.
    """

    steering_rate: float
    target_speed: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "steering_rate",
            _finite_float(self.steering_rate, "VehicleCommand.steering_rate"),
        )
        object.__setattr__(
            self,
            "target_speed",
            _finite_float(self.target_speed, "VehicleCommand.target_speed"),
        )

    def to_dict(self) -> dict[str, float]:
        """Return a JSON/YAML-compatible representation."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VehicleCommand:
        """Construct a command without accepting silent extra fields."""

        if set(data) != {"steering_rate", "target_speed"}:
            raise StateValidationError(
                "VehicleCommand mapping must contain only steering_rate and "
                "target_speed"
            )
        return cls(
            steering_rate=data["steering_rate"],
            target_speed=data["target_speed"],
        )


@dataclass(frozen=True, slots=True)
class VehicleState:
    """Complete planar state exposed by either simulator backend.

    Tractor pose is measured at the rear-axle midpoint and trailer pose at the
    trailer-axle midpoint.  Longitudinal speed is measured at the tractor rear
    axle and is positive forward.  All rates use radians per second.
    """

    tractor_pose: Pose2D
    trailer_pose: Pose2D
    hitch_position: Point2D
    steering_angle: float
    steering_rate: float
    longitudinal_speed: float
    tractor_yaw_rate: float
    trailer_yaw_rate: float
    simulation_time: float

    def __post_init__(self) -> None:
        if not isinstance(self.tractor_pose, Pose2D):
            raise StateValidationError("VehicleState.tractor_pose must be Pose2D")
        if not isinstance(self.trailer_pose, Pose2D):
            raise StateValidationError("VehicleState.trailer_pose must be Pose2D")
        if not isinstance(self.hitch_position, Point2D):
            raise StateValidationError("VehicleState.hitch_position must be Point2D")
        for field_name in (
            "steering_angle",
            "steering_rate",
            "longitudinal_speed",
            "tractor_yaw_rate",
            "trailer_yaw_rate",
            "simulation_time",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite_float(getattr(self, field_name), f"VehicleState.{field_name}"),
            )
        if self.simulation_time < 0.0:
            raise StateValidationError(
                "VehicleState.simulation_time cannot be negative"
            )

    @property
    def articulation_angle(self) -> float:
        """Return ``wrap(tractor_yaw - trailer_yaw)`` in radians."""

        return wrap_angle(self.tractor_pose.yaw - self.trailer_pose.yaw)

    @property
    def articulation_rate(self) -> float:
        """Return tractor yaw rate minus trailer yaw rate."""

        return self.tractor_yaw_rate - self.trailer_yaw_rate

    def to_dict(self, *, include_derived: bool = True) -> dict[str, Any]:
        """Return a stable JSON/YAML-compatible state representation."""

        result: dict[str, Any] = {
            "tractor_pose": self.tractor_pose.to_dict(),
            "trailer_pose": self.trailer_pose.to_dict(),
            "hitch_position": self.hitch_position.to_dict(),
            "steering_angle": self.steering_angle,
            "steering_rate": self.steering_rate,
            "longitudinal_speed": self.longitudinal_speed,
            "tractor_yaw_rate": self.tractor_yaw_rate,
            "trailer_yaw_rate": self.trailer_yaw_rate,
            "simulation_time": self.simulation_time,
        }
        if include_derived:
            result["articulation_angle"] = self.articulation_angle
            result["articulation_rate"] = self.articulation_rate
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VehicleState:
        """Deserialize primitive fields and verify optional derived values."""

        primitive_fields = {
            "tractor_pose",
            "trailer_pose",
            "hitch_position",
            "steering_angle",
            "steering_rate",
            "longitudinal_speed",
            "tractor_yaw_rate",
            "trailer_yaw_rate",
            "simulation_time",
        }
        derived_fields = {"articulation_angle", "articulation_rate"}
        keys = set(data)
        if not primitive_fields <= keys or keys - primitive_fields - derived_fields:
            missing = sorted(primitive_fields - keys)
            extra = sorted(keys - primitive_fields - derived_fields)
            raise StateValidationError(
                f"Invalid VehicleState mapping; missing={missing}, extra={extra}"
            )

        state = cls(
            tractor_pose=Pose2D.from_dict(data["tractor_pose"]),
            trailer_pose=Pose2D.from_dict(data["trailer_pose"]),
            hitch_position=Point2D.from_dict(data["hitch_position"]),
            steering_angle=data["steering_angle"],
            steering_rate=data["steering_rate"],
            longitudinal_speed=data["longitudinal_speed"],
            tractor_yaw_rate=data["tractor_yaw_rate"],
            trailer_yaw_rate=data["trailer_yaw_rate"],
            simulation_time=data["simulation_time"],
        )
        tolerance = 1e-12
        if (
            "articulation_angle" in data
            and abs(wrap_angle(data["articulation_angle"] - state.articulation_angle))
            > tolerance
        ):
            raise StateValidationError("Serialized articulation_angle is inconsistent")
        if (
            "articulation_rate" in data
            and abs(
                _finite_float(data["articulation_rate"], "articulation_rate")
                - state.articulation_rate
            )
            > tolerance
        ):
            raise StateValidationError("Serialized articulation_rate is inconsistent")
        return state
