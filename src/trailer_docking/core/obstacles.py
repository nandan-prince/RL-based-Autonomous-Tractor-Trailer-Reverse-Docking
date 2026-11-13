"""Static rectangular obstacles shared by both simulators.

An obstacle is a box standing on the yard floor, described in the dock frame
(origin at the bay's rear-wall centre, ``+x`` out of the bay).  The 2D task
collides the chassis footprints with its planar rectangle; the MuJoCo scene
contains the same box as a physical and visible body.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import cos, isfinite, sin
from numbers import Real
from typing import Any

from trailer_docking.core.state import Point2D, wrap_angle


class ObstacleError(ValueError):
    """Raised when an obstacle description is not a usable box."""


@dataclass(frozen=True, slots=True)
class Obstacle:
    """One box: centre ``(x, y)``, ``length`` along its own x axis, ``width``
    across it, rotated ``yaw`` radians counterclockwise, ``height`` tall."""

    x: float
    y: float
    length: float
    width: float
    yaw: float = 0.0
    height: float = 1.5

    def __post_init__(self) -> None:
        for name in ("x", "y", "length", "width", "yaw", "height"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not isfinite(value)
            ):
                raise ObstacleError(f"obstacle {name} must be a finite number")
            object.__setattr__(self, name, float(value))
        for name in ("length", "width", "height"):
            if getattr(self, name) <= 0.0:
                raise ObstacleError(f"obstacle {name} must be positive")
        object.__setattr__(self, "yaw", wrap_angle(self.yaw))

    def polygon(self) -> tuple[Point2D, ...]:
        """Corners of the planar rectangle, counterclockwise."""

        half_length, half_width = self.length / 2.0, self.width / 2.0
        cosine, sine = cos(self.yaw), sin(self.yaw)
        return tuple(
            Point2D(
                self.x + dx * cosine - dy * sine,
                self.y + dx * sine + dy * cosine,
            )
            for dx, dy in (
                (-half_length, -half_width),
                (half_length, -half_width),
                (half_length, half_width),
                (-half_length, half_width),
            )
        )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Any) -> Obstacle:
        if not isinstance(data, dict):
            raise ObstacleError("an obstacle must be a mapping")
        try:
            return cls(**data)
        except TypeError as error:
            raise ObstacleError(f"invalid obstacle: {error}") from error


def obstacle_name(index: int) -> str:
    """Stable name of the obstacle at *index* (zero-based) in reports."""

    return f"obstacle_{index + 1}"
