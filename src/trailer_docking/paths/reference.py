"""Immutable polyline reference paths with arc-length interpolation."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, hypot, isfinite
from numbers import Real

from trailer_docking.core.state import Point2D


class ReferencePathError(ValueError):
    """Raised when a reference path or query is invalid."""


def _number(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ReferencePathError(f"{name} must be a real number")
    result = float(value)
    if not isfinite(result):
        raise ReferencePathError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class PathSample:
    """A point and nominal vehicle heading at arc-length ``progress``."""

    point: Point2D
    heading: float
    progress: float


@dataclass(frozen=True, slots=True)
class NearestPathPoint:
    """Orthogonal polyline projection of a query point."""

    sample: PathSample
    signed_lateral_error: float
    distance: float
    segment_index: int


@dataclass(frozen=True, slots=True)
class ReferencePath:
    """Non-degenerate polyline ordered in nominal vehicle-heading direction.

    Reverse docking traverses this path toward decreasing progress.  Ordering
    the nominal straight path from the dock target outward preserves a zero
    vehicle heading while the vehicle moves with negative speed.
    """

    points: tuple[Point2D, ...]

    def __post_init__(self) -> None:
        if len(self.points) < 2:
            raise ReferencePathError("a reference path needs at least two points")
        if not all(isinstance(point, Point2D) for point in self.points):
            raise ReferencePathError("reference path entries must be Point2D")
        if any(
            first == second
            for first, second in zip(self.points, self.points[1:], strict=False)
        ):
            raise ReferencePathError("consecutive path points must be distinct")

    @property
    def segment_lengths(self) -> tuple[float, ...]:
        return tuple(
            hypot(second.x - first.x, second.y - first.y)
            for first, second in zip(self.points, self.points[1:], strict=False)
        )

    @property
    def cumulative_lengths(self) -> tuple[float, ...]:
        values = [0.0]
        for length in self.segment_lengths:
            values.append(values[-1] + length)
        return tuple(values)

    @property
    def length(self) -> float:
        return self.cumulative_lengths[-1]

    def sample(self, progress: float) -> PathSample:
        """Interpolate at arc length, clamping outside the path endpoints."""

        requested = _number(progress, "progress")
        value = min(max(requested, 0.0), self.length)
        cumulative = self.cumulative_lengths
        lengths = self.segment_lengths
        index = len(lengths) - 1
        for candidate, end in enumerate(cumulative[1:]):
            if value <= end:
                index = candidate
                break
        first, second = self.points[index], self.points[index + 1]
        ratio = (value - cumulative[index]) / lengths[index]
        return PathSample(
            point=Point2D(
                first.x + ratio * (second.x - first.x),
                first.y + ratio * (second.y - first.y),
            ),
            heading=atan2(second.y - first.y, second.x - first.x),
            progress=value,
        )

    def nearest(self, point: Point2D) -> NearestPathPoint:
        """Return the globally nearest projection and signed lateral error."""

        if not isinstance(point, Point2D):
            raise ReferencePathError("query must be Point2D")
        cumulative = self.cumulative_lengths
        best: NearestPathPoint | None = None
        for index, (first, second) in enumerate(
            zip(self.points, self.points[1:], strict=False)
        ):
            dx, dy = second.x - first.x, second.y - first.y
            squared = dx * dx + dy * dy
            projection = ((point.x - first.x) * dx + (point.y - first.y) * dy) / squared
            ratio = min(max(projection, 0.0), 1.0)
            projected = Point2D(first.x + ratio * dx, first.y + ratio * dy)
            error_x, error_y = point.x - projected.x, point.y - projected.y
            distance = hypot(error_x, error_y)
            length = self.segment_lengths[index]
            signed = (dx * error_y - dy * error_x) / length
            candidate = NearestPathPoint(
                sample=PathSample(
                    projected,
                    atan2(dy, dx),
                    cumulative[index] + ratio * length,
                ),
                signed_lateral_error=signed,
                distance=distance,
                segment_index=index,
            )
            if best is None or candidate.distance < best.distance:
                best = candidate
        if best is None:  # pragma: no cover - constructor guarantees a segment
            raise AssertionError("reference path has no segments")
        return best

    def previews(
        self, progress: float, distances: tuple[float, ...]
    ) -> tuple[PathSample, ...]:
        """Sample distances toward the docking goal (decreasing progress)."""

        current = _number(progress, "progress")
        checked: list[float] = []
        for distance in distances:
            value = _number(distance, "preview distance")
            if value < 0.0:
                raise ReferencePathError("preview distances cannot be negative")
            checked.append(value)
        return tuple(self.sample(current - distance) for distance in checked)
