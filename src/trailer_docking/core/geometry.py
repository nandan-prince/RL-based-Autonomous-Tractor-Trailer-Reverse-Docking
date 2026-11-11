"""Canonical planar geometry and semantic-frame transformations."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import cos, hypot, isclose, isfinite, sin

from trailer_docking.core.state import Point2D, Pose2D, VehicleState, wrap_angle


class GeometryValidationError(ValueError):
    """Raised when dimensions or geometric relationships are invalid."""


def _is_real(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _positive(value: float, name: str) -> float:
    if not _is_real(value) or not isfinite(value) or value <= 0.0:
        raise GeometryValidationError(f"{name} must be finite and greater than zero")
    return value


@dataclass(frozen=True, slots=True)
class Bounds1D:
    """Closed one-dimensional interval."""

    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        if not (_is_real(self.minimum) and _is_real(self.maximum)):
            raise GeometryValidationError("bounds must be real numbers")
        if not isfinite(self.minimum) or not isfinite(self.maximum):
            raise GeometryValidationError("bounds must be finite")
        if self.minimum >= self.maximum:
            raise GeometryValidationError("bounds require minimum < maximum")

    @property
    def span(self) -> float:
        return self.maximum - self.minimum


def rotate(point: Point2D, yaw: float) -> Point2D:
    """Rotate a point counterclockwise around the origin."""

    cosine = cos(yaw)
    sine = sin(yaw)
    return Point2D(
        x=cosine * point.x - sine * point.y,
        y=sine * point.x + cosine * point.y,
    )


def local_to_world(point: Point2D, frame: Pose2D) -> Point2D:
    """Transform a point from a body-local frame to the world frame."""

    rotated = rotate(point, frame.yaw)
    return Point2D(frame.x + rotated.x, frame.y + rotated.y)


def world_to_local(point: Point2D, frame: Pose2D) -> Point2D:
    """Transform a world-frame point into a body-local frame."""

    translated = Point2D(point.x - frame.x, point.y - frame.y)
    return rotate(translated, -frame.yaw)


def rectangle_corners(
    x_bounds: Bounds1D, y_bounds: Bounds1D, frame: Pose2D
) -> tuple[Point2D, Point2D, Point2D, Point2D]:
    """Return counterclockwise world-frame corners of a local rectangle."""

    local_corners = (
        Point2D(x_bounds.minimum, y_bounds.minimum),
        Point2D(x_bounds.maximum, y_bounds.minimum),
        Point2D(x_bounds.maximum, y_bounds.maximum),
        Point2D(x_bounds.minimum, y_bounds.maximum),
    )
    transformed = tuple(local_to_world(corner, frame) for corner in local_corners)
    return transformed[0], transformed[1], transformed[2], transformed[3]


def polygon_area(vertices: Iterable[Point2D]) -> float:
    """Return signed polygon area; positive values indicate CCW ordering."""

    points = tuple(vertices)
    if len(points) < 3:
        raise GeometryValidationError("a polygon requires at least three vertices")
    twice_area = sum(
        first.x * second.y - second.x * first.y
        for first, second in zip(points, points[1:] + points[:1], strict=True)
    )
    return 0.5 * twice_area


@dataclass(frozen=True, slots=True)
class TractorGeometry:
    """Tractor dimensions referenced to the rear-axle midpoint."""

    wheelbase: float
    front_overhang: float
    rear_overhang: float
    body_width: float
    front_track: float
    rear_track: float
    body_height: float
    body_bottom_height: float
    hitch_longitudinal: float
    hitch_lateral: float
    hitch_height: float
    footprint_x: Bounds1D
    footprint_y: Bounds1D

    def __post_init__(self) -> None:
        for name in (
            "wheelbase",
            "front_overhang",
            "rear_overhang",
            "body_width",
            "front_track",
            "rear_track",
            "body_height",
            "hitch_height",
        ):
            _positive(getattr(self, name), f"TractorGeometry.{name}")
        for name in ("body_bottom_height", "hitch_longitudinal", "hitch_lateral"):
            if not isfinite(getattr(self, name)):
                raise GeometryValidationError(f"TractorGeometry.{name} must be finite")
        if self.body_bottom_height < 0.0:
            raise GeometryValidationError(
                "tractor body_bottom_height cannot be negative"
            )
        expected_x = (-self.rear_overhang, self.wheelbase + self.front_overhang)
        expected_y = (-self.body_width / 2.0, self.body_width / 2.0)
        if not (
            isclose(self.footprint_x.minimum, expected_x[0])
            and isclose(self.footprint_x.maximum, expected_x[1])
        ):
            raise GeometryValidationError(
                "tractor collision footprint x does not match overhangs and wheelbase"
            )
        if not (
            isclose(self.footprint_y.minimum, expected_y[0])
            and isclose(self.footprint_y.maximum, expected_y[1])
        ):
            raise GeometryValidationError(
                "tractor collision footprint y does not match body width"
            )

    @property
    def body_length(self) -> float:
        return self.front_overhang + self.wheelbase + self.rear_overhang

    @property
    def body_center_local(self) -> Point2D:
        return Point2D(
            (self.footprint_x.minimum + self.footprint_x.maximum) / 2.0,
            0.0,
        )

    def front_axle(self, tractor_pose: Pose2D) -> Point2D:
        return local_to_world(Point2D(self.wheelbase, 0.0), tractor_pose)

    def hitch(self, tractor_pose: Pose2D) -> Point2D:
        return local_to_world(
            Point2D(self.hitch_longitudinal, self.hitch_lateral), tractor_pose
        )

    def footprint(self, tractor_pose: Pose2D) -> tuple[Point2D, ...]:
        return rectangle_corners(self.footprint_x, self.footprint_y, tractor_pose)


@dataclass(frozen=True, slots=True)
class TrailerGeometry:
    """Trailer dimensions referenced to the trailer-axle midpoint."""

    variant: str
    hitch_to_axle: float
    front_overhang: float
    rear_overhang: float
    body_width: float
    axle_track: float
    body_height: float
    body_bottom_height: float
    footprint_x: Bounds1D
    footprint_y: Bounds1D

    def __post_init__(self) -> None:
        if not isinstance(self.variant, str) or not self.variant.strip():
            raise GeometryValidationError("trailer variant must be a non-empty string")
        for name in (
            "hitch_to_axle",
            "front_overhang",
            "rear_overhang",
            "body_width",
            "axle_track",
            "body_height",
        ):
            _positive(getattr(self, name), f"TrailerGeometry.{name}")
        if not isfinite(self.body_bottom_height) or self.body_bottom_height < 0.0:
            raise GeometryValidationError(
                "trailer body_bottom_height cannot be negative"
            )
        expected_x = (
            -self.rear_overhang,
            self.hitch_to_axle + self.front_overhang,
        )
        expected_y = (-self.body_width / 2.0, self.body_width / 2.0)
        if not (
            isclose(self.footprint_x.minimum, expected_x[0])
            and isclose(self.footprint_x.maximum, expected_x[1])
        ):
            raise GeometryValidationError(
                "trailer collision footprint x does not match overhangs and "
                "axle distance"
            )
        if not (
            isclose(self.footprint_y.minimum, expected_y[0])
            and isclose(self.footprint_y.maximum, expected_y[1])
        ):
            raise GeometryValidationError(
                "trailer collision footprint y does not match body width"
            )

    @property
    def body_length(self) -> float:
        return self.front_overhang + self.hitch_to_axle + self.rear_overhang

    @property
    def body_center_local(self) -> Point2D:
        return Point2D(
            (self.footprint_x.minimum + self.footprint_x.maximum) / 2.0,
            0.0,
        )

    def hitch(self, trailer_pose: Pose2D) -> Point2D:
        return local_to_world(Point2D(self.hitch_to_axle, 0.0), trailer_pose)

    def rear_face(self, trailer_pose: Pose2D) -> Point2D:
        return local_to_world(Point2D(-self.rear_overhang, 0.0), trailer_pose)

    def front_face(self, trailer_pose: Pose2D) -> Point2D:
        return local_to_world(
            Point2D(self.hitch_to_axle + self.front_overhang, 0.0), trailer_pose
        )

    def footprint(self, trailer_pose: Pose2D) -> tuple[Point2D, ...]:
        return rectangle_corners(self.footprint_x, self.footprint_y, trailer_pose)


@dataclass(frozen=True, slots=True)
class WheelGeometry:
    """Shared wheel dimensions used by semantic sites and MuJoCo."""

    radius: float
    width: float

    def __post_init__(self) -> None:
        _positive(self.radius, "WheelGeometry.radius")
        _positive(self.width, "WheelGeometry.width")


@dataclass(frozen=True, slots=True)
class VehicleGeometry:
    """Combined tractor/trailer geometry with rigid assembly helpers."""

    tractor: TractorGeometry
    trailer: TrailerGeometry
    wheels: WheelGeometry

    def __post_init__(self) -> None:
        if not isinstance(self.tractor, TractorGeometry):
            raise GeometryValidationError("vehicle tractor must be TractorGeometry")
        if not isinstance(self.trailer, TrailerGeometry):
            raise GeometryValidationError("vehicle trailer must be TrailerGeometry")
        if not isinstance(self.wheels, WheelGeometry):
            raise GeometryValidationError("vehicle wheels must be WheelGeometry")
        # Collision is evaluated on the chassis footprints only, in both
        # simulators, so every wheel has to lie inside its body's footprint.
        slack = 1e-9
        for name, track, body_width in (
            ("tractor front", self.tractor.front_track, self.tractor.body_width),
            ("tractor rear", self.tractor.rear_track, self.tractor.body_width),
            ("trailer", self.trailer.axle_track, self.trailer.body_width),
        ):
            if track + self.wheels.width > body_width + slack:
                raise GeometryValidationError(
                    f"{name} wheels extend beyond the body width; collision"
                    " footprints would not cover them"
                )
        for name, overhang in (
            ("tractor front", self.tractor.front_overhang),
            ("tractor rear", self.tractor.rear_overhang),
            ("trailer rear", self.trailer.rear_overhang),
        ):
            if self.wheels.radius > overhang + slack:
                raise GeometryValidationError(
                    f"wheels extend beyond the {name} overhang; collision"
                    " footprints would not cover them"
                )

    def trailer_pose_from_hitch(self, hitch: Point2D, trailer_yaw: float) -> Pose2D:
        offset = rotate(Point2D(self.trailer.hitch_to_axle, 0.0), trailer_yaw)
        return Pose2D(hitch.x - offset.x, hitch.y - offset.y, trailer_yaw)

    def assembled_state(
        self,
        tractor_pose: Pose2D,
        trailer_yaw: float,
        *,
        steering_angle: float = 0.0,
        steering_rate: float = 0.0,
        longitudinal_speed: float = 0.0,
        tractor_yaw_rate: float = 0.0,
        trailer_yaw_rate: float = 0.0,
        simulation_time: float = 0.0,
    ) -> VehicleState:
        """Build a geometrically consistent state without advancing dynamics."""

        hitch = self.tractor.hitch(tractor_pose)
        trailer_pose = self.trailer_pose_from_hitch(hitch, trailer_yaw)
        return VehicleState(
            tractor_pose=tractor_pose,
            trailer_pose=trailer_pose,
            hitch_position=hitch,
            steering_angle=steering_angle,
            steering_rate=steering_rate,
            longitudinal_speed=longitudinal_speed,
            tractor_yaw_rate=tractor_yaw_rate,
            trailer_yaw_rate=trailer_yaw_rate,
            simulation_time=simulation_time,
        )

    def neutral_state(self) -> VehicleState:
        """Return the zero-pose rigid assembly defined by the active geometry."""

        return self.assembled_state(Pose2D(0.0, 0.0, 0.0), 0.0)

    def constraint_residuals(self, state: VehicleState) -> dict[str, float]:
        """Return geometric disagreement in metres for adapter diagnostics."""

        tractor_hitch = self.tractor.hitch(state.tractor_pose)
        trailer_hitch = self.trailer.hitch(state.trailer_pose)
        return {
            "reported_to_tractor_hitch": hypot(
                state.hitch_position.x - tractor_hitch.x,
                state.hitch_position.y - tractor_hitch.y,
            ),
            "reported_to_trailer_hitch": hypot(
                state.hitch_position.x - trailer_hitch.x,
                state.hitch_position.y - trailer_hitch.y,
            ),
            "tractor_to_trailer_hitch": hypot(
                tractor_hitch.x - trailer_hitch.x,
                tractor_hitch.y - trailer_hitch.y,
            ),
        }


@dataclass(frozen=True, slots=True)
class DockGeometry:
    """Loading-bay geometry and frozen docking-success tolerances."""

    specification_id: str
    interior_width: float
    interior_depth: float
    wall_thickness: float
    wall_height: float
    collision_margin: float
    target_position: Point2D
    target_trailer_yaw: float
    longitudinal_tolerance: float
    lateral_tolerance: float
    trailer_yaw_tolerance: float
    require_no_collision: bool
    require_articulation_below_hard_limit: bool
    require_complete_trailer_inside_bay: bool

    def __post_init__(self) -> None:
        if not self.specification_id.strip():
            raise GeometryValidationError("dock specification_id cannot be empty")
        for name in (
            "interior_width",
            "interior_depth",
            "wall_thickness",
            "wall_height",
            "longitudinal_tolerance",
            "lateral_tolerance",
            "trailer_yaw_tolerance",
        ):
            _positive(getattr(self, name), f"DockGeometry.{name}")
        if not isfinite(self.collision_margin) or self.collision_margin < 0.0:
            raise GeometryValidationError(
                "dock collision_margin must be finite and non-negative"
            )
        if not isinstance(self.target_position, Point2D):
            raise GeometryValidationError("dock target_position must be Point2D")
        if self.collision_margin * 2.0 >= self.interior_width:
            raise GeometryValidationError("dock collision margins consume bay width")
        object.__setattr__(
            self, "target_trailer_yaw", wrap_angle(self.target_trailer_yaw)
        )

    @property
    def x_bounds(self) -> Bounds1D:
        return Bounds1D(0.0, self.interior_depth)

    @property
    def y_bounds(self) -> Bounds1D:
        return Bounds1D(-self.interior_width / 2.0, self.interior_width / 2.0)
