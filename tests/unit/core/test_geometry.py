import math
import random

import pytest

from trailer_docking.core.config import VehicleConfiguration
from trailer_docking.core.geometry import (
    Bounds1D,
    GeometryValidationError,
    local_to_world,
    polygon_area,
    rectangle_corners,
    world_to_local,
)
from trailer_docking.core.state import Point2D, Pose2D


def test_nominal_golden_geometry(vehicle_config: VehicleConfiguration) -> None:
    geometry = vehicle_config.geometry
    state = geometry.neutral_state()

    assert geometry.tractor.body_length == pytest.approx(2.90)
    assert geometry.trailer.body_length == pytest.approx(5.50)
    assert state.tractor_pose == Pose2D(0.0, 0.0, 0.0)
    assert state.hitch_position == Point2D(0.0, 0.0)
    assert state.trailer_pose == Pose2D(-4.0, 0.0, 0.0)
    assert geometry.trailer.rear_face(state.trailer_pose) == Point2D(-5.0, 0.0)
    assert geometry.trailer.front_face(state.trailer_pose) == Point2D(0.5, 0.0)


def test_ninety_degree_neutral_assembly(vehicle_config: VehicleConfiguration) -> None:
    state = vehicle_config.geometry.assembled_state(
        Pose2D(0.0, 0.0, math.pi / 2.0), math.pi / 2.0
    )
    assert state.hitch_position.x == pytest.approx(0.0, abs=1e-12)
    assert state.hitch_position.y == pytest.approx(0.0, abs=1e-12)
    assert state.trailer_pose.x == pytest.approx(0.0, abs=1e-12)
    assert state.trailer_pose.y == pytest.approx(-4.0)
    rear = vehicle_config.trailer.rear_face(state.trailer_pose)
    assert rear.x == pytest.approx(0.0, abs=1e-12)
    assert rear.y == pytest.approx(-5.0)


def test_local_world_round_trip_for_random_frames() -> None:
    generator = random.Random(642)
    for _ in range(500):
        frame = Pose2D(
            generator.uniform(-100.0, 100.0),
            generator.uniform(-100.0, 100.0),
            generator.uniform(-20.0 * math.pi, 20.0 * math.pi),
        )
        local = Point2D(generator.uniform(-10.0, 10.0), generator.uniform(-10.0, 10.0))
        recovered = world_to_local(local_to_world(local, frame), frame)
        assert recovered.x == pytest.approx(local.x, abs=1e-11)
        assert recovered.y == pytest.approx(local.y, abs=1e-11)


def test_footprint_area_and_order_are_pose_invariant(
    vehicle_config: VehicleConfiguration,
) -> None:
    generator = random.Random(20)
    for _ in range(100):
        pose = Pose2D(
            generator.uniform(-10.0, 10.0),
            generator.uniform(-10.0, 10.0),
            generator.uniform(-math.pi, math.pi),
        )
        tractor_area = polygon_area(vehicle_config.tractor.footprint(pose))
        trailer_area = polygon_area(vehicle_config.trailer.footprint(pose))
        assert tractor_area == pytest.approx(2.9, abs=1e-12)
        assert trailer_area == pytest.approx(5.5, abs=1e-12)
        assert tractor_area > 0.0
        assert trailer_area > 0.0


@pytest.mark.parametrize(
    ("variant", "axle_distance", "body_length", "trailer_x"),
    [
        ("short", 2.0, 3.5, -2.0),
        ("nominal", 4.0, 5.5, -4.0),
        ("long", 5.0, 6.5, -5.0),
    ],
)
def test_all_trailer_variants_reconstruct_from_configuration(
    vehicle_config: VehicleConfiguration,
    variant: str,
    axle_distance: float,
    body_length: float,
    trailer_x: float,
) -> None:
    selected = vehicle_config.with_trailer_variant(variant)
    assert selected.trailer.hitch_to_axle == pytest.approx(axle_distance)
    assert selected.trailer.body_length == pytest.approx(body_length)
    assert selected.geometry.neutral_state().trailer_pose.x == pytest.approx(trailer_x)
    assert selected.effective_fingerprint != ""


def test_rigid_constraint_residuals_detect_inconsistent_state(
    vehicle_config: VehicleConfiguration,
) -> None:
    geometry = vehicle_config.geometry
    state = geometry.neutral_state()
    assert max(geometry.constraint_residuals(state).values()) == pytest.approx(0.0)

    shifted = type(state)(
        tractor_pose=state.tractor_pose,
        trailer_pose=Pose2D(state.trailer_pose.x + 0.1, 0.0, 0.0),
        hitch_position=state.hitch_position,
        steering_angle=0.0,
        steering_rate=0.0,
        longitudinal_speed=0.0,
        tractor_yaw_rate=0.0,
        trailer_yaw_rate=0.0,
        simulation_time=0.0,
    )
    residual = geometry.constraint_residuals(shifted)["tractor_to_trailer_hitch"]
    assert residual == pytest.approx(0.1)


def test_invalid_bounds_and_polygon_are_rejected() -> None:
    with pytest.raises(GeometryValidationError):
        Bounds1D(1.0, 1.0)
    with pytest.raises(GeometryValidationError, match="at least three"):
        polygon_area([Point2D(0.0, 0.0), Point2D(1.0, 0.0)])
    with pytest.raises(GeometryValidationError, match="finite"):
        Bounds1D(float("nan"), 1.0)


def test_rectangle_corners_are_counterclockwise() -> None:
    corners = rectangle_corners(
        Bounds1D(-1.0, 2.0), Bounds1D(-0.5, 0.5), Pose2D(4.0, -3.0, 1.2)
    )
    assert polygon_area(corners) == pytest.approx(3.0)
