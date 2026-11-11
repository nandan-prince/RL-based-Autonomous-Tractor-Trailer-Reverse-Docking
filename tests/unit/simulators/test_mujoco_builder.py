"""MuJoCo-independent checks of the Ackermann mapping and MJCF generator."""

import xml.etree.ElementTree as ElementTree
from math import atan, tan

import pytest

from trailer_docking.core.config import DockConfiguration, VehicleConfiguration
from trailer_docking.simulators.mujoco.ackermann import (
    ackermann_wheel_angles,
    virtual_steering_angle,
    virtual_steering_rate,
)
from trailer_docking.simulators.mujoco.builder import (
    DOCK_WALL_GEOMS,
    build_scene_xml,
    build_vehicle_xml,
    generated_files,
    mjcf_equivalent,
    steering_joint_ranges,
)
from trailer_docking.simulators.mujoco.builder import MujocoTuning as Tuning

WHEELBASE = 1.9
TRACK = 0.8


def test_zero_steering_keeps_both_wheels_straight() -> None:
    assert ackermann_wheel_angles(0.0, WHEELBASE, TRACK) == (0.0, 0.0)
    assert virtual_steering_angle(0.0, 0.0, WHEELBASE, TRACK) == 0.0


@pytest.mark.parametrize("virtual", [-0.5, -0.2, -0.01, 0.01, 0.3, 0.5])
def test_ackermann_matches_specification_and_round_trips(virtual: float) -> None:
    left, right = ackermann_wheel_angles(virtual, WHEELBASE, TRACK)
    turn_radius = WHEELBASE / tan(virtual)
    assert left == pytest.approx(atan(WHEELBASE / (turn_radius - TRACK / 2.0)))
    assert right == pytest.approx(atan(WHEELBASE / (turn_radius + TRACK / 2.0)))
    # The inner wheel steers more than the virtual wheel, the outer less.
    inner, outer = (left, right) if virtual > 0.0 else (right, left)
    assert abs(inner) > abs(virtual) > abs(outer)
    assert virtual_steering_angle(left, right, WHEELBASE, TRACK) == pytest.approx(
        virtual, abs=1e-12
    )


def test_virtual_steering_rate_matches_finite_difference() -> None:
    virtual, rate, step = 0.3, 0.25, 1e-6
    before = ackermann_wheel_angles(virtual - rate * step, WHEELBASE, TRACK)
    after = ackermann_wheel_angles(virtual + rate * step, WHEELBASE, TRACK)
    left, right = ackermann_wheel_angles(virtual, WHEELBASE, TRACK)
    left_rate = (after[0] - before[0]) / (2.0 * step)
    right_rate = (after[1] - before[1]) / (2.0 * step)
    assert virtual_steering_rate(
        left, right, left_rate, right_rate, WHEELBASE, TRACK
    ) == pytest.approx(rate, rel=1e-6)


def _names(root: ElementTree.Element, tag: str) -> set[str]:
    return {
        element.attrib["name"] for element in root.iter(tag) if "name" in element.attrib
    }


def test_vehicle_xml_exposes_every_frozen_name(
    vehicle_config: VehicleConfiguration,
) -> None:
    root = ElementTree.fromstring(build_vehicle_xml(vehicle_config))
    required = vehicle_config.mujoco.required_names
    joints = _names(root, "joint") | _names(root, "freejoint")
    actuators = _names(root, "position") | _names(root, "velocity")
    assert set(required.bodies) <= _names(root, "body")
    assert set(required.joints) <= joints
    assert set(required.sites) <= _names(root, "site")
    assert set(required.actuators) == actuators
    # The vehicle model is scene-independent.
    assert not _names(root, "light")
    assert "ground" not in _names(root, "geom")


def test_scene_xml_contains_dock_walls_only_with_a_dock(
    vehicle_config: VehicleConfiguration, dock_config: DockConfiguration
) -> None:
    yard = ElementTree.fromstring(build_scene_xml(vehicle_config))
    scene = ElementTree.fromstring(build_scene_xml(vehicle_config, dock_config))
    assert not set(DOCK_WALL_GEOMS) & _names(yard, "geom")
    assert set(DOCK_WALL_GEOMS) <= _names(scene, "geom")
    assert scene.attrib["model"] == dock_config.geometry.specification_id


def test_generation_is_deterministic_and_variant_aware(
    vehicle_config: VehicleConfiguration,
) -> None:
    assert build_vehicle_xml(vehicle_config) == build_vehicle_xml(vehicle_config)
    long_vehicle = vehicle_config.with_trailer_variant("long")
    assert build_vehicle_xml(long_vehicle) != build_vehicle_xml(vehicle_config)
    assert "trailer variant: long" in build_vehicle_xml(long_vehicle)


def test_steering_stops_sit_at_the_virtual_steering_limit(
    vehicle_config: VehicleConfiguration,
) -> None:
    inner, outer = ackermann_wheel_angles(
        vehicle_config.limits.steering_angle_abs_max,
        vehicle_config.tractor.wheelbase,
        vehicle_config.tractor.front_track,
    )
    ranges = steering_joint_ranges(vehicle_config, Tuning())
    assert ranges["left"] == pytest.approx((-outer, inner))
    assert ranges["right"] == pytest.approx((-inner, outer))
    widened = steering_joint_ranges(vehicle_config, Tuning(steering_range_margin=0.1))
    assert widened["left"] == pytest.approx((-outer - 0.1, inner + 0.1))


def test_committed_mjcf_matches_generator(
    vehicle_config: VehicleConfiguration, dock_config: DockConfiguration
) -> None:
    for path, expected in generated_files(vehicle_config, dock_config).items():
        assert path.is_file(), f"missing generated file {path}"
        assert mjcf_equivalent(path.read_text(encoding="utf-8"), expected), (
            f"{path.name} is out of date; run "
            "python -m trailer_docking.simulators.mujoco.builder --write"
        )
