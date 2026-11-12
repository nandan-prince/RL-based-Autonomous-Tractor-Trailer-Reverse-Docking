"""Planar collision, docking success, jackknife, and scenario overrides."""

import random
from dataclasses import replace
from pathlib import Path

import pytest

from trailer_docking.core.config import (
    ConfigurationError,
    DockConfiguration,
    VehicleConfiguration,
)
from trailer_docking.core.state import Point2D, Pose2D
from trailer_docking.evaluation.metrics import (
    docking_error,
    evaluate_docking,
    trailer_inside_bay,
)
from trailer_docking.evaluation.scenarios import (
    PHYSICS_ONLY_FIELDS,
    Scenario,
    ScenarioOverrides,
    StartDistribution,
    assembled_start,
    sample_start,
)
from trailer_docking.safety.collision import (
    convex_polygons_overlap,
    dock_wall_polygons,
    wall_collisions,
)
from trailer_docking.safety.jackknife import exceeds_soft_articulation, is_jackknifed


def _square(x: float, y: float, size: float = 1.0) -> tuple[Point2D, ...]:
    return (
        Point2D(x, y),
        Point2D(x + size, y),
        Point2D(x + size, y + size),
        Point2D(x, y + size),
    )


def test_convex_overlap_distinguishes_touching_from_penetrating() -> None:
    assert convex_polygons_overlap(_square(0, 0), _square(0.5, 0.5))
    assert not convex_polygons_overlap(_square(0, 0), _square(2, 0))
    assert not convex_polygons_overlap(_square(0, 0), _square(1, 0))  # touching edge
    assert convex_polygons_overlap(_square(0, 0), _square(0.999, 0))
    # A diamond whose edge x + y = 2 passes exactly through the corner (1, 1).
    diamond = (
        Point2D(1.5, 0.5),
        Point2D(2.5, 1.5),
        Point2D(1.5, 2.5),
        Point2D(0.5, 1.5),
    )
    assert not convex_polygons_overlap(_square(0, 0), diamond)
    closer = tuple(Point2D(p.x - 0.2, p.y - 0.2) for p in diamond)
    assert convex_polygons_overlap(_square(0, 0), closer)
    # Axis-aligned bounding boxes overlap here, but the shapes do not.
    farther = tuple(Point2D(p.x + 0.3, p.y + 0.3) for p in diamond)
    assert not convex_polygons_overlap(_square(0, 0), farther)


def test_wall_polygons_match_the_dock_specification(
    dock_config: DockConfiguration,
) -> None:
    bay = dock_config.geometry
    walls = dock_wall_polygons(bay)
    assert set(walls) == {"back_wall", "left_wall", "right_wall"}
    back_x = [point.x for point in walls["back_wall"]]
    assert (min(back_x), max(back_x)) == pytest.approx((-bay.wall_thickness, 0.0))
    left_y = [point.y for point in walls["left_wall"]]
    assert min(left_y) == pytest.approx(bay.interior_width / 2.0)
    assert max(left_y) == pytest.approx(bay.interior_width / 2.0 + bay.wall_thickness)
    right_y = [point.y for point in walls["right_wall"]]
    assert max(right_y) == pytest.approx(-bay.interior_width / 2.0)


def _docked_state(vehicle: VehicleConfiguration, x: float, y: float, yaw: float = 0.0):
    pose, trailer_yaw = assembled_start(vehicle.geometry, Point2D(x, y), yaw, 0.0)
    return vehicle.geometry.assembled_state(pose, trailer_yaw)


def test_docking_success_and_each_failure_condition(
    vehicle_config: VehicleConfiguration, dock_config: DockConfiguration
) -> None:
    geometry, bay, limits = (
        vehicle_config.geometry,
        dock_config.geometry,
        vehicle_config.limits,
    )
    target = bay.target_position

    at_target = _docked_state(vehicle_config, target.x, target.y)
    outcome = evaluate_docking(at_target, geometry, bay, limits)
    assert outcome.success and outcome.within_tolerance and outcome.trailer_inside_bay
    assert outcome.collisions == () and not outcome.jackknifed
    assert outcome.error.distance == pytest.approx(0.0, abs=1e-9)

    # Sign convention: still outside the target is positive longitudinal error,
    # left of the target is positive lateral error.
    outside = docking_error(_docked_state(vehicle_config, 3.0, 0.2), geometry, bay)
    assert outside.longitudinal == pytest.approx(3.0 - target.x)
    assert outside.lateral == pytest.approx(0.2)

    too_far = evaluate_docking(
        _docked_state(vehicle_config, target.x + 0.11, 0.0), geometry, bay, limits
    )
    assert not too_far.within_tolerance and not too_far.success

    # Inside tolerance but pressed into the side wall.
    wall = evaluate_docking(
        _docked_state(vehicle_config, target.x, 0.26), geometry, bay, limits
    )
    assert "trailer:left_wall" in wall.collisions and not wall.success
    assert not trailer_inside_bay(
        _docked_state(vehicle_config, target.x, 0.26), geometry, bay
    )

    through_back = wall_collisions(
        _docked_state(vehicle_config, -0.05, 0.0), geometry, bay
    )
    assert "trailer:back_wall" in through_back

    jackknifed = geometry.assembled_state(
        Pose2D(20.0, 5.0, limits.articulation_hard_abs_max), 0.0
    )
    assert is_jackknifed(jackknifed, limits)
    assert exceeds_soft_articulation(jackknifed, limits)
    assert not evaluate_docking(jackknifed, geometry, bay, limits).success
    soft = geometry.assembled_state(Pose2D(20.0, 5.0, 0.7), 0.0)
    assert exceeds_soft_articulation(soft, limits) and not is_jackknifed(soft, limits)

    # Relaxing a requirement in the dock specification relaxes success.
    relaxed = replace(
        bay, require_no_collision=False, require_complete_trailer_inside_bay=False
    )
    assert evaluate_docking(
        _docked_state(vehicle_config, target.x, 0.09), geometry, relaxed, limits
    ).success


def test_scenario_overrides_build_validate_and_round_trip(tmp_path: Path) -> None:
    base = Scenario().build()
    scenario = Scenario(
        name="wide-light",
        overrides=ScenarioOverrides(
            trailer_variant="short",
            dock_interior_width=2.0,
            dock_interior_depth=5.0,
            tractor_chassis_mass=500.0,
            trailer_body_mass=900.0,
            ground_friction=0.6,
        ),
    )
    built = scenario.build()
    assert built.vehicle.trailer.variant == "short"
    assert built.dock.geometry.interior_width == 2.0
    assert built.dock.geometry.interior_depth == 5.0
    assert built.vehicle.mujoco.tractor_chassis_mass == 500.0
    assert built.vehicle.mujoco.trailer_body_mass == 900.0
    assert built.vehicle.mujoco.contact_friction[0] == 0.6
    assert (
        built.vehicle.mujoco.contact_friction[1:]
        == base.vehicle.mujoco.contact_friction[1:]
    )
    # The frozen base specification is untouched.
    assert Scenario().build().dock.geometry.interior_width == 1.5
    assert {"tractor_chassis_mass", "trailer_body_mass", "ground_friction"} == set(
        PHYSICS_ONLY_FIELDS
    )

    path = scenario.save(tmp_path / "nested" / "scenario.yaml")
    loaded = Scenario.load(path)
    assert loaded == scenario
    assert loaded.fingerprint == scenario.fingerprint != Scenario().fingerprint

    # Invalid values are rejected when the override is created ...
    for bad in (
        {"dock_interior_width": -1.0},
        {"trailer_body_mass": 0.0},
        {"ground_friction": -0.5},
        {"ground_friction": float("nan")},
        {"trailer_body_mass": "heavy"},
        {"trailer_variant": ""},
    ):
        with pytest.raises(ConfigurationError):
            ScenarioOverrides(**bad)  # type: ignore[arg-type]
    # ... and cross-field problems when the scenario is built.
    for overrides in (
        ScenarioOverrides(dock_interior_width=0.01),
        ScenarioOverrides(trailer_variant="gigantic"),
    ):
        with pytest.raises(ValueError):
            Scenario(overrides=overrides).build()
    with pytest.raises(ConfigurationError):
        Scenario.from_dict({"name": "x", "surprise": 1})
    with pytest.raises(ConfigurationError):
        Scenario.from_dict({"overrides": {"not_a_field": 1}})
    with pytest.raises(ConfigurationError):
        Scenario.load(tmp_path / "missing.yaml")
    with pytest.raises(ConfigurationError):
        StartDistribution(standoff=(5.0, 3.0))
    with pytest.raises(ConfigurationError):
        Scenario(name="  ")


def test_sampled_starts_respect_the_ranges_and_are_reproducible() -> None:
    scenario = Scenario(
        start=StartDistribution((4.0, 6.0), (-0.3, 0.3), (-0.1, 0.1), (-0.05, 0.05))
    )
    built = scenario.build()
    geometry, bay = built.vehicle.geometry, built.dock.geometry
    first = [sample_start(scenario, built, random.Random(7)) for _ in range(2)]
    assert first[0] == first[1]
    rng = random.Random(1)
    for _ in range(50):
        pose, trailer_yaw = sample_start(scenario, built, rng)
        state = geometry.assembled_state(pose, trailer_yaw)
        docking = geometry.trailer.rear_face(state.trailer_pose)
        assert (
            bay.interior_depth + 4.0 - 1e-9
            <= docking.x
            <= bay.interior_depth + 6.0 + 1e-9
        )
        assert abs(docking.y) <= 0.3 + 1e-9
        assert abs(state.trailer_pose.yaw) <= 0.1 + 1e-9
        assert abs(state.articulation_angle) <= 0.05 + 1e-9
        assert max(geometry.constraint_residuals(state).values()) < 1e-9
