"""Scenario overrides for truck dimensions, limits, and domain randomization."""

from __future__ import annotations

from pathlib import Path

import pytest

from trailer_docking.core.config import ConfigurationError
from trailer_docking.evaluation.scenarios import (
    CUSTOM_TRAILER_VARIANT,
    DomainRandomization,
    Scenario,
    ScenarioOverrides,
)


def test_default_scenario_is_the_frozen_baseline() -> None:
    built = Scenario().build()
    assert built.vehicle.specification_id == "compact-semitrailer-v1"
    assert built.vehicle.trailer.variant == "nominal"
    assert built.vehicle.tractor.wheelbase == pytest.approx(1.90)
    assert built.task.specification_id == "gymnasium-docking-v1"
    assert built.vehicle.command.nominal_target_speed == pytest.approx(-0.2)


def test_dimension_overrides_rebuild_a_consistent_vehicle() -> None:
    scenario = Scenario(
        overrides=ScenarioOverrides(
            tractor_wheelbase=2.4,
            tractor_body_width=1.2,
            trailer_hitch_to_axle=3.2,
            trailer_rear_overhang=0.8,
            trailer_body_width=1.1,
        )
    )
    vehicle = scenario.build().vehicle
    tractor, trailer = vehicle.tractor, vehicle.trailer
    assert tractor.wheelbase == pytest.approx(2.4)
    assert tractor.footprint_x.maximum == pytest.approx(2.4 + tractor.front_overhang)
    assert tractor.footprint_y.maximum == pytest.approx(0.6)
    # The wheels stay the same 0.1 m inside each body side.
    assert tractor.front_track == pytest.approx(1.0)
    assert tractor.rear_track == pytest.approx(1.0)
    assert trailer.variant == CUSTOM_TRAILER_VARIANT
    assert trailer.hitch_to_axle == pytest.approx(3.2)
    assert trailer.footprint_x.minimum == pytest.approx(-0.8)
    assert trailer.axle_track == pytest.approx(0.9)
    # The neutral semantic sites follow the new geometry, so the MuJoCo
    # generator and the conformance validation see one consistent vehicle.
    assert vehicle.neutral_site("tractor_front_axle").x == pytest.approx(2.4)
    assert vehicle.neutral_site("trailer_axle").x == pytest.approx(-3.2)
    assert vehicle.neutral_site("docking_point").x == pytest.approx(-4.0)
    assert vehicle.trailer_variant(CUSTOM_TRAILER_VARIANT) == trailer
    # The frozen named variants are untouched.
    assert vehicle.trailer_variant("nominal").hitch_to_axle == pytest.approx(4.0)


def test_only_trailer_override_leaves_the_tractor_alone() -> None:
    base = Scenario().build().vehicle
    vehicle = (
        Scenario(overrides=ScenarioOverrides(trailer_hitch_to_axle=4.2)).build().vehicle
    )
    assert vehicle.tractor == base.tractor
    assert vehicle.trailer.body_width == base.trailer.body_width


def test_limit_overrides_set_limits_and_the_fixed_reverse_target() -> None:
    vehicle = (
        Scenario(
            overrides=ScenarioOverrides(
                steering_angle_max=0.4, steering_rate_max=0.5, reverse_speed=0.15
            )
        )
        .build()
        .vehicle
    )
    assert vehicle.limits.steering_angle_abs_max == pytest.approx(0.4)
    assert vehicle.limits.steering_rate_abs_max == pytest.approx(0.5)
    assert vehicle.limits.reverse_speed_abs_max == pytest.approx(0.15)
    assert vehicle.command.nominal_target_speed == pytest.approx(-0.15)


@pytest.mark.parametrize(
    "overrides",
    [
        {"tractor_wheelbase": -1.0},
        {"trailer_hitch_to_axle": 0.0},
        {"trailer_body_width": float("nan")},
        {"reverse_speed": float("inf")},
        {"tractor_wheelbase": True},
        # Not reachable by an Ackermann pair with the 1.9 m wheelbase.
        {"steering_angle_max": 1.6},
    ],
)
def test_invalid_overrides_are_rejected(overrides: dict) -> None:
    with pytest.raises(ValueError):
        Scenario(overrides=ScenarioOverrides(**overrides)).build()


def test_randomization_validation() -> None:
    assert DomainRandomization().trailer_hitch_to_axle is None
    assert DomainRandomization().observation_noise_std == 0.0
    chosen = DomainRandomization(trailer_hitch_to_axle=[3, 5])  # type: ignore[arg-type]
    assert chosen.trailer_hitch_to_axle == (3.0, 5.0)
    for bounds in ((5.0, 3.0), (0.0, 3.0), (1.0,), (1.0, float("nan")), "ab"):
        with pytest.raises(ConfigurationError):
            DomainRandomization(trailer_hitch_to_axle=bounds)  # type: ignore[arg-type]
    for noise in (-0.1, float("nan"), True, "x"):
        with pytest.raises(ConfigurationError):
            DomainRandomization(observation_noise_std=noise)  # type: ignore[arg-type]


def test_scenario_round_trip_and_fingerprint(tmp_path: Path) -> None:
    scenario = Scenario(
        name="custom",
        overrides=ScenarioOverrides(trailer_hitch_to_axle=3.5, reverse_speed=0.15),
        randomization=DomainRandomization(observation_noise_std=0.02),
    )
    path = scenario.save(tmp_path / "scenario.yaml")
    loaded = Scenario.load(path)
    assert loaded == scenario
    assert loaded.fingerprint == scenario.fingerprint
    assert loaded.to_dict()["task_path"].endswith("gymnasium_docking_v1.yaml")
    # Every part of the problem changes the fingerprint.
    base = Scenario(name="custom")
    assert base.fingerprint != scenario.fingerprint
    noisy = Scenario(
        name="custom", randomization=DomainRandomization(observation_noise_std=0.02)
    )
    assert len({base.fingerprint, noisy.fingerprint, scenario.fingerprint}) == 3


def test_combined_overrides_are_validated_together() -> None:
    # A short wheelbase is only valid with a smaller steering limit; applying
    # one before the other would reject a valid pair.
    vehicle = (
        Scenario(
            overrides=ScenarioOverrides(tractor_wheelbase=0.9, steering_angle_max=0.3)
        )
        .build()
        .vehicle
    )
    assert vehicle.tractor.wheelbase == pytest.approx(0.9)
    # A narrow body keeps its wheels inside it.
    narrow = Scenario(overrides=ScenarioOverrides(tractor_body_width=0.6)).build()
    assert narrow.vehicle.tractor.front_track == pytest.approx(0.4)


def test_long_tractor_does_not_start_out_of_bounds() -> None:
    import random

    from trailer_docking.environments.task import evaluate_task, workspace_bounds
    from trailer_docking.evaluation.scenarios import sample_start

    scenario = Scenario(overrides=ScenarioOverrides(tractor_wheelbase=3.0))
    built = scenario.build()
    pose, yaw = sample_start(scenario, built, random.Random(0))
    state = built.vehicle.geometry.assembled_state(pose, yaw)
    bounds = workspace_bounds(pose, built.dock.geometry, built.task, built.vehicle)
    status = evaluate_task(state, built.vehicle, built.dock.geometry, *bounds)
    assert not status.out_of_bounds and not status.terminated
    # The nominal tractor keeps exactly the bound the task always used.
    nominal = Scenario().build()
    assert workspace_bounds(
        pose, nominal.dock.geometry, nominal.task, nominal.vehicle
    ) == workspace_bounds(pose, nominal.dock.geometry, nominal.task)


@pytest.mark.parametrize(
    "scenario",
    [
        # Q-006: the long variant cannot be completely inside the 6 m bay.
        Scenario(overrides=ScenarioOverrides(trailer_variant="long")),
        Scenario(overrides=ScenarioOverrides(trailer_hitch_to_axle=4.6)),
        Scenario(overrides=ScenarioOverrides(trailer_body_width=1.6)),
        Scenario(overrides=ScenarioOverrides(dock_interior_width=0.8)),
        Scenario(randomization=DomainRandomization(trailer_hitch_to_axle=(3.0, 5.0))),
        # A fixed length and a randomized range contradict each other.
        Scenario(
            overrides=ScenarioOverrides(trailer_hitch_to_axle=3.5),
            randomization=DomainRandomization(trailer_hitch_to_axle=(3.0, 4.0)),
        ),
    ],
)
def test_unsolvable_or_contradictory_scenarios_are_rejected(scenario: Scenario) -> None:
    with pytest.raises(ConfigurationError):
        scenario.build()


def test_a_deeper_bay_admits_the_long_trailer() -> None:
    built = Scenario(
        overrides=ScenarioOverrides(trailer_variant="long", dock_interior_depth=7.0)
    ).build()
    assert built.vehicle.trailer.hitch_to_axle == pytest.approx(5.0)


@pytest.mark.parametrize(
    "start",
    [
        {"lateral": (6.0, 6.0)},  # outside the 5 m half-width workspace
        {"trailer_yaw": (3.0, 3.0)},  # pointing the wrong way, into the wall
        {"standoff": (0.0, 0.0), "lateral": (0.6, 0.6)},  # inside a side wall
    ],
)
def test_start_ranges_that_begin_failed_are_rejected(start: dict) -> None:
    from trailer_docking.evaluation.scenarios import StartDistribution

    with pytest.raises(ConfigurationError, match="begin already failed"):
        Scenario(start=StartDistribution(**start)).build()


def test_malformed_scenario_values_raise_configuration_errors() -> None:
    from trailer_docking.evaluation.scenarios import StartDistribution

    with pytest.raises(ConfigurationError):
        StartDistribution(standoff="12")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError):
        DomainRandomization(trailer_hitch_to_axle="34")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError):
        Scenario.from_dict({"start": {"standoff": ["a", "b"]}})
    with pytest.raises(ConfigurationError):
        Scenario(name=5)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError):
        Scenario.from_dict({"name": None})


def test_reference_goes_straight_to_the_goal_from_inside_the_entry_point() -> None:
    from trailer_docking.core.state import Point2D
    from trailer_docking.evaluation.scenarios import assembled_start
    from trailer_docking.paths.generators import docking_reference

    built = Scenario().build()
    geometry, bay = built.vehicle.geometry, built.dock.geometry

    def reference(x: float, y: float) -> tuple[Point2D, ...]:
        pose, yaw = assembled_start(geometry, Point2D(x, y), 0.0, 0.0)
        state = geometry.assembled_state(pose, yaw)
        return docking_reference(geometry, bay, state).points

    # Far, offset start: via the entry point outside the opening.
    assert len(reference(10.0, 0.3)) == 3
    # Offset start nearer than the entry point: a reversing vehicle cannot go
    # back out to it, so no doubling back.
    near = reference(6.3, 0.15)
    assert len(near) == 2 and near[-1].x == pytest.approx(6.3)
    assert len(reference(10.0, 0.0)) == 2


def test_older_scenario_files_without_new_sections_still_load() -> None:
    scenario = Scenario.from_dict({"name": "old", "overrides": {}, "start": {}})
    assert scenario.randomization == DomainRandomization()
    with pytest.raises(ConfigurationError):
        Scenario.from_dict({"randomization": {"unknown": 1}})
    with pytest.raises(ConfigurationError):
        Scenario.from_dict({"task_path": "does/not/exist.yaml"}).build()
