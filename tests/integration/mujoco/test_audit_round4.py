"""Regression tests for defects found by the fourth repository audit."""

from __future__ import annotations

import xml.etree.ElementTree as ElementTree
from dataclasses import replace
from pathlib import Path

import pytest

from trailer_docking.core.config import (
    ConfigurationError,
    DockConfiguration,
    VehicleConfiguration,
    load_vehicle_configuration,
)
from trailer_docking.core.geometry import GeometryValidationError, VehicleGeometry
from trailer_docking.core.state import VehicleCommand
from trailer_docking.safety.supervisor import (
    MeasurementTolerance,
    state_limit_violations,
)
from trailer_docking.simulators.mujoco.builder import (
    build_scene_xml,
    build_vehicle_xml,
    mjcf_equivalent,
)
from trailer_docking.validation import vehicle_spec

mujoco = pytest.importorskip("mujoco")

from trailer_docking.simulators.mujoco.adapter import (  # noqa: E402
    MUJOCO_MEASUREMENT_TOLERANCE,
    MujocoSimulator,
)


def test_limit_respecting_rollout_reports_no_violation_with_backend_tolerance(
    vehicle_config: VehicleConfiguration,
) -> None:
    simulator = MujocoSimulator(vehicle_config)
    simulator.reset(settle_time=0.5)
    commands = (
        [VehicleCommand(0.3, -0.2)] * 10
        + [VehicleCommand(0.0, -0.2)] * 30
        + [VehicleCommand(-0.3, -0.2)] * 20
        + [VehicleCommand(0.3, 0.2)] * 25
        + [VehicleCommand(0.0, 0.2)] * 40
        + [VehicleCommand(-0.3, -0.2)] * 40
    )
    exact: set[str] = set()
    for command in commands:
        state = simulator.step(command)
        exact.update(state_limit_violations(state, vehicle_config.limits))
        assert (
            state_limit_violations(
                state, vehicle_config.limits, MUJOCO_MEASUREMENT_TOLERANCE
            )
            == ()
        )
    # The exact-plant default does flag this physical plant, which is why the
    # backend publishes a tolerance.
    assert exact


def test_measurement_tolerance_never_relaxes_the_jackknife_threshold(
    vehicle_config: VehicleConfiguration,
) -> None:
    state = vehicle_config.geometry.assembled_state(
        vehicle_config.geometry.neutral_state().tractor_pose,
        -vehicle_config.limits.articulation_hard_abs_max,
        steering_angle=0.502,
    )
    loose = MeasurementTolerance(steering_angle=0.005, steering_rate=1.0, speed=1.0)
    assert state_limit_violations(state, vehicle_config.limits, loose) == (
        "jackknife_limit",
    )
    assert "steering_angle_limit" in state_limit_violations(
        state, vehicle_config.limits
    )
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            MeasurementTolerance(speed=bad)


@pytest.mark.parametrize("identifier", ['Dock "A"', "yard & bay", "a<b", "x--y-", "é"])
def test_free_text_identifiers_are_escaped_in_mjcf(
    vehicle_config: VehicleConfiguration,
    dock_config: DockConfiguration,
    identifier: str,
) -> None:
    dock = replace(
        dock_config,
        geometry=replace(dock_config.geometry, specification_id=identifier),
    )
    vehicle = replace(vehicle_config, specification_id=identifier)
    for text in (build_scene_xml(vehicle, dock), build_vehicle_xml(vehicle)):
        root = ElementTree.fromstring(text)
        assert root.attrib["model"] == identifier
        assert mjcf_equivalent(text, text)
        assert mujoco.MjModel.from_xml_string(text).nu == 4


@pytest.mark.parametrize("step", ["1e-320", "1e-9", "1e-300"])
@pytest.mark.parametrize("field", ["mujoco_physics_step", "kinematic_integration_step"])
def test_absurd_integration_steps_are_configuration_errors(
    tmp_path: Path, vehicle_config: VehicleConfiguration, field: str, step: str
) -> None:
    text = vehicle_config.source_path.read_text(encoding="utf-8")
    original = {
        "mujoco_physics_step": "  mujoco_physics_step: 0.002\n",
        "kinematic_integration_step": "  kinematic_integration_step: 0.02\n",
    }[field]
    assert original in text
    path = tmp_path / "tiny_step.yaml"
    path.write_text(text.replace(original, f"  {field}: {step}\n"), encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_vehicle_configuration(path)
    assert vehicle_spec.main(["--vehicle", str(path)]) == 2


def test_wheels_must_lie_inside_the_collision_footprints(
    vehicle_config: VehicleConfiguration,
) -> None:
    geometry = vehicle_config.geometry
    cases = (
        {"trailer": replace(geometry.trailer, axle_track=1.6)},
        {"tractor": replace(geometry.tractor, front_track=0.95)},
        {"tractor": replace(geometry.tractor, rear_track=0.95)},
        {"wheels": replace(geometry.wheels, width=0.3)},
        {"wheels": replace(geometry.wheels, radius=0.45)},
    )
    for change in cases:
        with pytest.raises(GeometryValidationError, match="footprints"):
            replace(geometry, **change)  # type: ignore[arg-type]
    # Exactly flush wheels are allowed.
    flush = replace(geometry, wheels=replace(geometry.wheels, width=0.2))
    assert isinstance(flush, VehicleGeometry)
