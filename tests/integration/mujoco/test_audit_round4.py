"""Regression tests for defects found by the fourth repository audit."""

from __future__ import annotations

import xml.etree.ElementTree as ElementTree
from dataclasses import replace

import pytest

from trailer_docking.core.config import (
    DockConfiguration,
    VehicleConfiguration,
)
from trailer_docking.core.state import VehicleCommand
from trailer_docking.safety.supervisor import (
    state_limit_violations,
)
from trailer_docking.simulators.mujoco.builder import (
    build_scene_xml,
    build_vehicle_xml,
    mjcf_equivalent,
)

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
