"""MuJoCo regression tests for defects found by the second repository audit."""

from __future__ import annotations

from dataclasses import replace

import pytest

from trailer_docking.core.config import DockConfiguration, VehicleConfiguration
from trailer_docking.core.state import Pose2D, VehicleCommand

mujoco = pytest.importorskip("mujoco")

from trailer_docking.simulators.mujoco.adapter import (  # noqa: E402
    MujocoSimulator,
)


def test_stepped_state_is_current_not_one_physics_step_old(
    vehicle_config: VehicleConfiguration,
) -> None:
    simulator = MujocoSimulator(vehicle_config)
    simulator.reset()
    for _ in range(31):
        state = simulator.step(VehicleCommand(0.1, -0.2))
    # Recomputing derived quantities from the integrated positions must not
    # change anything the adapter already reported.
    fresh = mujoco.MjData(simulator.model)
    fresh.qpos[:] = simulator.data.qpos
    fresh.qvel[:] = simulator.data.qvel
    mujoco.mj_forward(simulator.model, fresh)
    site = mujoco.mj_name2id(
        simulator.model, mujoco.mjtObj.mjOBJ_SITE, "tractor_rear_axle"
    )
    assert state.tractor_pose.x == pytest.approx(fresh.site_xpos[site][0], abs=1e-12)
    assert state.tractor_pose.y == pytest.approx(fresh.site_xpos[site][1], abs=1e-12)
    assert state.simulation_time == pytest.approx(3.1)


def test_reset_rejects_poses_the_model_cannot_represent(
    vehicle_config: VehicleConfiguration, dock_config: DockConfiguration
) -> None:
    simulator = MujocoSimulator(vehicle_config, dock_config)
    mechanical = vehicle_config.limits.mujoco_hitch_mechanical_abs_max
    for trailer_yaw in (1.6, 3.0, -1.6):
        with pytest.raises(ValueError, match="mechanical range"):
            simulator.reset(Pose2D(20.0, 0.0, 0.0), trailer_yaw)
    # Relative to the model's own start pose as well as an explicit one.
    with pytest.raises(ValueError, match="mechanical range"):
        simulator.reset(None, 1.6)
    with pytest.raises(ValueError, match="steering_angle"):
        simulator.reset(steering_angle=0.6)
    with pytest.raises(ValueError, match="steering_angle"):
        simulator.reset(steering_angle=float("nan"))
    with pytest.raises(ValueError, match="settle_time"):
        simulator.reset(settle_time=-1.0)

    state = simulator.reset(Pose2D(20.0, 0.0, 0.0), mechanical - 0.05)
    assert state.articulation_angle == pytest.approx(-(mechanical - 0.05))
    state = simulator.step(VehicleCommand(0.0, 0.0))
    assert abs(state.longitudinal_speed) < 0.01
    assert abs(state.tractor_yaw_rate) < 0.01


def test_non_terminating_physics_timestep_compiles(
    vehicle_config: VehicleConfiguration,
) -> None:
    timing = replace(vehicle_config.timing, mujoco_physics_step=0.1 / 30.0)
    simulator = MujocoSimulator(replace(vehicle_config, timing=timing))
    assert simulator.model.opt.timestep == 0.1 / 30.0
    state = simulator.step(VehicleCommand(0.0, 0.0))
    assert state.simulation_time == pytest.approx(0.1)
