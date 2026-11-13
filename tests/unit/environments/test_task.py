"""Scenario construction, goal, and terminal-event tests."""

from math import pi

import pytest

from trailer_docking.core.geometry import Bounds1D
from trailer_docking.core.state import Point2D, Pose2D
from trailer_docking.environments import DockingTaskSettings
from trailer_docking.environments.task import (
    aligned_pose_for_docking_point,
    default_start_pose,
    evaluate_task,
    workspace_bounds,
)


def test_aligned_pose_places_requested_docking_point(vehicle_config) -> None:
    requested = Point2D(3.0, -1.0)
    pose = aligned_pose_for_docking_point(vehicle_config, requested, pi / 4)
    state = vehicle_config.geometry.assembled_state(pose, pi / 4)
    actual = vehicle_config.trailer.rear_face(state.trailer_pose)
    assert actual.x == pytest.approx(requested.x)
    assert actual.y == pytest.approx(requested.y)
    assert max(vehicle_config.geometry.constraint_residuals(state).values()) < 1e-12


def test_default_start_is_outside_opening_and_in_workspace(
    vehicle_config, dock_config
) -> None:
    settings = DockingTaskSettings()
    pose = default_start_pose(vehicle_config, dock_config.geometry, settings)
    state = vehicle_config.geometry.assembled_state(pose, pose.yaw)
    docking = vehicle_config.trailer.rear_face(state.trailer_pose)
    x_bounds, y_bounds = workspace_bounds(pose, dock_config.geometry, settings)
    status = evaluate_task(
        state, vehicle_config, dock_config.geometry, x_bounds, y_bounds
    )
    assert docking.x == pytest.approx(10.0)
    assert not status.terminated


def test_success_requires_pose_clearance_and_complete_trailer(
    vehicle_config, dock_config
) -> None:
    dock = dock_config.geometry
    settings = DockingTaskSettings()
    start = default_start_pose(vehicle_config, dock, settings)
    bounds = workspace_bounds(start, dock, settings)
    pose = aligned_pose_for_docking_point(vehicle_config, dock.target_position)
    success = evaluate_task(
        vehicle_config.geometry.assembled_state(pose, 0),
        vehicle_config,
        dock,
        *bounds,
    )
    assert success.success
    assert success.reason == "success"

    wall_pose = aligned_pose_for_docking_point(vehicle_config, Point2D(0, 0))
    wall = evaluate_task(
        vehicle_config.geometry.assembled_state(wall_pose, 0),
        vehicle_config,
        dock,
        *bounds,
    )
    assert wall.collision.collided
    assert wall.reason == "collision"


def test_jackknife_and_out_of_bounds_are_independent(
    vehicle_config, dock_config
) -> None:
    dock = dock_config.geometry
    limits = vehicle_config.limits
    jackknife = vehicle_config.geometry.assembled_state(
        Pose2D(15, 0, limits.articulation_hard_abs_max), 0
    )
    status = evaluate_task(
        jackknife,
        vehicle_config,
        dock,
        Bounds1D(-100, 100),
        Bounds1D(-100, 100),
    )
    assert status.articulation.jackknifed
    assert status.reason == "jackknife"

    outside = vehicle_config.geometry.assembled_state(Pose2D(100, 0, 0), 0)
    status = evaluate_task(
        outside,
        vehicle_config,
        dock,
        Bounds1D(-2, 20),
        Bounds1D(-5, 5),
    )
    assert status.out_of_bounds
    assert status.reason == "out_of_bounds"
