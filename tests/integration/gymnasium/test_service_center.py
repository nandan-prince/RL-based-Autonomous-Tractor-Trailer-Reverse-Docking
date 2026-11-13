"""Full-size service-center scenario acceptance tests."""

from math import pi
from pathlib import Path

import numpy as np
import pytest

from trailer_docking.core.config import (
    load_dock_configuration,
    load_docking_task_configuration,
    load_vehicle_configuration,
)
from trailer_docking.core.state import Pose2D
from trailer_docking.environments import TrailerDockingEnv
from trailer_docking.safety.collision import (
    dock_wall_polygons,
    vehicle_dock_collision,
)
from trailer_docking.safety.jackknife import articulation_status
from trailer_docking.visualization.topdown import SERVICE_DOOR_CENTERS, TopDownView

ROOT = Path(__file__).resolve().parents[3]
VEHICLE = ROOT / "configs/vehicles/full_size_semitrailer_v1.yaml"
DOCK = ROOT / "configs/environments/service_center_dock_v1.yaml"
TASK = ROOT / "configs/environments/service_center_gymnasium_v1.yaml"


def _environment(**kwargs) -> TrailerDockingEnv:
    return TrailerDockingEnv(
        vehicle_path=VEHICLE,
        dock_path=DOCK,
        task_path=TASK,
        action_mode="teleop",
        **kwargs,
    )


def test_full_size_dimensions_and_200_by_100_workspace() -> None:
    vehicle = load_vehicle_configuration(VEHICLE)
    dock = load_dock_configuration(DOCK)
    task = load_docking_task_configuration(TASK)
    assert vehicle.tractor.body_length == pytest.approx(5.8)
    assert vehicle.trailer.body_length == pytest.approx(14.0)
    assert vehicle.trailer.body_width == pytest.approx(2.55)
    assert dock.geometry.interior_width == pytest.approx(3.7)
    assert dock.geometry.interior_depth == pytest.approx(18.0)
    assert task.workspace_lateral_half_width == pytest.approx(100.0)
    assert task.workspace_outward_limit == pytest.approx(100.0)
    assert task.workspace_rear_padding == pytest.approx(0.0)
    assert task.start_standoff == pytest.approx(15.0)
    assert len(SERVICE_DOOR_CENTERS) == 20
    assert 100.0 in SERVICE_DOOR_CENTERS


def test_full_size_terminal_jackknife_threshold_is_60_degrees() -> None:
    vehicle = load_vehicle_configuration(VEHICLE)
    hard_limit = vehicle.limits.articulation_hard_abs_max
    assert hard_limit == pytest.approx(pi / 3, abs=1e-6)

    below = vehicle.geometry.assembled_state(
        Pose2D(35.0, 0.0, hard_limit - 1e-6),
        trailer_yaw=0.0,
    )
    boundary = vehicle.geometry.assembled_state(
        Pose2D(35.0, 0.0, hard_limit),
        trailer_yaw=0.0,
    )
    assert not articulation_status(below, vehicle.limits).jackknifed
    assert articulation_status(boundary, vehicle.limits).jackknifed


def test_service_center_facade_is_collision_geometry() -> None:
    vehicle = load_vehicle_configuration(VEHICLE)
    dock = load_dock_configuration(DOCK)
    state = vehicle.geometry.assembled_state(Pose2D(1.0, 8.0, pi), pi)
    contacts = vehicle_dock_collision(
        state, vehicle.geometry, dock.geometry
    ).contacts
    assert "tractor:building_left_facade" in contacts


def test_service_center_loading_zone_has_no_side_collision_walls() -> None:
    dock = load_dock_configuration(DOCK)
    walls = dock_wall_polygons(dock.geometry)
    assert "left_wall" not in walls
    assert "right_wall" not in walls
    assert "rear_wall" in walls


def test_service_center_display_transform_preserves_articulated_hitch() -> None:
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg", force=True)
    vehicle = load_vehicle_configuration(VEHICLE)
    dock = load_dock_configuration(DOCK)
    state = vehicle.geometry.assembled_state(
        Pose2D(35.0, 8.0, -0.35),
        trailer_yaw=0.12,
    )
    view = TopDownView(vehicle, dock)
    displayed_hitch = view.display_xy(state.hitch_position)
    displayed_tractor_hitch = vehicle.tractor.hitch(
        view.display_pose(state.tractor_pose)
    )
    displayed_trailer_hitch = vehicle.trailer.hitch(
        view.display_pose(state.trailer_pose)
    )

    assert (displayed_tractor_hitch.x, displayed_tractor_hitch.y) == pytest.approx(
        displayed_hitch, abs=1e-12
    )
    assert (displayed_trailer_hitch.x, displayed_trailer_hitch.y) == pytest.approx(
        displayed_hitch, abs=1e-12
    )
    plt.close(view.figure)


def test_vehicle_can_maneuver_and_default_reverse_docks() -> None:
    env = _environment()
    env.reset(seed=0)
    start = env.state.tractor_pose
    for _ in range(20):
        _, _, terminated, truncated, _ = env.step(
            np.array([0.5, 1.0], dtype=np.float32)
        )
        assert not terminated and not truncated
    assert env.state.tractor_pose.x > start.x + 1.0
    assert env.state.tractor_pose.y > start.y

    env.reset(seed=0)
    reverse = np.array([0.0, -1.0], dtype=np.float32)
    info = {}
    for _ in range(env.settings.max_episode_steps):
        _, _, terminated, truncated, info = env.step(reverse)
        if terminated or truncated:
            break
    assert terminated and not truncated
    assert info["termination_reason"] == "success"
    assert not info["collision"]
    env.close()


def test_large_yard_rgb_render() -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    env = _environment(render_mode="rgb_array")
    env.reset()
    image = env.render()
    assert image is not None
    assert image.shape[0] > 400 and image.shape[1] > 800
    assert env.figure is not None
    axes = env.figure.axes[0]
    assert axes.get_xlim() == pytest.approx((0.0, 200.0))
    assert axes.get_ylim() == pytest.approx((0.0, 100.0))
    env.close()
