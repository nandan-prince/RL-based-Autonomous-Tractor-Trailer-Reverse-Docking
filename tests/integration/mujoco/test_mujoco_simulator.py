"""Validation gates for the MuJoCo vehicle, dock scene, and shared adapter.

These tests skip cleanly when the optional ``mujoco`` extra is not installed.
"""

from __future__ import annotations

from math import cos, pi, sin, tan

import pytest

from trailer_docking.core.config import DockConfiguration, VehicleConfiguration
from trailer_docking.core.state import Pose2D, VehicleCommand, VehicleState
from trailer_docking.safety.supervisor import state_limit_violations

mujoco = pytest.importorskip("mujoco")

from trailer_docking.simulators.mujoco.adapter import (  # noqa: E402
    MujocoSimulator,
)
from trailer_docking.simulators.mujoco.builder import (  # noqa: E402
    DOCK_WALL_GEOMS,
    TRACTOR_COLLISION_GEOM,
    TRAILER_COLLISION_GEOM,
    build_vehicle_xml,
    generated_files,
)

STOP = VehicleCommand(0.0, 0.0)


@pytest.fixture
def yard(vehicle_config: VehicleConfiguration) -> MujocoSimulator:
    """Vehicle on an empty flat yard, settled at the neutral pose."""

    simulator = MujocoSimulator(vehicle_config)
    simulator.reset(settle_time=0.5)
    return simulator


@pytest.fixture
def docked_scene(
    vehicle_config: VehicleConfiguration, dock_config: DockConfiguration
) -> MujocoSimulator:
    return MujocoSimulator(vehicle_config, dock_config)


def _run(
    simulator: MujocoSimulator, command: VehicleCommand, steps: int
) -> VehicleState:
    state = simulator.state
    for _ in range(steps):
        state = simulator.step(command)
    return state


# ----------------------------------------------------------- model contract


def test_committed_models_compile(
    vehicle_config: VehicleConfiguration, dock_config: DockConfiguration
) -> None:
    for path in generated_files(vehicle_config, dock_config):
        model = mujoco.MjModel.from_xml_path(str(path))
        assert model.nu == 4


def test_neutral_semantic_sites_match_configuration(
    vehicle_config: VehicleConfiguration,
) -> None:
    model = mujoco.MjModel.from_xml_string(build_vehicle_xml(vehicle_config))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for site in vehicle_config.neutral_pose_sites:
        identifier = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site.name)
        assert identifier >= 0
        assert data.site_xpos[identifier] == pytest.approx(
            (site.x, site.y, site.z), abs=1e-9
        )


@pytest.mark.parametrize("variant", ["short", "nominal", "long"])
def test_trailer_variants_share_one_parameterized_model(
    vehicle_config: VehicleConfiguration, variant: str
) -> None:
    vehicle = vehicle_config.with_trailer_variant(variant)
    simulator = MujocoSimulator(vehicle)
    for site in vehicle.neutral_pose_sites:
        assert simulator.semantic_site_position(site.name) == pytest.approx(
            (site.x, site.y, site.z), abs=1e-9
        )
    residuals = vehicle.geometry.constraint_residuals(simulator.state)
    assert max(residuals.values()) < 1e-9


def test_collision_boxes_match_planar_footprints_and_masses(
    yard: MujocoSimulator, vehicle_config: VehicleConfiguration
) -> None:
    model, data = yard.model, yard.data
    state = yard.reset()
    cases = (
        (
            TRACTOR_COLLISION_GEOM,
            vehicle_config.tractor,
            state.tractor_pose,
            vehicle_config.mujoco.tractor_chassis_mass,
        ),
        (
            TRAILER_COLLISION_GEOM,
            vehicle_config.trailer,
            state.trailer_pose,
            vehicle_config.mujoco.trailer_body_mass,
        ),
    )
    for name, geometry, pose, mass in cases:
        identifier = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        center = data.geom_xpos[identifier]
        half = model.geom_size[identifier]
        corners = geometry.footprint(pose)
        assert min(corner.x for corner in corners) == pytest.approx(center[0] - half[0])
        assert max(corner.x for corner in corners) == pytest.approx(center[0] + half[0])
        assert min(corner.y for corner in corners) == pytest.approx(center[1] - half[1])
        assert max(corner.y for corner in corners) == pytest.approx(center[1] + half[1])
        assert center[2] - half[2] == pytest.approx(geometry.body_bottom_height)
        assert 2.0 * half[2] == pytest.approx(geometry.body_height)
        body = model.geom_bodyid[identifier]
        assert model.body_mass[body] == pytest.approx(mass)
    wheel = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rear_left_wheel")
    assert model.body_mass[wheel] == pytest.approx(
        vehicle_config.mujoco.wheel_mass_each
    )


def test_physical_limits_and_timing_match_configuration(
    yard: MujocoSimulator, vehicle_config: VehicleConfiguration
) -> None:
    model = yard.model
    assert model.opt.timestep == pytest.approx(0.002)
    assert tuple(model.opt.gravity) == pytest.approx(vehicle_config.mujoco.gravity)
    hitch = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hitch_yaw")
    mechanical = vehicle_config.limits.mujoco_hitch_mechanical_abs_max
    assert tuple(model.jnt_range[hitch]) == pytest.approx((-mechanical, mechanical))
    for name, torque in (
        ("front_left_steer_position", vehicle_config.mujoco.steering_torque_each),
        ("rear_right_wheel_velocity", vehicle_config.mujoco.rear_wheel_torque_each),
    ):
        actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        assert tuple(model.actuator_forcerange[actuator]) == pytest.approx(
            (-torque, torque)
        )


# ------------------------------------------------------- reset and timing


def test_reset_reproduces_the_shared_assembled_state(
    yard: MujocoSimulator, vehicle_config: VehicleConfiguration
) -> None:
    pose = Pose2D(3.0, -2.0, 0.7)
    state = yard.reset(pose, 0.4, steering_angle=-0.25)
    expected = vehicle_config.geometry.assembled_state(pose, 0.4, steering_angle=-0.25)
    assert state.tractor_pose.to_dict() == pytest.approx(
        expected.tractor_pose.to_dict(), abs=1e-9
    )
    assert state.trailer_pose.to_dict() == pytest.approx(
        expected.trailer_pose.to_dict(), abs=1e-9
    )
    assert state.hitch_position.to_dict() == pytest.approx(
        expected.hitch_position.to_dict(), abs=1e-9
    )
    assert state.steering_angle == pytest.approx(-0.25, abs=1e-9)
    assert state.articulation_angle == pytest.approx(0.3, abs=1e-9)
    assert state.simulation_time == 0.0
    assert state.longitudinal_speed == pytest.approx(0.0, abs=1e-9)


def test_vehicle_rests_on_the_ground_without_drifting(yard: MujocoSimulator) -> None:
    state = _run(yard, STOP, 30)
    assert abs(state.tractor_pose.x) < 1e-3
    assert abs(state.tractor_pose.y) < 1e-3
    assert abs(state.tractor_pose.yaw) < 1e-4
    assert abs(state.articulation_angle) < 1e-4
    assert abs(state.longitudinal_speed) < 1e-3
    assert yard.semantic_site_position("tractor_rear_axle")[2] == pytest.approx(
        0.30, abs=2e-3
    )
    assert yard.semantic_site_position("docking_point")[2] == pytest.approx(
        0.0, abs=2e-3
    )


def test_one_step_is_one_controller_period(
    yard: MujocoSimulator, vehicle_config: VehicleConfiguration
) -> None:
    assert vehicle_config.timing.mujoco_substeps_per_control == 50
    for expected_steps in (1, 2, 3):
        state = yard.step(STOP)
        assert state.simulation_time == pytest.approx(0.1 * expected_steps)


def test_fixed_command_sequence_is_repeatable(
    vehicle_config: VehicleConfiguration,
) -> None:
    def rollout() -> dict[str, object]:
        simulator = MujocoSimulator(vehicle_config)
        simulator.reset()
        state = simulator.state
        for index in range(40):
            state = simulator.step(VehicleCommand(0.2 if index < 8 else -0.1, -0.2))
        return state.to_dict()

    assert rollout() == rollout()


# -------------------------------------------------------- sign conventions


def test_speed_sign_and_tracking(yard: MujocoSimulator) -> None:
    forward = _run(yard, VehicleCommand(0.0, 0.2), 40)
    assert forward.tractor_pose.x > 0.5
    assert forward.longitudinal_speed == pytest.approx(0.2, abs=0.005)
    assert abs(forward.tractor_pose.y) < 5e-3
    yard.reset(settle_time=0.5)
    reverse = _run(yard, VehicleCommand(0.0, -0.2), 40)
    assert reverse.tractor_pose.x < -0.5
    assert reverse.longitudinal_speed == pytest.approx(-0.2, abs=0.005)
    assert abs(reverse.trailer_pose.y) < 5e-3
    assert abs(reverse.articulation_angle) < 2e-3


def test_plant_limits_are_enforced_and_reported(
    yard: MujocoSimulator, vehicle_config: VehicleConfiguration
) -> None:
    limits = vehicle_config.limits
    state = yard.step(VehicleCommand(5.0, 5.0))
    # One period of acceleration-limited ramp, not an instantaneous jump, and
    # the applied command reports what the plant actually received.
    ramp = limits.longitudinal_acceleration_abs_max * 0.1
    assert yard.speed_target == pytest.approx(ramp)
    assert yard.applied_command.steering_rate == pytest.approx(
        limits.steering_rate_abs_max
    )
    assert yard.applied_command.target_speed == pytest.approx(ramp)
    assert state.longitudinal_speed < 0.06
    state = _run(yard, VehicleCommand(5.0, 0.0), 40)
    assert yard.steering_target == pytest.approx(limits.steering_angle_abs_max)
    assert state.steering_angle == pytest.approx(
        limits.steering_angle_abs_max, abs=0.01
    )
    # Mechanical stops hold the measured angle at the limit to within
    # joint-limit compliance; it is not an exact bound as in the 2D plant.
    assert abs(state.steering_angle) < limits.steering_angle_abs_max + 2e-3
    # Held at the angle limit, the realised steering rate is zero.
    yard.step(VehicleCommand(5.0, 0.0))
    assert yard.applied_command.steering_rate == pytest.approx(0.0)
    assert "steering_rate_limit" not in state_limit_violations(state, limits)


def test_steering_sign_rate_and_virtual_angle(
    yard: MujocoSimulator, vehicle_config: VehicleConfiguration
) -> None:
    state = _run(yard, VehicleCommand(0.3, 0.0), 10)
    # Positive steering is a left turn; the left wheel is the inner wheel.
    left = yard.data.joint("front_left_steer").qpos[0]
    right = yard.data.joint("front_right_steer").qpos[0]
    assert left > right > 0.0
    assert state.steering_angle == pytest.approx(yard.steering_target, abs=0.01)
    assert yard.steering_target == pytest.approx(0.3)
    assert state.steering_rate == pytest.approx(0.3, abs=0.03)

    turning = _run(yard, VehicleCommand(0.0, 0.2), 60)
    assert turning.tractor_pose.yaw > 0.05
    assert turning.tractor_yaw_rate > 0.0
    # Tractor heading leads the trailer counterclockwise: positive articulation.
    assert turning.articulation_angle > 0.0
    kinematic_yaw_rate = (
        turning.longitudinal_speed
        * tan(turning.steering_angle)
        / vehicle_config.tractor.wheelbase
    )
    assert turning.tractor_yaw_rate == pytest.approx(kinematic_yaw_rate, rel=0.03)


def test_articulation_sign_matches_shared_definition(yard: MujocoSimulator) -> None:
    state = yard.reset(Pose2D(0.0, 0.0, 0.2), -0.1)
    assert state.articulation_angle == pytest.approx(0.3, abs=1e-9)
    state = yard.reset(Pose2D(0.0, 0.0, pi - 0.1), -pi + 0.1)
    assert state.articulation_angle == pytest.approx(-0.2, abs=1e-9)


# ------------------------------------------------- 2D kinematic agreement


def _reference_rollout(
    vehicle: VehicleConfiguration,
    start: VehicleState,
    commands: list[VehicleCommand],
) -> tuple[float, float, float, float]:
    """Integrate the specification's on-axle kinematic equations with RK4.

    This is a test-local reference, not the project's 2D simulator.  It applies
    the same rate, angle, speed, and acceleration limits as the plant.
    """

    limits = vehicle.limits
    wheelbase = vehicle.tractor.wheelbase
    trailer_length = vehicle.trailer.hitch_to_axle
    step = vehicle.timing.mujoco_physics_step
    substeps = vehicle.timing.mujoco_substeps_per_control
    steering_limit = limits.steering_angle_abs_max
    x, y = start.tractor_pose.x, start.tractor_pose.y
    yaw, trailer_yaw = start.tractor_pose.yaw, start.trailer_pose.yaw
    steering, speed = start.steering_angle, 0.0

    def derivative(
        values: tuple[float, float, float, float], delta: float, velocity: float
    ) -> tuple[float, float, float, float]:
        _, _, psi, psi_r = values
        return (
            velocity * cos(psi),
            velocity * sin(psi),
            velocity * tan(delta) / wheelbase,
            velocity * sin(psi - psi_r) / trailer_length,
        )

    for command in commands:
        for _ in range(substeps):
            steering = min(
                max(steering + command.steering_rate * step, -steering_limit),
                steering_limit,
            )
            increment = limits.longitudinal_acceleration_abs_max * step
            speed += min(max(command.target_speed - speed, -increment), increment)
            values = (x, y, yaw, trailer_yaw)
            k1 = derivative(values, steering, speed)
            k2 = derivative(
                tuple(v + 0.5 * step * k for v, k in zip(values, k1, strict=True)),  # type: ignore[arg-type]
                steering,
                speed,
            )
            k3 = derivative(
                tuple(v + 0.5 * step * k for v, k in zip(values, k2, strict=True)),  # type: ignore[arg-type]
                steering,
                speed,
            )
            k4 = derivative(
                tuple(v + step * k for v, k in zip(values, k3, strict=True)),  # type: ignore[arg-type]
                steering,
                speed,
            )
            x, y, yaw, trailer_yaw = (
                v + step * (a + 2.0 * b + 2.0 * c + d) / 6.0
                for v, a, b, c, d in zip(values, k1, k2, k3, k4, strict=True)
            )
    return x, y, yaw, trailer_yaw


@pytest.mark.parametrize("speed", [0.2, -0.2])
def test_low_speed_maneuver_agrees_with_kinematic_equations(
    yard: MujocoSimulator, vehicle_config: VehicleConfiguration, speed: float
) -> None:
    commands = (
        [VehicleCommand(0.3, speed)] * 8
        + [VehicleCommand(0.0, speed)] * 60
        + [VehicleCommand(-0.3, speed)] * 16
        + [VehicleCommand(0.0, speed)] * 66
    )
    start = yard.state
    state = start
    for command in commands:
        state = yard.step(command)
    x, y, yaw, trailer_yaw = _reference_rollout(vehicle_config, start, commands)
    # Exact equality is not expected: MuJoCo adds contact and actuator dynamics.
    assert state.tractor_pose.x == pytest.approx(x, abs=0.05)
    assert state.tractor_pose.y == pytest.approx(y, abs=0.05)
    assert state.tractor_pose.yaw == pytest.approx(yaw, abs=0.02)
    assert state.trailer_pose.yaw == pytest.approx(trailer_yaw, abs=0.02)
    residuals = vehicle_config.geometry.constraint_residuals(state)
    assert max(residuals.values()) < 1e-4


# ---------------------------------------------------------------- the dock


def test_dock_walls_match_environment_configuration(
    docked_scene: MujocoSimulator, dock_config: DockConfiguration
) -> None:
    bay = dock_config.geometry
    model, data = docked_scene.model, docked_scene.data
    boxes = {}
    for name in DOCK_WALL_GEOMS:
        identifier = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        center, half = data.geom_xpos[identifier], model.geom_size[identifier]
        boxes[name] = (center - half, center + half)
        assert center[2] + half[2] == pytest.approx(bay.wall_height)
        assert center[2] - half[2] == pytest.approx(0.0)
    back_low, back_high = boxes["dock_back_wall"]
    assert back_high[0] == pytest.approx(0.0)
    assert back_low[0] == pytest.approx(-bay.wall_thickness)
    left_low, left_high = boxes["dock_left_wall"]
    right_low, right_high = boxes["dock_right_wall"]
    assert left_low[1] == pytest.approx(bay.interior_width / 2.0)
    assert right_high[1] == pytest.approx(-bay.interior_width / 2.0)
    assert left_high[1] - left_low[1] == pytest.approx(bay.wall_thickness)
    assert (left_low[0], left_high[0]) == pytest.approx((0.0, bay.interior_depth))
    assert (right_low[0], right_high[0]) == pytest.approx((0.0, bay.interior_depth))
    target = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "dock_target")
    assert tuple(data.site_xpos[target]) == pytest.approx(
        (bay.target_position.x, bay.target_position.y, 0.0)
    )


def test_scene_start_pose_faces_the_bay_without_contact(
    docked_scene: MujocoSimulator,
    vehicle_config: VehicleConfiguration,
    dock_config: DockConfiguration,
) -> None:
    state = docked_scene.state
    rear = vehicle_config.trailer.rear_face(state.trailer_pose)
    assert rear.x > dock_config.geometry.interior_depth
    assert rear.y == pytest.approx(0.0, abs=1e-9)
    assert state.trailer_pose.yaw == pytest.approx(0.0, abs=1e-9)
    assert docked_scene.wall_contacts() == ()


def _tractor_pose_for_docking_point(
    vehicle: VehicleConfiguration, x: float, y: float
) -> Pose2D:
    offset = vehicle.trailer.rear_overhang + vehicle.trailer.hitch_to_axle
    return Pose2D(x + offset, y, 0.0)


def test_target_pose_is_collision_free_and_wall_contact_is_detected(
    docked_scene: MujocoSimulator,
    vehicle_config: VehicleConfiguration,
    dock_config: DockConfiguration,
) -> None:
    bay = dock_config.geometry
    target = bay.target_position
    docked_scene.reset(
        _tractor_pose_for_docking_point(vehicle_config, target.x, target.y),
        bay.target_trailer_yaw,
        settle_time=0.3,
    )
    assert docked_scene.wall_contacts() == ()
    position = docked_scene.semantic_site_position("docking_point")
    assert position[:2] == pytest.approx((target.x, target.y), abs=2e-3)

    clearance = (bay.interior_width - vehicle_config.trailer.body_width) / 2.0
    docked_scene.reset(
        _tractor_pose_for_docking_point(vehicle_config, target.x, clearance + 0.02),
        0.0,
    )
    assert "trailer_body" in docked_scene.wall_contacts()


def test_reversing_into_the_back_wall_is_stopped_by_contact(
    docked_scene: MujocoSimulator,
    vehicle_config: VehicleConfiguration,
    dock_config: DockConfiguration,
) -> None:
    docked_scene.reset(
        _tractor_pose_for_docking_point(vehicle_config, 0.15, 0.0),
        0.0,
        settle_time=0.3,
    )
    assert docked_scene.wall_contacts() == ()
    state = _run(docked_scene, VehicleCommand(0.0, -0.2), 30)
    assert docked_scene.wall_contacts() == ("trailer_body",)
    rear = vehicle_config.trailer.rear_face(state.trailer_pose)
    # The rear face is held at the wall plane instead of passing through it.
    assert rear.x == pytest.approx(0.0, abs=0.02)
    assert dock_config.geometry.wall_height > vehicle_config.trailer.body_bottom_height
