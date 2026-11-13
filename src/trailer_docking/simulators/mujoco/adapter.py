"""Translate shared commands and states to and from a MuJoCo model.

:class:`MujocoSimulator` is the 3D counterpart of the 2D kinematic backend.  It
accepts the same :class:`VehicleCommand` at the shared controller period and
returns the same :class:`VehicleState`, measured only from named semantic
sites and joints.  A controller or policy developed against the shared
contract therefore needs no MuJoCo-specific observation or action code.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import atan2, cos, isclose, isfinite, sin
from numbers import Real
from typing import Any

import mujoco
import numpy as np

from trailer_docking.core.config import DockConfiguration, VehicleConfiguration
from trailer_docking.core.obstacles import Obstacle
from trailer_docking.core.state import (
    Point2D,
    Pose2D,
    VehicleCommand,
    VehicleState,
    wrap_angle,
)
from trailer_docking.safety.supervisor import MeasurementTolerance
from trailer_docking.simulators.mujoco.ackermann import (
    ackermann_wheel_angles,
    virtual_steering_angle,
    virtual_steering_rate,
)
from trailer_docking.simulators.mujoco.builder import (
    DOCK_WALL_GEOMS,
    MujocoTuning,
    build_scene_xml,
    obstacle_geom_name,
)

# Excess of measured state over the plant limits that this backend produces
# while every command stays within limits (see research/decisions.md, D-007).
# Pass it to ``state_limit_violations`` when checking MuJoCo states.
#
# The values are empirical.  Across lock-to-lock steering sweeps, speed
# reversals, all trailer variants, and off-axle hitches the worst observed
# excess was 0.0012 rad, 0.054 rad/s, and 0.014 m/s; each tolerance leaves
# margin above that.  The steering servo briefly exceeds the commanded rate
# while it catches up after a reversal, which is why the rate margin is the
# largest.  Re-measure after changing actuator gains or geometry.
#
# The tolerance holds only below the jackknife threshold.  With the hitch
# driven onto its mechanical stop (articulation about 1.48 rad) the measured
# steering rate has exceeded the limit by up to 0.098 rad/s.  An episode ends
# at the jackknife threshold long before that, so no tolerance is published
# for the post-jackknife regime.
MUJOCO_MEASUREMENT_TOLERANCE = MeasurementTolerance(
    steering_angle=0.005, steering_rate=0.08, speed=0.03
)


class MujocoModelError(RuntimeError):
    """Raised when a compiled model violates the shared MuJoCo contract."""


def _clip(value: float, minimum: float, maximum: float) -> float:
    return min(max(value, minimum), maximum)


class MujocoSimulator:
    """MuJoCo backend exposing the shared planar state and command contract.

    Plant limits are enforced here exactly as the specification describes for
    the 2D plant: steering rate, steering angle, speed, and longitudinal
    acceleration are bounded before they reach the actuators.  What the plant
    actually received during the last step, after all four limits, is
    available as :attr:`applied_command`.
    """

    def __init__(
        self,
        vehicle: VehicleConfiguration,
        dock: DockConfiguration | None = None,
        tuning: MujocoTuning | None = None,
        *,
        obstacles: Sequence[Obstacle] = (),
    ) -> None:
        if not isclose(vehicle.timing.nominal_command_delay, 0.0, abs_tol=1e-12):
            raise NotImplementedError(
                "a nonzero command delay is not implemented in the MuJoCo backend"
            )
        self.vehicle = vehicle
        self.dock = dock
        self.tuning = tuning or MujocoTuning()
        self.obstacles = tuple(obstacles)
        self.xml = build_scene_xml(vehicle, dock, self.tuning, obstacles=self.obstacles)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self._substeps = vehicle.timing.mujoco_substeps_per_control
        if not isclose(
            self.model.opt.timestep, vehicle.timing.mujoco_physics_step, rel_tol=1e-12
        ):
            raise MujocoModelError("compiled timestep disagrees with configuration")

        self._validate_required_names()
        self._site = {
            name: self._id(mujoco.mjtObj.mjOBJ_SITE, name)
            for name in vehicle.mujoco.required_names.sites
        }
        joint = mujoco.mjtObj.mjOBJ_JOINT
        self._free_qpos = int(self.model.jnt_qposadr[self._id(joint, "tractor_free")])
        self._free_qvel = int(self.model.jnt_dofadr[self._id(joint, "tractor_free")])
        self._qpos = {
            name: int(self.model.jnt_qposadr[self._id(joint, name)])
            for name in ("hitch_yaw", "front_left_steer", "front_right_steer")
        }
        self._qvel = {
            name: int(self.model.jnt_dofadr[self._id(joint, name)])
            for name in ("front_left_steer", "front_right_steer")
        }
        actuator = mujoco.mjtObj.mjOBJ_ACTUATOR
        self._ctrl = {
            name: self._id(actuator, name)
            for name in vehicle.mujoco.required_names.actuators
        }
        geom = mujoco.mjtObj.mjOBJ_GEOM
        # Obstacles are reported by ``wall_contacts`` like dock walls.
        fixed = [*(DOCK_WALL_GEOMS if dock is not None else ())]
        fixed += [obstacle_geom_name(index) for index in range(len(self.obstacles))]
        self._wall_geoms = frozenset(self._id(geom, name) for name in fixed)
        self._steering_target = 0.0
        self._speed_target = 0.0
        self._applied = VehicleCommand(0.0, 0.0)
        self.reset()

    # ------------------------------------------------------------------ setup

    def _id(self, kind: Any, name: str) -> int:
        identifier = int(mujoco.mj_name2id(self.model, kind, name))
        if identifier < 0:
            raise MujocoModelError(f"compiled model has no {kind.name} named {name!r}")
        return identifier

    def _validate_required_names(self) -> None:
        names = self.vehicle.mujoco.required_names
        for kind, required in (
            (mujoco.mjtObj.mjOBJ_BODY, names.bodies),
            (mujoco.mjtObj.mjOBJ_JOINT, names.joints),
            (mujoco.mjtObj.mjOBJ_SITE, names.sites),
            (mujoco.mjtObj.mjOBJ_ACTUATOR, names.actuators),
        ):
            for name in required:
                self._id(kind, name)

    # ------------------------------------------------------------ properties

    @property
    def controller_period(self) -> float:
        return self.vehicle.timing.controller_period

    @property
    def steering_target(self) -> float:
        """Virtual steering angle currently commanded to the wheel actuators."""

        return self._steering_target

    @property
    def speed_target(self) -> float:
        """Acceleration-limited speed currently commanded to the drive wheels."""

        return self._speed_target

    @property
    def applied_command(self) -> VehicleCommand:
        """What the actuators effectively received during the last step.

        ``steering_rate`` is the realised rate of the virtual steering target
        over the period (zero when held at the angle limit) and
        ``target_speed`` is the acceleration-limited speed target reached at
        the end of the period.
        """

        return self._applied

    # ------------------------------------------------------------------ reset

    def reset(
        self,
        tractor_pose: Pose2D | None = None,
        trailer_yaw: float | None = None,
        *,
        steering_angle: float = 0.0,
        settle_time: float = 0.0,
    ) -> VehicleState:
        """Place the vehicle at rest in a geometrically consistent pose.

        The arguments mirror :meth:`VehicleGeometry.assembled_state`.  Without a
        pose the model's own initial pose is used.  ``settle_time`` optionally
        lets contacts come to rest (with zero speed target) before simulation
        time is reset to zero.

        Raises :class:`ValueError` for a steering angle beyond the shared limit
        or an articulation beyond the hitch's mechanical range, because the
        model cannot represent either pose.
        """

        # Validate every argument before touching simulator state, so a
        # rejected call leaves the previous episode fully intact.
        if tractor_pose is not None and not isinstance(tractor_pose, Pose2D):
            raise ValueError("tractor_pose must be a Pose2D")
        limit = self.vehicle.limits.steering_angle_abs_max
        for name, value in (
            ("steering_angle", steering_angle),
            ("settle_time", settle_time),
        ):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} must be a real number")
        try:
            steering = float(steering_angle)
            settle = float(settle_time)
        except OverflowError as error:
            raise ValueError("steering_angle and settle_time must be finite") from error
        if not abs(steering) <= limit:
            raise ValueError(
                f"steering_angle={steering} exceeds the +-{limit} rad limit"
            )
        if not (isfinite(settle) and settle >= 0.0):
            raise ValueError("settle_time must be finite and non-negative")
        if trailer_yaw is not None:
            # wrap_angle rejects non-finite and non-numeric input.
            trailer_yaw = wrap_angle(trailer_yaw)
            start_yaw = (
                tractor_pose.yaw
                if tractor_pose is not None
                else self._model_start_yaw()
            )
            hitch_angle = wrap_angle(trailer_yaw - start_yaw)
            mechanical = self.vehicle.limits.mujoco_hitch_mechanical_abs_max
            if not abs(hitch_angle) <= mechanical:
                raise ValueError(
                    f"articulation {-hitch_angle} rad exceeds the hitch's"
                    f" +-{mechanical} rad mechanical range"
                )
        mujoco.mj_resetData(self.model, self.data)
        base = self._free_qpos
        if tractor_pose is not None:
            self.data.qpos[base : base + 3] = (tractor_pose.x, tractor_pose.y, 0.0)
            self.data.qpos[base + 3 : base + 7] = (
                cos(tractor_pose.yaw / 2.0),
                0.0,
                0.0,
                sin(tractor_pose.yaw / 2.0),
            )
        quaternion = self.data.qpos[base + 3 : base + 7]
        tractor_yaw = 2.0 * atan2(quaternion[3], quaternion[0])
        if trailer_yaw is not None:
            # The hinge coordinate is trailer yaw relative to the tractor, the
            # negative of the shared articulation angle.
            self.data.qpos[self._qpos["hitch_yaw"]] = wrap_angle(
                trailer_yaw - wrap_angle(tractor_yaw)
            )
        left, right = self._wheel_angles(steering)
        self.data.qpos[self._qpos["front_left_steer"]] = left
        self.data.qpos[self._qpos["front_right_steer"]] = right
        self._steering_target = steering
        self._speed_target = 0.0
        self._applied = VehicleCommand(0.0, 0.0)
        self._write_controls()
        mujoco.mj_forward(self.model, self.data)
        if settle > 0.0:
            for _ in range(round(settle / self.model.opt.timestep)):
                mujoco.mj_step(self.model, self.data)
            self.data.time = 0.0
            mujoco.mj_forward(self.model, self.data)
        return self.state

    # ------------------------------------------------------------------- step

    def step(self, command: VehicleCommand) -> VehicleState:
        """Advance exactly one controller period under a held command."""

        limits = self.vehicle.limits
        rate = _clip(
            command.steering_rate,
            -limits.steering_rate_abs_max,
            limits.steering_rate_abs_max,
        )
        speed = _clip(
            command.target_speed,
            -limits.reverse_speed_abs_max,
            limits.forward_speed_max,
        )
        timestep = float(self.model.opt.timestep)
        speed_increment = limits.longitudinal_acceleration_abs_max * timestep
        steering_before = self._steering_target
        for _ in range(self._substeps):
            self._steering_target = _clip(
                self._steering_target + rate * timestep,
                -limits.steering_angle_abs_max,
                limits.steering_angle_abs_max,
            )
            self._speed_target += _clip(
                speed - self._speed_target, -speed_increment, speed_increment
            )
            self._write_controls()
            mujoco.mj_step(self.model, self.data)
        # mj_step integrates positions and velocities but leaves site poses,
        # body velocities, and contacts from the start of its last substep.
        # Recompute them so every field of the returned state is at time t.
        mujoco.mj_forward(self.model, self.data)
        self._applied = VehicleCommand(
            steering_rate=(self._steering_target - steering_before)
            / self.controller_period,
            target_speed=self._speed_target,
        )
        return self.state

    def _model_start_yaw(self) -> float:
        """Tractor yaw of the model's own initial pose."""

        quaternion = self.model.qpos0[self._free_qpos + 3 : self._free_qpos + 7]
        return 2.0 * atan2(float(quaternion[3]), float(quaternion[0]))

    def _wheel_angles(self, virtual_angle: float) -> tuple[float, float]:
        tractor = self.vehicle.tractor
        return ackermann_wheel_angles(
            virtual_angle, tractor.wheelbase, tractor.front_track
        )

    def _write_controls(self) -> None:
        tractor = self.vehicle.tractor
        left, right = self._wheel_angles(self._steering_target)
        self.data.ctrl[self._ctrl["front_left_steer_position"]] = left
        self.data.ctrl[self._ctrl["front_right_steer_position"]] = right
        # omega = speed / wheel_radius at the rear-axle midpoint; the optional
        # differential term gives each wheel the speed of its own contact
        # point on the commanded turning circle.
        yaw_rate = 0.0
        if self.tuning.rear_differential:
            yaw_rate = (
                self._speed_target * np.tan(self._steering_target) / tractor.wheelbase
            )
        half_track = tractor.rear_track / 2.0
        radius = self.vehicle.wheels.radius
        self.data.ctrl[self._ctrl["rear_left_wheel_velocity"]] = (
            self._speed_target - yaw_rate * half_track
        ) / radius
        self.data.ctrl[self._ctrl["rear_right_wheel_velocity"]] = (
            self._speed_target + yaw_rate * half_track
        ) / radius

    # ------------------------------------------------------------------ state

    def _site_pose(self, name: str) -> Pose2D:
        identifier = self._site[name]
        position = self.data.site_xpos[identifier]
        rotation = self.data.site_xmat[identifier]
        # Row-major 3x3: element (1, 0) is index 3, element (0, 0) is index 0.
        yaw = atan2(float(rotation[3]), float(rotation[0]))
        return Pose2D(float(position[0]), float(position[1]), yaw)

    def _site_velocity(self, name: str) -> np.ndarray:
        """Return ``[angular(3), linear(3)]`` in world-aligned axes."""

        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_SITE,
            self._site[name],
            velocity,
            0,
        )
        return velocity

    @property
    def state(self) -> VehicleState:
        """Measure the shared planar state from semantic sites and joints."""

        tractor = self.vehicle.tractor
        tractor_pose = self._site_pose("tractor_rear_axle")
        trailer_pose = self._site_pose("trailer_axle")
        hitch = self.data.site_xpos[self._site["hitch"]]
        tractor_velocity = self._site_velocity("tractor_rear_axle")
        trailer_velocity = self._site_velocity("trailer_axle")
        left = float(self.data.qpos[self._qpos["front_left_steer"]])
        right = float(self.data.qpos[self._qpos["front_right_steer"]])
        left_rate = float(self.data.qvel[self._qvel["front_left_steer"]])
        right_rate = float(self.data.qvel[self._qvel["front_right_steer"]])
        return VehicleState(
            tractor_pose=tractor_pose,
            trailer_pose=trailer_pose,
            hitch_position=Point2D(float(hitch[0]), float(hitch[1])),
            steering_angle=virtual_steering_angle(
                left, right, tractor.wheelbase, tractor.front_track
            ),
            steering_rate=virtual_steering_rate(
                left,
                right,
                left_rate,
                right_rate,
                tractor.wheelbase,
                tractor.front_track,
            ),
            longitudinal_speed=float(
                tractor_velocity[3] * cos(tractor_pose.yaw)
                + tractor_velocity[4] * sin(tractor_pose.yaw)
            ),
            tractor_yaw_rate=float(tractor_velocity[2]),
            trailer_yaw_rate=float(trailer_velocity[2]),
            simulation_time=float(self.data.time),
        )

    def semantic_site_position(self, name: str) -> tuple[float, float, float]:
        """Return the world position of one required semantic site."""

        position = self.data.site_xpos[self._site[name]]
        return float(position[0]), float(position[1]), float(position[2])

    def wall_contacts(self) -> tuple[str, ...]:
        """Return sorted names of vehicle bodies touching a wall or obstacle."""

        touching: set[str] = set()
        for index in range(int(self.data.ncon)):
            contact = self.data.contact[index]
            if contact.dist > 0.0:
                continue
            first, second = int(contact.geom1), int(contact.geom2)
            if first in self._wall_geoms:
                other = second
            elif second in self._wall_geoms:
                other = first
            else:
                continue
            body = int(self.model.geom_bodyid[other])
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
            touching.add(str(name))
        return tuple(sorted(touching))
