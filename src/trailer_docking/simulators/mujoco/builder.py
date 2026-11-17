"""Deterministic MJCF generation from the shared vehicle and dock configuration.

The YAML specifications are the numeric authority.  This module turns a
validated :class:`VehicleConfiguration` (and optionally a
:class:`DockConfiguration`) into MJCF text, so every dimension, mass, limit,
and name in the compiled model traces back to one versioned source and the
trailer variants share one code path.

The committed files under ``models/`` and ``scenes/`` are generated output kept
for inspection and viewer use.  Regenerate them with::

    python -m trailer_docking.simulators.mujoco.builder --write

A parity test fails if they drift from this generator.  This module does not
import MuJoCo; it only produces text.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ElementTree
from collections.abc import Sequence
from dataclasses import dataclass
from math import cos, isclose, sin
from pathlib import Path
from xml.sax.saxutils import escape

from trailer_docking.core.config import (
    ConfigurationError,
    DockConfiguration,
    VehicleConfiguration,
    load_dock_configuration,
    load_vehicle_configuration,
)
from trailer_docking.core.obstacles import Obstacle, obstacle_name
from trailer_docking.core.state import Pose2D
from trailer_docking.simulators.mujoco.ackermann import ackermann_wheel_angles

PACKAGE_DIRECTORY = Path(__file__).resolve().parent
VEHICLE_MODEL_PATH = PACKAGE_DIRECTORY / "models" / "compact_semitrailer_v1.xml"
DOCK_SCENE_PATH = PACKAGE_DIRECTORY / "scenes" / "nominal_dock_v1.xml"

TRACTOR_COLLISION_GEOM = "tractor_chassis_collision"
TRAILER_COLLISION_GEOM = "trailer_body_collision"
GROUND_GEOM = "ground"
DOCK_WALL_GEOMS = ("dock_back_wall", "dock_left_wall", "dock_right_wall")
DOCK_WALL_VISUAL_SUFFIX = "_visual"
DOCK_ORIGIN_SITE = "dock_origin"
DOCK_TARGET_SITE = "dock_target"


@dataclass(frozen=True, slots=True)
class MujocoTuning:
    """MuJoCo-only modeling choices that are not part of the frozen contract.

    The specification allows damping, armature, actuator gains, inertias, and
    solver settings to be tuned without an interface change.  They are
    collected here, with their rationale in ``research/decisions.md``, until a
    schema revision moves them into the vehicle YAML.
    """

    # Steering links carry no specified mass; a small inertia keeps the joint
    # well conditioned without affecting axle loads.
    steer_link_mass: float = 2.0
    steer_link_inertia: float = 0.05
    steering_kp: float = 20000.0
    steering_damping: float = 120.0
    steering_armature: float = 1.0
    # Mechanical steering stops sit exactly at the Ackermann angles of the
    # virtual steering limit (plus this margin), so the measured virtual angle
    # cannot run past the shared limit by more than joint-limit compliance.
    steering_range_margin: float = 0.0
    drive_kv: float = 1500.0
    wheel_spin_damping: float = 0.05
    wheel_armature: float = 0.05
    # A full-width rigid cylinder touches the ground at both rims.  On a curve
    # or while steering those two points must scrub, which loads the steering
    # actuators beyond their torque limit and has no counterpart in the 2D
    # model.  Collision therefore uses a thin cylinder of this width on the
    # wheel centre plane; the full-width cylinder carries the mass and visuals.
    wheel_contact_width: float = 0.02
    hitch_damping: float = 5.0
    # Elliptic cones with a raised impedance ratio suppress lateral tire creep,
    # which is the closest contact analogue of the kinematic no-slip model.
    friction_cone: str = "elliptic"
    impedance_ratio: float = 10.0
    integrator: str = "implicitfast"
    # Rear wheels receive individual speed targets consistent with the virtual
    # steering angle so the rear-axle midpoint tracks the commanded speed.
    rear_differential: bool = True
    # Distance of the docking point outside the bay opening in the scene's
    # default start pose.
    scene_start_standoff: float = 4.0


def _n(value: float) -> str:
    """Format a number deterministically without negative zero."""

    number = float(value)
    if number == 0.0:
        return "0"
    # Prefer the short form when it is exact; otherwise keep full precision so
    # values such as a 1/300 s timestep survive the round trip through MJCF.
    text = format(number, ".15g")
    return text if float(text) == number else repr(number)


def _attribute(text: str) -> str:
    """Escape free text for use inside a double-quoted XML attribute."""

    return escape(text, {'"': "&quot;"})


def _comment(text: str) -> str:
    """Make free text safe inside an XML comment (no ``--``)."""

    while "--" in text:
        text = text.replace("--", "- -")
    return text.removesuffix("-")


def _v(*values: float) -> str:
    return " ".join(_n(value) for value in values)


def steering_joint_ranges(
    vehicle: VehicleConfiguration, tuning: MujocoTuning
) -> dict[str, tuple[float, float]]:
    """Physical steering ranges that admit exactly the virtual steering range.

    At full left lock the left wheel is the inner wheel and turns further than
    the right one, and vice versa, so each joint has an asymmetric range.
    """

    inner, outer = ackermann_wheel_angles(
        vehicle.limits.steering_angle_abs_max,
        vehicle.tractor.wheelbase,
        vehicle.tractor.front_track,
    )
    margin = tuning.steering_range_margin
    return {
        "left": (-outer - margin, inner + margin),
        "right": (-inner - margin, outer + margin),
    }


def default_scene_start_pose(
    vehicle: VehicleConfiguration, dock: DockConfiguration, tuning: MujocoTuning
) -> Pose2D:
    """Straight start with the trailer rear facing the bay opening."""

    docking_point_x = dock.geometry.interior_depth + tuning.scene_start_standoff
    tractor_x = (
        docking_point_x
        + vehicle.trailer.rear_overhang
        + vehicle.trailer.hitch_to_axle
        - vehicle.tractor.hitch_longitudinal
    )
    return Pose2D(tractor_x, -vehicle.tractor.hitch_lateral, 0.0)


def _header(vehicle: VehicleConfiguration, source: str) -> list[str]:
    return [
        "<!--",
        "  GENERATED FILE - do not edit by hand.",
        f"  Source: {source}",
        f"  Vehicle specification: {_comment(vehicle.specification_id)}"
        f" (trailer variant: {_comment(vehicle.trailer.variant)})",
        "  Regenerate with the write option of",
        "  trailer_docking.simulators.mujoco.builder (see its module docstring).",
        "-->",
    ]


def _options(vehicle: VehicleConfiguration, tuning: MujocoTuning) -> list[str]:
    friction = vehicle.mujoco.contact_friction
    wheels = vehicle.wheels
    return [
        '  <compiler angle="radian" autolimits="true"/>',
        f'  <option timestep="{_n(vehicle.timing.mujoco_physics_step)}"'
        f' gravity="{_v(*vehicle.mujoco.gravity)}"'
        f' integrator="{tuning.integrator}" cone="{tuning.friction_cone}"'
        f' impratio="{_n(tuning.impedance_ratio)}"/>',
        "  <default>",
        # Collision masks: wheels touch only the ground, chassis boxes touch
        # only dock walls.  The wheels sit inside the chassis footprint, so
        # without masks they would collide with their own vehicle.
        '    <default class="wheel">',
        '      <geom type="cylinder"'
        f' size="{_v(wheels.radius, tuning.wheel_contact_width / 2.0)}"'
        f' zaxis="0 1 0" mass="0" friction="{_v(*friction)}" condim="3"'
        ' contype="1" conaffinity="0" group="3"/>',
        "    </default>",
        '    <default class="wheel_visual">',
        f'      <geom type="cylinder" size="{_v(wheels.radius, wheels.width / 2.0)}"'
        f' zaxis="0 1 0" mass="{_n(vehicle.mujoco.wheel_mass_each)}"'
        ' contype="0" conaffinity="0" rgba="0.12 0.12 0.12 1"/>',
        "    </default>",
        '    <default class="chassis">',
        f'      <geom type="box" friction="{_v(*friction)}" contype="2"'
        ' conaffinity="0"/>',
        "    </default>",
        '    <default class="wheel_spin">',
        f'      <joint type="hinge" axis="0 1 0"'
        f' damping="{_n(tuning.wheel_spin_damping)}"'
        f' armature="{_n(tuning.wheel_armature)}"/>',
        "    </default>",
        '    <default class="semantic">',
        '      <site type="sphere" size="0.04" rgba="1 0.8 0 1"/>',
        "    </default>",
        "  </default>",
    ]


def _wheel_body(name: str, position: str, indent: str) -> list[str]:
    return [
        f'{indent}<body name="{name}" pos="{position}">',
        f'{indent}  <joint name="{name}_spin" class="wheel_spin"/>',
        f'{indent}  <geom name="{name}_collision" class="wheel"/>',
        f'{indent}  <geom name="{name}_visual" class="wheel_visual"/>',
        f"{indent}</body>",
    ]


def _steered_wheel(
    side: str,
    position: str,
    limits: tuple[float, float],
    tuning: MujocoTuning,
    indent: str,
) -> list[str]:
    inertia = _n(tuning.steer_link_inertia)
    return [
        f'{indent}<body name="front_{side}_steer_link" pos="{position}">',
        f'{indent}  <inertial pos="0 0 0" mass="{_n(tuning.steer_link_mass)}"'
        f' diaginertia="{inertia} {inertia} {inertia}"/>',
        f'{indent}  <joint name="front_{side}_steer" type="hinge" axis="0 0 1"'
        f' range="{_v(*limits)}" damping="{_n(tuning.steering_damping)}"'
        f' armature="{_n(tuning.steering_armature)}"/>',
        *_wheel_body(f"front_{side}_wheel", "0 0 0", indent + "  "),
        f"{indent}</body>",
    ]


def _vehicle_body(
    vehicle: VehicleConfiguration, tuning: MujocoTuning, pose: Pose2D
) -> list[str]:
    """Return the tractor body tree.

    The tractor body frame is the ground-plane projection of the rear-axle
    midpoint; the trailer body frame is the hitch point.  Both use +x forward,
    +y left, +z up.
    """

    tractor = vehicle.tractor
    trailer = vehicle.trailer
    radius = vehicle.wheels.radius
    hitch = (tractor.hitch_longitudinal, tractor.hitch_lateral, tractor.hitch_height)
    steer_ranges = steering_joint_ranges(vehicle, tuning)

    tractor_center = (
        tractor.body_center_local.x,
        0.0,
        tractor.body_bottom_height + tractor.body_height / 2.0,
    )
    tractor_half = (
        tractor.footprint_x.span / 2.0,
        tractor.footprint_y.span / 2.0,
        tractor.body_height / 2.0,
    )
    # Trailer quantities are expressed in the hitch frame.
    axle_x = -trailer.hitch_to_axle
    below_hitch = -tractor.hitch_height
    trailer_center = (
        axle_x + trailer.body_center_local.x,
        0.0,
        trailer.body_bottom_height + trailer.body_height / 2.0 + below_hitch,
    )
    trailer_half = (
        trailer.footprint_x.span / 2.0,
        trailer.footprint_y.span / 2.0,
        trailer.body_height / 2.0,
    )
    rear_face_x = axle_x - trailer.rear_overhang
    quaternion = _v(cos(pose.yaw / 2.0), 0.0, 0.0, sin(pose.yaw / 2.0))

    lines = [
        f'    <body name="tractor_body" pos="{_v(pose.x, pose.y, 0.0)}"'
        f' quat="{quaternion}">',
        '      <freejoint name="tractor_free"/>',
        f'      <geom name="{TRACTOR_COLLISION_GEOM}" class="chassis"'
        f' pos="{_v(*tractor_center)}" size="{_v(*tractor_half)}"'
        f' mass="{_n(vehicle.mujoco.tractor_chassis_mass)}"'
        ' rgba="0.16 0.36 0.72 1"/>',
        '      <site name="tractor_rear_axle" class="semantic"'
        f' pos="{_v(0.0, 0.0, radius)}"/>',
        '      <site name="tractor_front_axle" class="semantic"'
        f' pos="{_v(tractor.wheelbase, 0.0, radius)}"/>',
        f'      <site name="hitch" class="semantic" pos="{_v(*hitch)}"/>',
    ]
    for side, sign in (("left", 1.0), ("right", -1.0)):
        lines += _steered_wheel(
            side,
            _v(tractor.wheelbase, sign * tractor.front_track / 2.0, radius),
            steer_ranges[side],
            tuning,
            "      ",
        )
    for side, sign in (("left", 1.0), ("right", -1.0)):
        lines += _wheel_body(
            f"rear_{side}_wheel",
            _v(0.0, sign * tractor.rear_track / 2.0, radius),
            "      ",
        )
    mechanical = vehicle.limits.mujoco_hitch_mechanical_abs_max
    lines += [
        f'      <body name="trailer_body" pos="{_v(*hitch)}">',
        '        <joint name="hitch_yaw" type="hinge" axis="0 0 1"'
        f' range="{_v(-mechanical, mechanical)}"'
        f' damping="{_n(tuning.hitch_damping)}"/>',
        f'        <geom name="{TRAILER_COLLISION_GEOM}" class="chassis"'
        f' pos="{_v(*trailer_center)}" size="{_v(*trailer_half)}"'
        f' mass="{_n(vehicle.mujoco.trailer_body_mass)}"'
        ' rgba="0.82 0.82 0.85 0.85"/>',
        '        <site name="trailer_axle" class="semantic"'
        f' pos="{_v(axle_x, 0.0, radius + below_hitch)}"/>',
        '        <site name="trailer_rear_face" class="semantic"'
        f' pos="{_v(rear_face_x, 0.0, trailer_center[2])}"/>',
        '        <site name="docking_point" class="semantic"'
        f' pos="{_v(rear_face_x, 0.0, below_hitch)}" rgba="1 0.2 0.2 1"/>',
    ]
    for side, sign in (("left", 1.0), ("right", -1.0)):
        lines += _wheel_body(
            f"trailer_{side}_wheel",
            _v(axle_x, sign * trailer.axle_track / 2.0, radius + below_hitch),
            "        ",
        )
    lines += ["      </body>", "    </body>"]
    return lines


def _actuators(vehicle: VehicleConfiguration, tuning: MujocoTuning) -> list[str]:
    steer_ranges = steering_joint_ranges(vehicle, tuning)
    steer_torque = vehicle.mujoco.steering_torque_each
    drive_torque = vehicle.mujoco.rear_wheel_torque_each
    lines = ["  <actuator>"]
    for side in ("left", "right"):
        lines.append(
            f'    <position name="front_{side}_steer_position"'
            f' joint="front_{side}_steer" kp="{_n(tuning.steering_kp)}"'
            f' ctrlrange="{_v(*steer_ranges[side])}"'
            f' forcerange="{_v(-steer_torque, steer_torque)}"/>'
        )
    for side in ("left", "right"):
        lines.append(
            f'    <velocity name="rear_{side}_wheel_velocity"'
            f' joint="rear_{side}_wheel_spin" kv="{_n(tuning.drive_kv)}"'
            f' forcerange="{_v(-drive_torque, drive_torque)}"/>'
        )
    lines.append("  </actuator>")
    return lines


def build_vehicle_xml(
    vehicle: VehicleConfiguration,
    tuning: MujocoTuning | None = None,
    *,
    tractor_pose: Pose2D | None = None,
) -> str:
    """Return the scene-independent vehicle model (no ground, dock, or lights)."""

    tuning = tuning or MujocoTuning()
    pose = tractor_pose or Pose2D(0.0, 0.0, 0.0)
    lines = [
        *_header(vehicle, "configs/vehicles (vehicle only)"),
        f'<mujoco model="{_attribute(vehicle.specification_id)}">',
        *_options(vehicle, tuning),
        "  <worldbody>",
        *_vehicle_body(vehicle, tuning, pose),
        "  </worldbody>",
        *_actuators(vehicle, tuning),
        "</mujoco>",
    ]
    return "\n".join(lines) + "\n"


def dock_collision_height(
    vehicle: VehicleConfiguration, dock: DockConfiguration
) -> float:
    """Height of the wall collision volumes.

    Docking collision is a planar concept in the shared contract: a footprint
    either overlaps a wall or it does not.  The collision volumes therefore
    span every chassis box regardless of the configured wall height, which
    only sets how tall the visible wall is.  Otherwise a wall lower than the
    trailer floor would let the trailer pass over it undetected.
    """

    tractor_top = vehicle.tractor.body_bottom_height + vehicle.tractor.body_height
    trailer_top = vehicle.trailer.body_bottom_height + vehicle.trailer.body_height
    return max(dock.geometry.wall_height, tractor_top, trailer_top)


def _dock_geometry(vehicle: VehicleConfiguration, dock: DockConfiguration) -> list[str]:
    """Return dock walls and markers; the dock frame is the world frame."""

    bay = dock.geometry
    half_width = bay.interior_width / 2.0
    half_thickness = bay.wall_thickness / 2.0
    half_depth = bay.interior_depth / 2.0
    boxes = (
        (
            DOCK_WALL_GEOMS[0],
            (-half_thickness, 0.0),
            (half_thickness, half_width + bay.wall_thickness),
        ),
        (
            DOCK_WALL_GEOMS[1],
            (half_depth, half_width + half_thickness),
            (half_depth, half_thickness),
        ),
        (
            DOCK_WALL_GEOMS[2],
            (half_depth, -(half_width + half_thickness)),
            (half_depth, half_thickness),
        ),
    )
    collision_half = dock_collision_height(vehicle, dock) / 2.0
    visual_half = bay.wall_height / 2.0
    lines = []
    for name, (x, y), (half_x, half_y) in boxes:
        # Invisible planar collision volume (group 3 is hidden by default).
        lines.append(
            f'    <geom name="{name}" type="box" contype="0" conaffinity="2"'
            f' group="3" pos="{_v(x, y, collision_half)}"'
            f' size="{_v(half_x, half_y, collision_half)}"/>'
        )
        # Visible wall at the configured height; never collides.
        lines.append(
            f'    <geom name="{name}{DOCK_WALL_VISUAL_SUFFIX}" type="box"'
            f' contype="0" conaffinity="0" rgba="0.55 0.55 0.5 1"'
            f' pos="{_v(x, y, visual_half)}"'
            f' size="{_v(half_x, half_y, visual_half)}"/>'
        )
    lines += [
        f'    <site name="{DOCK_ORIGIN_SITE}" pos="0 0 0" size="0.05" rgba="0 0 0 1"/>',
        f'    <site name="{DOCK_TARGET_SITE}"'
        f' pos="{_v(bay.target_position.x, bay.target_position.y, 0.0)}"'
        ' size="0.06" rgba="0.1 0.8 0.2 1"/>',
        '    <site name="dock_bay_floor" type="box"'
        f' pos="{_v(half_depth, 0.0, 0.0)}"'
        f' size="{_v(half_depth, half_width, 0.002)}" rgba="0.95 0.85 0.2 0.35"/>',
    ]
    return lines


def obstacle_geom_name(index: int) -> str:
    """Name of the collision geom of the obstacle at zero-based *index*."""

    return obstacle_name(index)


def _obstacle_geometry(
    vehicle: VehicleConfiguration, obstacles: Sequence[Obstacle]
) -> list[str]:
    lines = []
    tractor_top = vehicle.tractor.body_bottom_height + vehicle.tractor.body_height
    trailer_top = vehicle.trailer.body_bottom_height + vehicle.trailer.body_height
    for index, obstacle in enumerate(obstacles):
        name = obstacle_geom_name(index)
        # Extruded above both chassis so a low box cannot be driven over,
        # matching the planar collision rule of the 2D task.
        collision_half = max(obstacle.height, tractor_top, trailer_top) / 2.0
        visual_half = obstacle.height / 2.0
        size = (obstacle.length / 2.0, obstacle.width / 2.0)
        euler = _v(0.0, 0.0, obstacle.yaw)
        lines.append(
            f'    <geom name="{name}" type="box" contype="0" conaffinity="2"'
            f' group="3" pos="{_v(obstacle.x, obstacle.y, collision_half)}"'
            f' euler="{euler}" size="{_v(*size, collision_half)}"/>'
        )
        lines.append(
            f'    <geom name="{name}{DOCK_WALL_VISUAL_SUFFIX}" type="box"'
            f' contype="0" conaffinity="0" rgba="0.85 0.45 0.1 1"'
            f' pos="{_v(obstacle.x, obstacle.y, visual_half)}"'
            f' euler="{euler}" size="{_v(*size, visual_half)}"/>'
        )
    return lines


def build_scene_xml(
    vehicle: VehicleConfiguration,
    dock: DockConfiguration | None = None,
    tuning: MujocoTuning | None = None,
    *,
    tractor_pose: Pose2D | None = None,
    obstacles: Sequence[Obstacle] = (),
) -> str:
    """Return a complete self-contained world: ground, vehicle, optional dock.

    *obstacles* are static boxes in the world frame.  Like the dock walls
    they collide with the chassis volumes through an invisible extruded box
    and are shown at their own height.

    With a dock, the world frame is the dock frame (origin at the rear-wall
    center, +x outward).  Without one, the result is an empty flat yard used
    by focused vehicle tests.
    """

    tuning = tuning or MujocoTuning()
    if tractor_pose is not None:
        pose = tractor_pose
    elif dock is not None:
        pose = default_scene_start_pose(vehicle, dock, tuning)
    else:
        pose = Pose2D(0.0, 0.0, 0.0)
    name = dock.geometry.specification_id if dock is not None else "flat-yard"
    source = "configs/vehicles + configs/environments"
    lines = [
        *_header(vehicle, source),
        f'<mujoco model="{_attribute(name)}">',
        *_options(vehicle, tuning),
        "  <asset>",
        '    <texture name="yard" type="2d" builtin="checker" width="256"'
        ' height="256" rgb1="0.36 0.38 0.4" rgb2="0.32 0.34 0.36"/>',
        '    <material name="yard" texture="yard" texrepeat="1 1" texuniform="true"'
        ' reflectance="0"/>',
        "  </asset>",
        "  <visual>",
        '    <headlight ambient="0.45 0.45 0.45" diffuse="0.6 0.6 0.6"/>',
        '    <map zfar="200"/>',
        "  </visual>",
        "  <worldbody>",
        '    <light name="sun" pos="0 0 30" dir="0.2 0.1 -1" directional="true"/>',
        f'    <geom name="{GROUND_GEOM}" type="plane" size="0 0 1" material="yard"'
        f' friction="{_v(*vehicle.mujoco.contact_friction)}" condim="3"'
        ' contype="0" conaffinity="1"/>',
        '    <camera name="overhead" pos="9 0 26" xyaxes="1 0 0 0 1 0"/>',
        '    <camera name="yard" pos="9 -16 9" xyaxes="1 0 0 0 0.469 0.883"/>',
    ]
    if dock is not None:
        lines += _dock_geometry(vehicle, dock)
    lines += _obstacle_geometry(vehicle, obstacles)
    lines += [
        *_vehicle_body(vehicle, tuning, pose),
        "  </worldbody>",
        *_actuators(vehicle, tuning),
        "</mujoco>",
    ]
    return "\n".join(lines) + "\n"


def generated_files(
    vehicle: VehicleConfiguration, dock: DockConfiguration
) -> dict[Path, str]:
    """Return the committed generated files and their expected contents."""

    return {
        VEHICLE_MODEL_PATH: build_vehicle_xml(vehicle),
        DOCK_SCENE_PATH: build_scene_xml(vehicle, dock),
    }


def _numbers(text: str) -> list[float] | None:
    try:
        return [float(token) for token in text.split()]
    except ValueError:
        return None


def mjcf_equivalent(first: str, second: str, *, tolerance: float = 1e-9) -> bool:
    """Compare two MJCF documents structurally, numbers within a tolerance.

    Derived values such as Ackermann steering ranges pass through ``tan`` and
    ``atan``, whose last bit may differ between platforms.  A byte comparison
    of generated files would then report drift where there is none.
    """

    def same(a: ElementTree.Element, b: ElementTree.Element) -> bool:
        if a.tag != b.tag or set(a.attrib) != set(b.attrib) or len(a) != len(b):
            return False
        for key, value in a.attrib.items():
            other = b.attrib[key]
            if value == other:
                continue
            left, right = _numbers(value), _numbers(other)
            if left is None or right is None or len(left) != len(right):
                return False
            if not all(
                isclose(x, y, rel_tol=tolerance, abs_tol=tolerance)
                for x, y in zip(left, right, strict=True)
            ):
                return False
        return all(same(x, y) for x, y in zip(a, b, strict=True))

    try:
        return same(ElementTree.fromstring(first), ElementTree.fromstring(second))
    except ElementTree.ParseError:
        return False


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vehicle",
        type=Path,
        default=Path("configs/vehicles/compact_semitrailer_v1.yaml"),
    )
    parser.add_argument(
        "--dock",
        type=Path,
        default=Path("configs/environments/nominal_dock_v1.yaml"),
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="write the generated files; otherwise only report drift",
    )
    args = parser.parse_args(argv)
    try:
        vehicle = load_vehicle_configuration(args.vehicle)
        dock = load_dock_configuration(args.dock)
    except ConfigurationError as error:
        print(f"error: {error} (run from the repository root or pass paths)")
        return 2
    drift = False
    for path, content in generated_files(vehicle, dock).items():
        current = path.read_text(encoding="utf-8") if path.is_file() else None
        if current is not None and mjcf_equivalent(current, content):
            print(f"up to date: {path}")
        elif args.write:
            path.write_text(content, encoding="utf-8", newline="\n")
            print(f"wrote: {path}")
        else:
            drift = True
            print(f"OUT OF DATE: {path}")
    return 1 if drift else 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
