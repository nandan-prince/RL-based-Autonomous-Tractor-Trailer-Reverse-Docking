"""Strict loaders and typed configuration for frozen project specifications."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from math import isclose, isfinite
from pathlib import Path
from typing import Any

import yaml

from trailer_docking.core.geometry import (
    Bounds1D,
    DockGeometry,
    TractorGeometry,
    TrailerGeometry,
    VehicleGeometry,
    WheelGeometry,
)
from trailer_docking.core.state import Point2D, wrap_angle

CURRENT_SCHEMA_VERSION = 1


class ConfigurationError(ValueError):
    """Raised when a configuration violates the shared specification schema."""


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{path} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ConfigurationError(f"{path} keys must be strings")
    return value


def _exact_keys(data: Mapping[str, Any], expected: set[str], path: str) -> None:
    actual = set(data)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ConfigurationError(
            f"{path} has invalid fields; missing={missing}, unknown={unknown}"
        )


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{path} must be a non-empty string")
    return value


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{path} must be a real number")
    converted = float(value)
    if not isfinite(converted):
        raise ConfigurationError(f"{path} must be finite")
    return converted


def _integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{path} must be an integer")
    return value


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{path} must be a boolean")
    return value


def _number_sequence(value: Any, length: int, path: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigurationError(f"{path} must be a sequence")
    if len(value) != length:
        raise ConfigurationError(f"{path} must contain exactly {length} values")
    return tuple(_number(item, f"{path}[{index}]") for index, item in enumerate(value))


def _string_sequence(value: Any, path: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigurationError(f"{path} must be a sequence")
    result = tuple(_text(item, f"{path}[{index}]") for index, item in enumerate(value))
    if len(result) != len(set(result)):
        raise ConfigurationError(f"{path} cannot contain duplicate names")
    return result


def _bounds(value: Any, path: str) -> Bounds1D:
    minimum, maximum = _number_sequence(value, 2, path)
    try:
        return Bounds1D(minimum, maximum)
    except ValueError as error:
        raise ConfigurationError(f"{path}: {error}") from error


def _normalized_fingerprint(data: Mapping[str, Any]) -> str:
    serialized = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return sha256(serialized.encode("utf-8")).hexdigest()


def _read_yaml(path: str | Path) -> tuple[Mapping[str, Any], Path]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ConfigurationError(f"configuration file does not exist: {resolved}")
    try:
        loaded = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConfigurationError(f"invalid YAML in {resolved}: {error}") from error
    return _mapping(loaded, str(resolved)), resolved


@dataclass(frozen=True, slots=True)
class UnitSystem:
    """Units admitted by schema version 1."""

    length: str
    angle: str
    time: str | None = None
    mass: str | None = None

    def __post_init__(self) -> None:
        if self.length != "m" or self.angle != "rad":
            raise ConfigurationError("schema version 1 requires metres and radians")
        if self.time is not None and self.time != "s":
            raise ConfigurationError("schema version 1 requires seconds")
        if self.mass is not None and self.mass != "kg":
            raise ConfigurationError("schema version 1 requires kilograms")


@dataclass(frozen=True, slots=True)
class CoordinateConventions:
    """Fixed cross-simulator axes and sign conventions."""

    body_forward_axis: str
    body_left_axis: str
    world_up_axis: str
    positive_yaw: str
    positive_steering: str
    positive_speed: str
    articulation: str

    def __post_init__(self) -> None:
        expected = {
            "body_forward_axis": "+x",
            "body_left_axis": "+y",
            "world_up_axis": "+z",
            "positive_yaw": "counterclockwise",
            "positive_steering": "left",
            "positive_speed": "forward",
            "articulation": "wrap(tractor_yaw - trailer_yaw)",
        }
        for name, expected_value in expected.items():
            if getattr(self, name) != expected_value:
                raise ConfigurationError(
                    f"schema version 1 requires {name}={expected_value!r}"
                )


@dataclass(frozen=True, slots=True)
class VehicleLimits:
    """Shared plant and safety limits."""

    steering_angle_abs_max: float
    steering_rate_abs_max: float
    forward_speed_max: float
    reverse_speed_abs_max: float
    longitudinal_acceleration_abs_max: float
    articulation_soft_abs_max: float
    articulation_hard_abs_max: float
    mujoco_hitch_mechanical_abs_max: float

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isfinite(value) or value <= 0.0:
                raise ConfigurationError(f"limits.{name} must be finite and positive")
        if not (
            self.articulation_soft_abs_max
            < self.articulation_hard_abs_max
            < self.mujoco_hitch_mechanical_abs_max
        ):
            raise ConfigurationError(
                "articulation limits must satisfy soft < hard < mechanical"
            )


@dataclass(frozen=True, slots=True)
class SimulationTiming:
    """Controller and backend integration periods."""

    controller_period: float
    kinematic_integration_step: float
    mujoco_physics_step: float
    nominal_command_delay: float

    def __post_init__(self) -> None:
        for name in (
            "controller_period",
            "kinematic_integration_step",
            "mujoco_physics_step",
        ):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0.0:
                raise ConfigurationError(f"timing.{name} must be finite and positive")
        if not isfinite(self.nominal_command_delay) or self.nominal_command_delay < 0.0:
            raise ConfigurationError(
                "timing.nominal_command_delay must be finite and non-negative"
            )
        for step_name, step in (
            ("kinematic_integration_step", self.kinematic_integration_step),
            ("mujoco_physics_step", self.mujoco_physics_step),
        ):
            ratio = self.controller_period / step
            if not isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1e-10):
                raise ConfigurationError(
                    f"controller_period must be an integer multiple of {step_name}"
                )

    @property
    def kinematic_substeps_per_control(self) -> int:
        return round(self.controller_period / self.kinematic_integration_step)

    @property
    def mujoco_substeps_per_control(self) -> int:
        return round(self.controller_period / self.mujoco_physics_step)


@dataclass(frozen=True, slots=True)
class CommandConfiguration:
    """Shared controller command schema and nominal speed target."""

    fields: tuple[str, ...]
    nominal_target_speed: float

    def __post_init__(self) -> None:
        if self.fields != ("steering_rate", "target_speed"):
            raise ConfigurationError(
                "schema version 1 command fields must be steering_rate, target_speed"
            )
        if not isfinite(self.nominal_target_speed):
            raise ConfigurationError("command.nominal_target_speed must be finite")


@dataclass(frozen=True, slots=True)
class NamedPoint3D:
    """A named semantic-site position in the neutral MuJoCo pose."""

    name: str
    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ConfigurationError("semantic site name cannot be empty")
        if not all(isfinite(value) for value in (self.x, self.y, self.z)):
            raise ConfigurationError(f"neutral site {self.name} must be finite")


@dataclass(frozen=True, slots=True)
class MujocoNames:
    """Stable MuJoCo identifiers used by the future adapter."""

    bodies: tuple[str, ...]
    joints: tuple[str, ...]
    sites: tuple[str, ...]
    actuators: tuple[str, ...]

    def __post_init__(self) -> None:
        required = {
            "bodies": {
                "tractor_body",
                "trailer_body",
                "front_left_steer_link",
                "front_right_steer_link",
                "front_left_wheel",
                "front_right_wheel",
                "rear_left_wheel",
                "rear_right_wheel",
                "trailer_left_wheel",
                "trailer_right_wheel",
            },
            "joints": {
                "tractor_free",
                "hitch_yaw",
                "front_left_steer",
                "front_right_steer",
                "front_left_wheel_spin",
                "front_right_wheel_spin",
                "rear_left_wheel_spin",
                "rear_right_wheel_spin",
                "trailer_left_wheel_spin",
                "trailer_right_wheel_spin",
            },
            "sites": {
                "tractor_rear_axle",
                "tractor_front_axle",
                "hitch",
                "trailer_axle",
                "trailer_rear_face",
                "docking_point",
            },
            "actuators": {
                "front_left_steer_position",
                "front_right_steer_position",
                "rear_left_wheel_velocity",
                "rear_right_wheel_velocity",
            },
        }
        for field_name, required_names in required.items():
            configured = set(getattr(self, field_name))
            missing = sorted(required_names - configured)
            if missing:
                raise ConfigurationError(
                    f"mujoco.required_names.{field_name} missing {missing}"
                )


@dataclass(frozen=True, slots=True)
class MujocoParameters:
    """Initial physical parameters not used by the 2D kinematic model."""

    gravity: tuple[float, float, float]
    tractor_chassis_mass: float
    trailer_body_mass: float
    wheel_mass_each: float
    contact_friction: tuple[float, float, float]
    rear_wheel_torque_each: float
    steering_torque_each: float
    required_names: MujocoNames

    def __post_init__(self) -> None:
        for name in (
            "tractor_chassis_mass",
            "trailer_body_mass",
            "wheel_mass_each",
            "rear_wheel_torque_each",
            "steering_torque_each",
        ):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0.0:
                raise ConfigurationError(f"mujoco.{name} must be finite and positive")
        if not all(isfinite(value) for value in self.gravity + self.contact_friction):
            raise ConfigurationError("MuJoCo vectors must be finite")
        if any(value < 0.0 for value in self.contact_friction):
            raise ConfigurationError("MuJoCo contact friction cannot be negative")


@dataclass(frozen=True, slots=True)
class VehicleConfiguration:
    """Fully validated vehicle configuration loaded from one source file."""

    specification_id: str
    schema_version: int
    units: UnitSystem
    conventions: CoordinateConventions
    tractor: TractorGeometry
    trailer: TrailerGeometry
    trailer_variants: tuple[TrailerGeometry, ...]
    wheels: WheelGeometry
    neutral_pose_sites: tuple[NamedPoint3D, ...]
    limits: VehicleLimits
    timing: SimulationTiming
    command: CommandConfiguration
    mujoco: MujocoParameters
    source_path: Path
    source_fingerprint: str

    def __post_init__(self) -> None:
        if self.schema_version != CURRENT_SCHEMA_VERSION:
            raise ConfigurationError(
                f"unsupported vehicle schema_version={self.schema_version}; "
                f"expected {CURRENT_SCHEMA_VERSION}"
            )
        if not self.specification_id.strip():
            raise ConfigurationError("vehicle specification_id cannot be empty")
        names = tuple(variant.variant for variant in self.trailer_variants)
        if len(names) != len(set(names)):
            raise ConfigurationError("trailer variant names must be unique")
        if self.trailer.variant not in names:
            raise ConfigurationError("selected trailer variant is not defined")
        selected = self.trailer_variant(self.trailer.variant)
        for field_name in ("hitch_to_axle", "front_overhang", "rear_overhang"):
            if not isclose(
                getattr(self.trailer, field_name), getattr(selected, field_name)
            ):
                raise ConfigurationError(
                    f"selected trailer does not match variant field {field_name}"
                )
        if not (
            -self.limits.reverse_speed_abs_max
            <= self.command.nominal_target_speed
            <= self.limits.forward_speed_max
        ):
            raise ConfigurationError("nominal target speed exceeds vehicle limits")
        self._validate_neutral_pose_sites()

    @property
    def geometry(self) -> VehicleGeometry:
        return VehicleGeometry(self.tractor, self.trailer, self.wheels)

    @property
    def effective_fingerprint(self) -> str:
        """Fingerprint the source plus selected effective trailer variant."""

        payload = {
            "source_fingerprint": self.source_fingerprint,
            "trailer_variant": self.trailer.variant,
            "trailer_geometry": asdict(self.trailer),
        }
        return _normalized_fingerprint(payload)

    def trailer_variant(self, name: str) -> TrailerGeometry:
        for variant in self.trailer_variants:
            if variant.variant == name:
                return variant
        raise ConfigurationError(f"unknown trailer variant: {name}")

    def with_trailer_variant(self, name: str) -> VehicleConfiguration:
        """Return a validated configuration selecting another named variant."""

        trailer = self.trailer_variant(name)
        geometry = VehicleGeometry(self.tractor, trailer, self.wheels)
        state = geometry.neutral_state()
        rear_face = trailer.rear_face(state.trailer_pose)
        updated_positions = {
            "trailer_axle": (
                state.trailer_pose.x,
                state.trailer_pose.y,
                self.wheels.radius,
            ),
            "trailer_rear_face": (
                rear_face.x,
                rear_face.y,
                trailer.body_bottom_height + trailer.body_height / 2.0,
            ),
            "docking_point": (rear_face.x, rear_face.y, 0.0),
        }
        neutral_sites = tuple(
            NamedPoint3D(site.name, *updated_positions[site.name])
            if site.name in updated_positions
            else site
            for site in self.neutral_pose_sites
        )
        return replace(
            self,
            trailer=trailer,
            neutral_pose_sites=neutral_sites,
        )

    def neutral_site(self, name: str) -> NamedPoint3D:
        for site in self.neutral_pose_sites:
            if site.name == name:
                return site
        raise ConfigurationError(f"unknown neutral semantic site: {name}")

    def manifest_summary(self) -> dict[str, Any]:
        """Return stable provenance fields suitable for experiment manifests."""

        return {
            "specification_id": self.specification_id,
            "schema_version": self.schema_version,
            "source_path": str(self.source_path),
            "source_fingerprint": self.source_fingerprint,
            "effective_fingerprint": self.effective_fingerprint,
            "trailer_variant": self.trailer.variant,
        }

    def _validate_neutral_pose_sites(self) -> None:
        configured = {site.name: site for site in self.neutral_pose_sites}
        required = set(self.mujoco.required_names.sites)
        missing = sorted(required - set(configured))
        if missing:
            raise ConfigurationError(f"neutral_pose_sites missing {missing}")

        state = self.geometry.neutral_state()
        trailer = self.trailer
        expected = {
            "tractor_rear_axle": (
                state.tractor_pose.x,
                state.tractor_pose.y,
                self.wheels.radius,
            ),
            "tractor_front_axle": (
                self.tractor.front_axle(state.tractor_pose).x,
                self.tractor.front_axle(state.tractor_pose).y,
                self.wheels.radius,
            ),
            "hitch": (
                state.hitch_position.x,
                state.hitch_position.y,
                self.tractor.hitch_height,
            ),
            "trailer_axle": (
                state.trailer_pose.x,
                state.trailer_pose.y,
                self.wheels.radius,
            ),
            "trailer_rear_face": (
                trailer.rear_face(state.trailer_pose).x,
                trailer.rear_face(state.trailer_pose).y,
                trailer.body_bottom_height + trailer.body_height / 2.0,
            ),
            "docking_point": (
                trailer.rear_face(state.trailer_pose).x,
                trailer.rear_face(state.trailer_pose).y,
                0.0,
            ),
        }
        for name, expected_position in expected.items():
            actual = configured[name]
            if not all(
                isclose(value, target, rel_tol=0.0, abs_tol=1e-9)
                for value, target in zip(
                    (actual.x, actual.y, actual.z), expected_position, strict=True
                )
            ):
                raise ConfigurationError(
                    f"neutral site {name} disagrees with derived geometry: "
                    f"configured={(actual.x, actual.y, actual.z)}, "
                    f"expected={expected_position}"
                )


@dataclass(frozen=True, slots=True)
class DockConfiguration:
    """Validated loading-bay configuration with provenance."""

    schema_version: int
    units: UnitSystem
    frame_origin: str
    outward_axis: str
    left_axis: str
    up_axis: str
    target_reference: str
    geometry: DockGeometry
    source_path: Path
    source_fingerprint: str

    def __post_init__(self) -> None:
        if self.schema_version != CURRENT_SCHEMA_VERSION:
            raise ConfigurationError(
                f"unsupported dock schema_version={self.schema_version}; "
                f"expected {CURRENT_SCHEMA_VERSION}"
            )
        expected = (
            "rear_wall_center",
            "+x",
            "+y",
            "+z",
            "docking_point",
        )
        actual = (
            self.frame_origin,
            self.outward_axis,
            self.left_axis,
            self.up_axis,
            self.target_reference,
        )
        if actual != expected:
            raise ConfigurationError(
                "dock frame and target reference do not match schema version 1"
            )

    def manifest_summary(self) -> dict[str, Any]:
        return {
            "specification_id": self.geometry.specification_id,
            "schema_version": self.schema_version,
            "source_path": str(self.source_path),
            "source_fingerprint": self.source_fingerprint,
        }


def _load_units(data: Mapping[str, Any], *, vehicle: bool) -> UnitSystem:
    expected = {"length", "angle", "time", "mass"} if vehicle else {"length", "angle"}
    _exact_keys(data, expected, "units")
    return UnitSystem(
        length=_text(data["length"], "units.length"),
        angle=_text(data["angle"], "units.angle"),
        time=_text(data["time"], "units.time") if vehicle else None,
        mass=_text(data["mass"], "units.mass") if vehicle else None,
    )


def _load_tractor(data: Mapping[str, Any]) -> TractorGeometry:
    expected = {
        "wheelbase",
        "front_overhang",
        "rear_overhang",
        "body_width",
        "front_track",
        "rear_track",
        "body_height",
        "body_bottom_height",
        "hitch_offset",
        "collision_footprint_from_rear_axle",
    }
    _exact_keys(data, expected, "tractor")
    hitch = _mapping(data["hitch_offset"], "tractor.hitch_offset")
    _exact_keys(hitch, {"longitudinal", "lateral", "height"}, "tractor.hitch_offset")
    footprint = _mapping(
        data["collision_footprint_from_rear_axle"],
        "tractor.collision_footprint_from_rear_axle",
    )
    _exact_keys(footprint, {"x", "y"}, "tractor.collision_footprint_from_rear_axle")
    try:
        return TractorGeometry(
            wheelbase=_number(data["wheelbase"], "tractor.wheelbase"),
            front_overhang=_number(data["front_overhang"], "tractor.front_overhang"),
            rear_overhang=_number(data["rear_overhang"], "tractor.rear_overhang"),
            body_width=_number(data["body_width"], "tractor.body_width"),
            front_track=_number(data["front_track"], "tractor.front_track"),
            rear_track=_number(data["rear_track"], "tractor.rear_track"),
            body_height=_number(data["body_height"], "tractor.body_height"),
            body_bottom_height=_number(
                data["body_bottom_height"], "tractor.body_bottom_height"
            ),
            hitch_longitudinal=_number(
                hitch["longitudinal"], "tractor.hitch_offset.longitudinal"
            ),
            hitch_lateral=_number(hitch["lateral"], "tractor.hitch_offset.lateral"),
            hitch_height=_number(hitch["height"], "tractor.hitch_offset.height"),
            footprint_x=_bounds(
                footprint["x"], "tractor.collision_footprint_from_rear_axle.x"
            ),
            footprint_y=_bounds(
                footprint["y"], "tractor.collision_footprint_from_rear_axle.y"
            ),
        )
    except ValueError as error:
        raise ConfigurationError(f"invalid tractor geometry: {error}") from error


def _load_trailer(data: Mapping[str, Any]) -> TrailerGeometry:
    expected = {
        "variant",
        "hitch_to_axle",
        "front_overhang",
        "rear_overhang",
        "body_width",
        "axle_track",
        "body_height",
        "body_bottom_height",
        "collision_footprint_from_axle",
    }
    _exact_keys(data, expected, "trailer")
    footprint = _mapping(
        data["collision_footprint_from_axle"],
        "trailer.collision_footprint_from_axle",
    )
    _exact_keys(footprint, {"x", "y"}, "trailer.collision_footprint_from_axle")
    try:
        return TrailerGeometry(
            variant=_text(data["variant"], "trailer.variant"),
            hitch_to_axle=_number(data["hitch_to_axle"], "trailer.hitch_to_axle"),
            front_overhang=_number(data["front_overhang"], "trailer.front_overhang"),
            rear_overhang=_number(data["rear_overhang"], "trailer.rear_overhang"),
            body_width=_number(data["body_width"], "trailer.body_width"),
            axle_track=_number(data["axle_track"], "trailer.axle_track"),
            body_height=_number(data["body_height"], "trailer.body_height"),
            body_bottom_height=_number(
                data["body_bottom_height"], "trailer.body_bottom_height"
            ),
            footprint_x=_bounds(
                footprint["x"], "trailer.collision_footprint_from_axle.x"
            ),
            footprint_y=_bounds(
                footprint["y"], "trailer.collision_footprint_from_axle.y"
            ),
        )
    except ValueError as error:
        raise ConfigurationError(f"invalid trailer geometry: {error}") from error


def _load_variants(
    data: Mapping[str, Any], base: TrailerGeometry
) -> tuple[TrailerGeometry, ...]:
    variants: list[TrailerGeometry] = []
    for name, raw_variant in data.items():
        values = _mapping(raw_variant, f"trailer_variants.{name}")
        _exact_keys(
            values,
            {"hitch_to_axle", "front_overhang", "rear_overhang"},
            f"trailer_variants.{name}",
        )
        hitch_to_axle = _number(
            values["hitch_to_axle"], f"trailer_variants.{name}.hitch_to_axle"
        )
        front_overhang = _number(
            values["front_overhang"], f"trailer_variants.{name}.front_overhang"
        )
        rear_overhang = _number(
            values["rear_overhang"], f"trailer_variants.{name}.rear_overhang"
        )
        try:
            variants.append(
                TrailerGeometry(
                    variant=name,
                    hitch_to_axle=hitch_to_axle,
                    front_overhang=front_overhang,
                    rear_overhang=rear_overhang,
                    body_width=base.body_width,
                    axle_track=base.axle_track,
                    body_height=base.body_height,
                    body_bottom_height=base.body_bottom_height,
                    footprint_x=Bounds1D(
                        -rear_overhang, hitch_to_axle + front_overhang
                    ),
                    footprint_y=Bounds1D(-base.body_width / 2.0, base.body_width / 2.0),
                )
            )
        except ValueError as error:
            raise ConfigurationError(
                f"invalid trailer variant {name}: {error}"
            ) from error
    if not variants:
        raise ConfigurationError("trailer_variants must define at least one variant")
    return tuple(variants)


def _load_mujoco(data: Mapping[str, Any]) -> MujocoParameters:
    _exact_keys(
        data,
        {"gravity", "masses", "contact_friction", "actuator_limits", "required_names"},
        "mujoco",
    )
    masses = _mapping(data["masses"], "mujoco.masses")
    _exact_keys(
        masses,
        {"tractor_chassis", "trailer_body", "wheel_each"},
        "mujoco.masses",
    )
    actuator_limits = _mapping(data["actuator_limits"], "mujoco.actuator_limits")
    _exact_keys(
        actuator_limits,
        {"rear_wheel_torque_each", "steering_torque_each"},
        "mujoco.actuator_limits",
    )
    names = _mapping(data["required_names"], "mujoco.required_names")
    _exact_keys(
        names,
        {"bodies", "joints", "sites", "actuators"},
        "mujoco.required_names",
    )
    return MujocoParameters(
        gravity=_number_sequence(data["gravity"], 3, "mujoco.gravity"),  # type: ignore[arg-type]
        tractor_chassis_mass=_number(
            masses["tractor_chassis"], "mujoco.masses.tractor_chassis"
        ),
        trailer_body_mass=_number(masses["trailer_body"], "mujoco.masses.trailer_body"),
        wheel_mass_each=_number(masses["wheel_each"], "mujoco.masses.wheel_each"),
        contact_friction=_number_sequence(
            data["contact_friction"], 3, "mujoco.contact_friction"
        ),  # type: ignore[arg-type]
        rear_wheel_torque_each=_number(
            actuator_limits["rear_wheel_torque_each"],
            "mujoco.actuator_limits.rear_wheel_torque_each",
        ),
        steering_torque_each=_number(
            actuator_limits["steering_torque_each"],
            "mujoco.actuator_limits.steering_torque_each",
        ),
        required_names=MujocoNames(
            bodies=_string_sequence(names["bodies"], "mujoco.required_names.bodies"),
            joints=_string_sequence(names["joints"], "mujoco.required_names.joints"),
            sites=_string_sequence(names["sites"], "mujoco.required_names.sites"),
            actuators=_string_sequence(
                names["actuators"], "mujoco.required_names.actuators"
            ),
        ),
    )


def load_vehicle_configuration(path: str | Path) -> VehicleConfiguration:
    """Load and strictly validate a schema-version-1 vehicle YAML file."""

    data, resolved = _read_yaml(path)
    expected = {
        "specification_id",
        "schema_version",
        "units",
        "conventions",
        "tractor",
        "trailer",
        "trailer_variants",
        "wheels",
        "neutral_pose_sites",
        "limits",
        "timing",
        "command",
        "mujoco",
    }
    _exact_keys(data, expected, "vehicle")

    conventions = _mapping(data["conventions"], "conventions")
    _exact_keys(
        conventions,
        {
            "body_forward_axis",
            "body_left_axis",
            "world_up_axis",
            "positive_yaw",
            "positive_steering",
            "positive_speed",
            "articulation",
        },
        "conventions",
    )
    wheels = _mapping(data["wheels"], "wheels")
    _exact_keys(wheels, {"radius", "width"}, "wheels")
    limits = _mapping(data["limits"], "limits")
    limit_fields = {
        "steering_angle_abs_max",
        "steering_rate_abs_max",
        "forward_speed_max",
        "reverse_speed_abs_max",
        "longitudinal_acceleration_abs_max",
        "articulation_soft_abs_max",
        "articulation_hard_abs_max",
        "mujoco_hitch_mechanical_abs_max",
    }
    _exact_keys(limits, limit_fields, "limits")
    timing = _mapping(data["timing"], "timing")
    timing_fields = {
        "controller_period",
        "kinematic_integration_step",
        "mujoco_physics_step",
        "nominal_command_delay",
    }
    _exact_keys(timing, timing_fields, "timing")
    command = _mapping(data["command"], "command")
    _exact_keys(command, {"fields", "nominal_target_speed"}, "command")

    tractor = _load_tractor(_mapping(data["tractor"], "tractor"))
    trailer = _load_trailer(_mapping(data["trailer"], "trailer"))
    variants = _load_variants(
        _mapping(data["trailer_variants"], "trailer_variants"), trailer
    )

    neutral_data = _mapping(data["neutral_pose_sites"], "neutral_pose_sites")
    neutral_sites = tuple(
        NamedPoint3D(name, *_number_sequence(value, 3, f"neutral_pose_sites.{name}"))
        for name, value in neutral_data.items()
    )
    if len(neutral_sites) != len({site.name for site in neutral_sites}):
        raise ConfigurationError("neutral_pose_sites contains duplicate names")

    try:
        return VehicleConfiguration(
            specification_id=_text(data["specification_id"], "specification_id"),
            schema_version=_integer(data["schema_version"], "schema_version"),
            units=_load_units(_mapping(data["units"], "units"), vehicle=True),
            conventions=CoordinateConventions(
                **{
                    name: _text(value, f"conventions.{name}")
                    for name, value in conventions.items()
                }
            ),
            tractor=tractor,
            trailer=trailer,
            trailer_variants=variants,
            wheels=WheelGeometry(
                radius=_number(wheels["radius"], "wheels.radius"),
                width=_number(wheels["width"], "wheels.width"),
            ),
            neutral_pose_sites=neutral_sites,
            limits=VehicleLimits(
                **{
                    name: _number(value, f"limits.{name}")
                    for name, value in limits.items()
                }
            ),
            timing=SimulationTiming(
                **{
                    name: _number(value, f"timing.{name}")
                    for name, value in timing.items()
                }
            ),
            command=CommandConfiguration(
                fields=_string_sequence(command["fields"], "command.fields"),
                nominal_target_speed=_number(
                    command["nominal_target_speed"], "command.nominal_target_speed"
                ),
            ),
            mujoco=_load_mujoco(_mapping(data["mujoco"], "mujoco")),
            source_path=resolved,
            source_fingerprint=_normalized_fingerprint(data),
        )
    except (TypeError, ValueError) as error:
        if isinstance(error, ConfigurationError):
            raise
        raise ConfigurationError(f"invalid vehicle configuration: {error}") from error


def load_dock_configuration(path: str | Path) -> DockConfiguration:
    """Load and strictly validate a schema-version-1 loading-dock YAML file."""

    data, resolved = _read_yaml(path)
    _exact_keys(
        data,
        {
            "specification_id",
            "schema_version",
            "units",
            "dock_frame",
            "bay",
            "target",
            "success_tolerances",
        },
        "dock",
    )
    frame = _mapping(data["dock_frame"], "dock_frame")
    _exact_keys(frame, {"origin", "outward_axis", "left_axis", "up_axis"}, "dock_frame")
    bay = _mapping(data["bay"], "bay")
    _exact_keys(
        bay,
        {
            "interior_width",
            "interior_depth",
            "wall_thickness",
            "wall_height",
            "collision_margin",
        },
        "bay",
    )
    target = _mapping(data["target"], "target")
    _exact_keys(target, {"reference", "position", "trailer_yaw"}, "target")
    tolerances = _mapping(data["success_tolerances"], "success_tolerances")
    _exact_keys(
        tolerances,
        {
            "longitudinal",
            "lateral",
            "trailer_yaw",
            "require_no_collision",
            "require_articulation_below_hard_limit",
            "require_complete_trailer_inside_bay",
        },
        "success_tolerances",
    )
    target_x, target_y = _number_sequence(target["position"], 2, "target.position")
    try:
        geometry = DockGeometry(
            specification_id=_text(data["specification_id"], "specification_id"),
            interior_width=_number(bay["interior_width"], "bay.interior_width"),
            interior_depth=_number(bay["interior_depth"], "bay.interior_depth"),
            wall_thickness=_number(bay["wall_thickness"], "bay.wall_thickness"),
            wall_height=_number(bay["wall_height"], "bay.wall_height"),
            collision_margin=_number(bay["collision_margin"], "bay.collision_margin"),
            target_position=Point2D(target_x, target_y),
            target_trailer_yaw=wrap_angle(
                _number(target["trailer_yaw"], "target.trailer_yaw")
            ),
            longitudinal_tolerance=_number(
                tolerances["longitudinal"], "success_tolerances.longitudinal"
            ),
            lateral_tolerance=_number(
                tolerances["lateral"], "success_tolerances.lateral"
            ),
            trailer_yaw_tolerance=_number(
                tolerances["trailer_yaw"], "success_tolerances.trailer_yaw"
            ),
            require_no_collision=_boolean(
                tolerances["require_no_collision"],
                "success_tolerances.require_no_collision",
            ),
            require_articulation_below_hard_limit=_boolean(
                tolerances["require_articulation_below_hard_limit"],
                "success_tolerances.require_articulation_below_hard_limit",
            ),
            require_complete_trailer_inside_bay=_boolean(
                tolerances["require_complete_trailer_inside_bay"],
                "success_tolerances.require_complete_trailer_inside_bay",
            ),
        )
        return DockConfiguration(
            schema_version=_integer(data["schema_version"], "schema_version"),
            units=_load_units(_mapping(data["units"], "units"), vehicle=False),
            frame_origin=_text(frame["origin"], "dock_frame.origin"),
            outward_axis=_text(frame["outward_axis"], "dock_frame.outward_axis"),
            left_axis=_text(frame["left_axis"], "dock_frame.left_axis"),
            up_axis=_text(frame["up_axis"], "dock_frame.up_axis"),
            target_reference=_text(target["reference"], "target.reference"),
            geometry=geometry,
            source_path=resolved,
            source_fingerprint=_normalized_fingerprint(data),
        )
    except (TypeError, ValueError) as error:
        if isinstance(error, ConfigurationError):
            raise
        raise ConfigurationError(f"invalid dock configuration: {error}") from error
