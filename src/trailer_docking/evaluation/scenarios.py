"""Editable scenarios: frozen base specifications plus explicit overrides.

A :class:`Scenario` is the parameter model edited by the UI and consumed by
training and docking tests.  It never modifies the frozen YAML files.  It
names a base vehicle, dock, and Gymnasium task specification and records the
handful of values that differ, so every run can be traced to the frozen
baseline plus a small, serializable set of changes.

Parameters fall in two groups:

* *shared* parameters (dock size, truck dimensions, steering and speed
  limits, start-pose ranges, static obstacles) change the planar problem and
  therefore apply to every simulator backend;
* *physics-only* parameters (masses, ground friction) exist only in the
  MuJoCo backend and have no counterpart in the 2D kinematic model.  They
  take effect only when a run uses the 3D backend.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field, fields, replace
from hashlib import sha256
from math import cos, isfinite, pi, sin
from numbers import Real
from pathlib import Path
from typing import Any

import yaml

from trailer_docking.core.config import (
    ConfigurationError,
    DockConfiguration,
    DockingTaskConfiguration,
    NamedPoint3D,
    VehicleConfiguration,
    load_dock_configuration,
    load_docking_task_configuration,
    load_vehicle_configuration,
)
from trailer_docking.core.geometry import Bounds1D, VehicleGeometry
from trailer_docking.core.obstacles import Obstacle
from trailer_docking.core.state import Point2D, Pose2D, VehicleState, wrap_angle
from trailer_docking.paths.planners import DEFAULT_PLANNER, validate_planner

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VEHICLE_PATH = Path("configs/vehicles/compact_semitrailer_v1.yaml")
DEFAULT_DOCK_PATH = Path("configs/environments/nominal_dock_v1.yaml")
DEFAULT_TASK_PATH = Path("configs/environments/gymnasium_docking_v1.yaml")

# Names of override fields that only the MuJoCo backend can represent.
PHYSICS_ONLY_FIELDS = frozenset(
    {"tractor_chassis_mass", "trailer_body_mass", "ground_friction"}
)
# Name given to the trailer when any trailer dimension is overridden.
CUSTOM_TRAILER_VARIANT = "custom"
_TRACTOR_DIMENSIONS = ("tractor_wheelbase", "tractor_body_width")
_TRAILER_DIMENSIONS = (
    "trailer_hitch_to_axle",
    "trailer_rear_overhang",
    "trailer_body_width",
)
_LIMIT_FIELDS = ("steering_angle_max", "steering_rate_max", "reverse_speed")


def resolve_config_path(path: Path) -> Path:
    """Resolve a specification path from the cwd or the repository root.

    Relative paths are tried against the current directory first and then
    against the repository checkout, so the default scenario works from any
    working directory in a source checkout.
    """

    if path.is_absolute() or path.exists():
        return path
    candidate = REPOSITORY_ROOT / path
    return candidate if candidate.exists() else path


@dataclass(frozen=True, slots=True)
class StartDistribution:
    """Ranges from which an episode's start pose is sampled.

    The docking point starts ``standoff`` metres outside the bay opening and
    ``lateral`` metres off the bay centreline, with the trailer heading
    rotated ``trailer_yaw`` from the target heading and the tractor
    articulated by ``articulation``.  Each range is ``(minimum, maximum)``.
    """

    standoff: tuple[float, float] = (6.0, 9.0)
    lateral: tuple[float, float] = (-0.4, 0.4)
    trailer_yaw: tuple[float, float] = (-0.1, 0.1)
    articulation: tuple[float, float] = (-0.05, 0.05)

    def __post_init__(self) -> None:
        for item in fields(self):
            raw = getattr(self, item.name)
            if isinstance(raw, str):
                raise ConfigurationError(
                    f"start range {item.name} must be finite (minimum, maximum)"
                )
            value = tuple(float(v) for v in raw)
            if (
                len(value) != 2
                or not all(isfinite(bound) for bound in value)
                or not value[0] <= value[1]
            ):
                raise ConfigurationError(
                    f"start range {item.name} must be finite (minimum, maximum)"
                )
            object.__setattr__(self, item.name, value)
        if self.standoff[0] < 0.0:
            raise ConfigurationError("start standoff cannot be negative")


@dataclass(frozen=True, slots=True)
class DomainRandomization:
    """Per-episode variation beyond the start pose; everything is off by default.

    ``trailer_hitch_to_axle`` is a ``(minimum, maximum)`` range in metres from
    which each episode's trailer length is drawn.  ``observation_noise_std``
    is the standard deviation of zero-mean Gaussian noise added to every
    element of the normalized observation, which is then clipped to its
    bounds.  The noise is in normalized observation units; it is not a model
    of any particular sensor.
    """

    trailer_hitch_to_axle: tuple[float, float] | None = None
    observation_noise_std: float = 0.0

    def __post_init__(self) -> None:
        bounds = self.trailer_hitch_to_axle
        if bounds is not None:
            try:
                value = (
                    () if isinstance(bounds, str) else tuple(float(v) for v in bounds)
                )
            except (TypeError, ValueError):
                value = ()
            if (
                len(value) != 2
                or not all(isfinite(bound) for bound in value)
                or not 0.0 < value[0] <= value[1]
            ):
                raise ConfigurationError(
                    "trailer_hitch_to_axle range must be positive (minimum, maximum)"
                )
            object.__setattr__(self, "trailer_hitch_to_axle", value)
        deviation = self.observation_noise_std
        if (
            isinstance(deviation, bool)
            or not isinstance(deviation, Real)
            or not isfinite(deviation)
            or deviation < 0.0
        ):
            raise ConfigurationError(
                "observation_noise_std must be finite and not negative"
            )
        object.__setattr__(self, "observation_noise_std", float(deviation))

    def to_dict(self) -> dict[str, Any]:
        bounds = self.trailer_hitch_to_axle
        return {
            "trailer_hitch_to_axle": None if bounds is None else list(bounds),
            "observation_noise_std": self.observation_noise_std,
        }


@dataclass(frozen=True, slots=True)
class ScenarioOverrides:
    """Values that replace the base specification; ``None`` keeps the base."""

    trailer_variant: str | None = None
    dock_interior_width: float | None = None
    dock_interior_depth: float | None = None
    tractor_wheelbase: float | None = None
    tractor_body_width: float | None = None
    trailer_hitch_to_axle: float | None = None
    trailer_rear_overhang: float | None = None
    trailer_body_width: float | None = None
    steering_angle_max: float | None = None
    steering_rate_max: float | None = None
    reverse_speed: float | None = None
    path_lateral_weight: float | None = None
    path_heading_weight: float | None = None
    tractor_chassis_mass: float | None = None
    trailer_body_mass: float | None = None
    ground_friction: float | None = None

    def __post_init__(self) -> None:
        variant = self.trailer_variant
        if variant is not None and (not isinstance(variant, str) or not variant):
            raise ConfigurationError("trailer_variant must be a non-empty string")
        for item in fields(self):
            if item.name == "trailer_variant":
                continue
            value = getattr(self, item.name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ConfigurationError(f"{item.name} must be a number")
            number = float(value)
            if not isfinite(number) or number <= 0.0:
                raise ConfigurationError(f"{item.name} must be finite and positive")
            object.__setattr__(self, item.name, number)


@dataclass(frozen=True, slots=True)
class BuiltScenario:
    """Validated configuration objects produced from a scenario."""

    vehicle: VehicleConfiguration
    dock: DockConfiguration
    task: DockingTaskConfiguration
    obstacles: tuple[Obstacle, ...] = ()


@dataclass(frozen=True, slots=True)
class Scenario:
    """A named, serializable docking problem."""

    name: str = "nominal"
    vehicle_path: Path = DEFAULT_VEHICLE_PATH
    dock_path: Path = DEFAULT_DOCK_PATH
    task_path: Path = DEFAULT_TASK_PATH
    overrides: ScenarioOverrides = field(default_factory=ScenarioOverrides)
    start: StartDistribution = field(default_factory=StartDistribution)
    randomization: DomainRandomization = field(default_factory=DomainRandomization)
    planner: str = DEFAULT_PLANNER
    obstacles: tuple[Obstacle, ...] = ()

    def __post_init__(self) -> None:
        obstacles = tuple(self.obstacles)
        if not all(isinstance(item, Obstacle) for item in obstacles):
            raise ConfigurationError("obstacles must be Obstacle objects")
        object.__setattr__(self, "obstacles", obstacles)
        if not isinstance(self.name, str) or not self.name.strip():
            raise ConfigurationError("scenario name must be a non-empty string")
        object.__setattr__(self, "vehicle_path", Path(self.vehicle_path))
        object.__setattr__(self, "dock_path", Path(self.dock_path))
        object.__setattr__(self, "task_path", Path(self.task_path))

    def build(self) -> BuiltScenario:
        """Load the base specifications and apply the overrides.

        Raises :class:`ConfigurationError` (or a geometry validation error,
        which is also a ``ValueError``) if an override is invalid.
        """

        vehicle = load_vehicle_configuration(resolve_config_path(self.vehicle_path))
        dock = load_dock_configuration(resolve_config_path(self.dock_path))
        task = load_docking_task_configuration(resolve_config_path(self.task_path))
        changes = self.overrides
        try:
            validate_planner(self.planner)
        except ValueError as error:
            raise ConfigurationError(str(error)) from error
        # Reward weights for staying on the planned route: the task penalizes
        # the squared lateral and heading deviation from the reference path.
        weights: dict[str, Any] = {}
        if changes.path_lateral_weight is not None:
            weights["reward_lateral_error"] = changes.path_lateral_weight
        if changes.path_heading_weight is not None:
            weights["reward_heading_error"] = changes.path_heading_weight
        if weights:
            task = replace(task, **weights)
        if changes.trailer_variant is not None:
            vehicle = vehicle.with_trailer_variant(changes.trailer_variant)
        vehicle = _with_overrides(vehicle, changes)
        physics: dict[str, Any] = {}
        if changes.tractor_chassis_mass is not None:
            physics["tractor_chassis_mass"] = float(changes.tractor_chassis_mass)
        if changes.trailer_body_mass is not None:
            physics["trailer_body_mass"] = float(changes.trailer_body_mass)
        if changes.ground_friction is not None:
            _, torsional, rolling = vehicle.mujoco.contact_friction
            physics["contact_friction"] = (
                float(changes.ground_friction),
                torsional,
                rolling,
            )
        if physics:
            vehicle = replace(vehicle, mujoco=replace(vehicle.mujoco, **physics))
        bay: dict[str, Any] = {}
        if changes.dock_interior_width is not None:
            bay["interior_width"] = float(changes.dock_interior_width)
        if changes.dock_interior_depth is not None:
            bay["interior_depth"] = float(changes.dock_interior_depth)
        if bay:
            dock = replace(dock, geometry=replace(dock.geometry, **bay))
        # An episode must not begin already jackknifed.
        hard_limit = vehicle.limits.articulation_hard_abs_max
        if max(abs(bound) for bound in self.start.articulation) >= hard_limit:
            raise ConfigurationError(
                "start articulation range must stay inside the"
                f" +-{hard_limit:.3f} rad jackknife threshold"
            )
        if max(abs(bound) for bound in self.start.trailer_yaw) > pi:
            raise ConfigurationError("start trailer_yaw range must be within +-pi")
        built = BuiltScenario(
            vehicle=vehicle, dock=dock, task=task, obstacles=self.obstacles
        )
        _check_target_clear(built)
        _check_trailer_fits(vehicle, dock)
        lengths = self.randomization.trailer_hitch_to_axle
        if lengths is not None:
            if changes.trailer_hitch_to_axle is not None:
                raise ConfigurationError(
                    "set either a fixed trailer hitch-to-axle length or a"
                    " randomized range, not both"
                )
            # The longest and shortest trailer an episode can draw must both
            # be valid vehicles that fit the bay; lengths between them then do.
            for length in lengths:
                _check_trailer_fits(
                    _with_overrides(
                        vehicle, replace(changes, trailer_hitch_to_axle=length)
                    ),
                    dock,
                )
        _check_starts(self, built)
        return built

    # ------------------------------------------------------- serialization

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "vehicle_path": self.vehicle_path.as_posix(),
            "dock_path": self.dock_path.as_posix(),
            "task_path": self.task_path.as_posix(),
            "overrides": asdict(self.overrides),
            "start": {key: list(value) for key, value in asdict(self.start).items()},
            "randomization": self.randomization.to_dict(),
            "planner": self.planner,
            "obstacles": [item.to_dict() for item in self.obstacles],
        }

    @classmethod
    def from_dict(cls, data: Any) -> Scenario:
        if not isinstance(data, dict):
            raise ConfigurationError("scenario must be a mapping")
        allowed = {
            "name",
            "vehicle_path",
            "dock_path",
            "task_path",
            "overrides",
            "start",
            "randomization",
            "planner",
            "obstacles",
        }
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ConfigurationError(f"scenario has unknown fields {unknown}")
        try:
            return cls(
                name=data.get("name", "nominal"),
                vehicle_path=Path(data.get("vehicle_path", DEFAULT_VEHICLE_PATH)),
                dock_path=Path(data.get("dock_path", DEFAULT_DOCK_PATH)),
                task_path=Path(data.get("task_path", DEFAULT_TASK_PATH)),
                overrides=ScenarioOverrides(**(data.get("overrides") or {})),
                start=StartDistribution(**(data.get("start") or {})),
                randomization=DomainRandomization(**(data.get("randomization") or {})),
                planner=data.get("planner", DEFAULT_PLANNER),
                obstacles=tuple(
                    Obstacle.from_dict(item) for item in data.get("obstacles") or ()
                ),
            )
        except ConfigurationError:
            raise
        except (TypeError, ValueError) as error:
            raise ConfigurationError(f"invalid scenario: {error}") from error

    def save(self, path: str | Path) -> Path:
        """Write the scenario as YAML and return the path."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8"
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> Scenario:
        source = Path(path)
        try:
            data = yaml.safe_load(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
            raise ConfigurationError(
                f"cannot read scenario {source}: {error}"
            ) from error
        return cls.from_dict(data)

    @property
    def fingerprint(self) -> str:
        """Stable hash of the scenario's content for experiment manifests.

        It covers the contents of the base specifications (through their own
        fingerprints, including the task reward and observation settings),
        the overrides, the start distribution, and the randomization, but not the
        file paths, so the same scenario hashes identically from any working
        directory.
        """

        vehicle = load_vehicle_configuration(resolve_config_path(self.vehicle_path))
        dock = load_dock_configuration(resolve_config_path(self.dock_path))
        content = self.to_dict()
        content["vehicle_path"] = vehicle.source_fingerprint
        content["dock_path"] = dock.source_fingerprint
        content["task_path"] = load_docking_task_configuration(
            resolve_config_path(self.task_path)
        ).source_fingerprint
        serialized = json.dumps(content, sort_keys=True, separators=(",", ":"))
        return sha256(serialized.encode("utf-8")).hexdigest()


def _scaled_track(track: float, old_width: float, new_width: float) -> float:
    """Keep the wheels the same distance inside the body sides."""

    return track + (new_width - old_width)


def _with_overrides(
    vehicle: VehicleConfiguration, changes: ScenarioOverrides
) -> VehicleConfiguration:
    """Apply dimension and limit overrides in one validated replacement.

    Footprints, wheel tracks, and the neutral-pose semantic sites are derived
    again from the new dimensions, and the result passes the same validation
    as a configuration loaded from YAML.  A trailer with any overridden
    dimension becomes the ``custom`` variant.  ``reverse_speed`` sets both the
    reverse-speed limit and the fixed reverse target that the
    ``fixed_reverse`` task commands.  Dimensions and limits are applied
    together because their validity depends on each other (the steering limit
    must be reachable with the wheelbase and track).
    """

    tractor_changed = any(getattr(changes, n) is not None for n in _TRACTOR_DIMENSIONS)
    trailer_changed = any(getattr(changes, n) is not None for n in _TRAILER_DIMENSIONS)
    limits_changed = any(getattr(changes, n) is not None for n in _LIMIT_FIELDS)
    if not (tractor_changed or trailer_changed or limits_changed):
        return vehicle
    updates: dict[str, Any] = {}

    tractor = vehicle.tractor
    if tractor_changed:
        wheelbase = changes.tractor_wheelbase or tractor.wheelbase
        width = changes.tractor_body_width or tractor.body_width
        tractor = replace(
            tractor,
            wheelbase=wheelbase,
            body_width=width,
            front_track=_scaled_track(tractor.front_track, tractor.body_width, width),
            rear_track=_scaled_track(tractor.rear_track, tractor.body_width, width),
            footprint_x=Bounds1D(
                -tractor.rear_overhang, wheelbase + tractor.front_overhang
            ),
            footprint_y=Bounds1D(-width / 2.0, width / 2.0),
        )
        updates["tractor"] = tractor

    trailer = vehicle.trailer
    if trailer_changed:
        hitch_to_axle = changes.trailer_hitch_to_axle or trailer.hitch_to_axle
        rear_overhang = changes.trailer_rear_overhang or trailer.rear_overhang
        width = changes.trailer_body_width or trailer.body_width
        trailer = replace(
            trailer,
            variant=CUSTOM_TRAILER_VARIANT,
            hitch_to_axle=hitch_to_axle,
            rear_overhang=rear_overhang,
            body_width=width,
            axle_track=_scaled_track(trailer.axle_track, trailer.body_width, width),
            footprint_x=Bounds1D(
                -rear_overhang, hitch_to_axle + trailer.front_overhang
            ),
            footprint_y=Bounds1D(-width / 2.0, width / 2.0),
        )
        updates["trailer"] = trailer
        updates["trailer_variants"] = (
            *(v for v in vehicle.trailer_variants if v.variant != trailer.variant),
            trailer,
        )

    if tractor_changed or trailer_changed:
        geometry = VehicleGeometry(tractor, trailer, vehicle.wheels)
        state = geometry.neutral_state()
        front_axle = tractor.front_axle(state.tractor_pose)
        rear_face = trailer.rear_face(state.trailer_pose)
        radius = vehicle.wheels.radius
        positions = {
            "tractor_rear_axle": (state.tractor_pose.x, state.tractor_pose.y, radius),
            "tractor_front_axle": (front_axle.x, front_axle.y, radius),
            "hitch": (
                state.hitch_position.x,
                state.hitch_position.y,
                tractor.hitch_height,
            ),
            "trailer_axle": (state.trailer_pose.x, state.trailer_pose.y, radius),
            "trailer_rear_face": (
                rear_face.x,
                rear_face.y,
                trailer.body_bottom_height + trailer.body_height / 2.0,
            ),
            "docking_point": (rear_face.x, rear_face.y, 0.0),
        }
        updates["neutral_pose_sites"] = tuple(
            NamedPoint3D(site.name, *positions[site.name])
            if site.name in positions
            else site
            for site in vehicle.neutral_pose_sites
        )

    if limits_changed:
        limits = vehicle.limits
        if changes.steering_angle_max is not None:
            limits = replace(limits, steering_angle_abs_max=changes.steering_angle_max)
        if changes.steering_rate_max is not None:
            limits = replace(limits, steering_rate_abs_max=changes.steering_rate_max)
        if changes.reverse_speed is not None:
            limits = replace(limits, reverse_speed_abs_max=changes.reverse_speed)
            updates["command"] = replace(
                vehicle.command, nominal_target_speed=-changes.reverse_speed
            )
        updates["limits"] = limits
    return replace(vehicle, **updates)


def _aligned_state(
    geometry: VehicleGeometry,
    docking_point: Point2D,
    trailer_yaw: float,
    articulation: float = 0.0,
) -> VehicleState:
    tractor_pose, yaw = assembled_start(
        geometry, docking_point, trailer_yaw, articulation
    )
    return geometry.assembled_state(tractor_pose, yaw)


def _check_trailer_fits(vehicle: VehicleConfiguration, dock: DockConfiguration) -> None:
    """Reject a trailer that can never satisfy the docking-success rule."""

    from trailer_docking.safety.collision import (
        trailer_inside_bay,
        vehicle_dock_collision,
    )

    bay = dock.geometry
    if not bay.require_complete_trailer_inside_bay:
        return
    geometry = vehicle.geometry
    # The best case is the trailer as deep as the tolerance allows.
    for offset in (0.0, -bay.longitudinal_tolerance, bay.longitudinal_tolerance):
        state = _aligned_state(
            geometry,
            Point2D(bay.target_position.x + offset, bay.target_position.y),
            bay.target_trailer_yaw,
        )
        if (
            trailer_inside_bay(state, geometry, bay)
            and not vehicle_dock_collision(state, geometry, bay).collided
        ):
            return
    trailer = vehicle.trailer
    raise ConfigurationError(
        f"a trailer {trailer.body_length:.2f} m long and"
        f" {trailer.body_width:.2f} m wide cannot dock completely inside a bay"
        f" {bay.interior_depth:.2f} m deep and {bay.interior_width:.2f} m wide;"
        " enlarge the bay or shorten the trailer"
    )


def _check_target_clear(built: BuiltScenario) -> None:
    """Reject an obstacle that the docked vehicle would be standing in."""

    from trailer_docking.safety.collision import vehicle_obstacle_contacts

    if not built.obstacles:
        return
    bay = built.dock.geometry
    geometry = built.vehicle.geometry
    state = _aligned_state(geometry, bay.target_position, bay.target_trailer_yaw)
    contacts = vehicle_obstacle_contacts(state, geometry, built.obstacles)
    if contacts:
        raise ConfigurationError(
            "the docked vehicle would overlap an obstacle"
            f" ({', '.join(contacts)}); move or remove it"
        )


def _check_starts(scenario: Scenario, built: BuiltScenario) -> None:
    """Reject start ranges whose extremes begin an episode already failed."""

    # Imported here: the task module pulls in the Gymnasium environment.
    from trailer_docking.environments.task import evaluate_task, workspace_bounds

    start = scenario.start
    bay = built.dock.geometry
    yaw = bay.target_trailer_yaw
    geometry = built.vehicle.geometry
    for standoff in set(start.standoff):
        for lateral in set(start.lateral):
            for offset in set(start.trailer_yaw):
                for articulation in set(start.articulation):
                    along = bay.interior_depth + standoff
                    point = Point2D(
                        along * cos(yaw) - lateral * sin(yaw),
                        along * sin(yaw) + lateral * cos(yaw),
                    )
                    state = _aligned_state(geometry, point, yaw + offset, articulation)
                    x_bounds, y_bounds = workspace_bounds(
                        state.tractor_pose, bay, built.task, built.vehicle
                    )
                    status = evaluate_task(
                        state, built.vehicle, bay, x_bounds, y_bounds, built.obstacles
                    )
                    if status.terminated and not status.success:
                        raise ConfigurationError(
                            "an episode starting at standoff"
                            f" {standoff:g} m, lateral {lateral:g} m, trailer yaw"
                            f" {offset:g} rad, articulation {articulation:g} rad"
                            f" would begin already failed ({status.reason})"
                        )


def assembled_start(
    geometry: VehicleGeometry,
    docking_point: Point2D,
    trailer_yaw: float,
    articulation: float,
) -> tuple[Pose2D, float]:
    """Return ``(tractor_pose, trailer_yaw)`` placing the docking point.

    The result can be passed directly to ``Simulator.reset`` or
    ``VehicleGeometry.assembled_state``.
    """

    trailer_yaw = wrap_angle(trailer_yaw)
    tractor_yaw = wrap_angle(trailer_yaw + articulation)
    trailer = geometry.trailer
    tractor = geometry.tractor
    reach = trailer.rear_overhang + trailer.hitch_to_axle
    hitch_x = docking_point.x + reach * cos(trailer_yaw)
    hitch_y = docking_point.y + reach * sin(trailer_yaw)
    offset_x = tractor.hitch_longitudinal * cos(
        tractor_yaw
    ) - tractor.hitch_lateral * sin(tractor_yaw)
    offset_y = tractor.hitch_longitudinal * sin(
        tractor_yaw
    ) + tractor.hitch_lateral * cos(tractor_yaw)
    return Pose2D(hitch_x - offset_x, hitch_y - offset_y, tractor_yaw), trailer_yaw


def sample_start(
    scenario: Scenario, built: BuiltScenario, rng: random.Random
) -> tuple[Pose2D, float]:
    """Sample one start pose from the scenario's start distribution."""

    start = scenario.start
    bay = built.dock.geometry
    target_yaw = bay.target_trailer_yaw
    standoff = rng.uniform(*start.standoff)
    lateral = rng.uniform(*start.lateral)
    # The opening lies one bay depth out from the rear wall along the target
    # heading; offsets are expressed in that same dock-aligned frame.
    along = bay.interior_depth + standoff
    docking_point = Point2D(
        along * cos(target_yaw) - lateral * sin(target_yaw),
        along * sin(target_yaw) + lateral * cos(target_yaw),
    )
    return assembled_start(
        built.vehicle.geometry,
        docking_point,
        target_yaw + rng.uniform(*start.trailer_yaw),
        rng.uniform(*start.articulation),
    )
