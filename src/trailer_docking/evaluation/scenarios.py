"""Editable scenarios: frozen base specifications plus explicit overrides.

A :class:`Scenario` is the parameter model edited by the UI and consumed by
training and docking tests.  It never modifies the frozen YAML files.  It
names a base vehicle and dock specification and records the handful of values
that differ, so every run can be traced to the frozen baseline plus a small,
serializable set of changes.

Parameters fall in two groups:

* *shared* parameters (dock size, trailer variant, start-pose ranges) change
  the planar problem and therefore apply to every simulator backend;
* *physics-only* parameters (masses, ground friction) exist only in the
  MuJoCo backend and have no counterpart in the 2D kinematic model.
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
    VehicleConfiguration,
    load_dock_configuration,
    load_vehicle_configuration,
)
from trailer_docking.core.geometry import VehicleGeometry
from trailer_docking.core.state import Point2D, Pose2D, wrap_angle

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VEHICLE_PATH = Path("configs/vehicles/compact_semitrailer_v1.yaml")
DEFAULT_DOCK_PATH = Path("configs/environments/nominal_dock_v1.yaml")

# Names of override fields that only the MuJoCo backend can represent.
PHYSICS_ONLY_FIELDS = frozenset(
    {"tractor_chassis_mass", "trailer_body_mass", "ground_friction"}
)


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
            value = tuple(float(v) for v in getattr(self, item.name))
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
class ScenarioOverrides:
    """Values that replace the base specification; ``None`` keeps the base."""

    trailer_variant: str | None = None
    dock_interior_width: float | None = None
    dock_interior_depth: float | None = None
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


@dataclass(frozen=True, slots=True)
class Scenario:
    """A named, serializable docking problem."""

    name: str = "nominal"
    vehicle_path: Path = DEFAULT_VEHICLE_PATH
    dock_path: Path = DEFAULT_DOCK_PATH
    overrides: ScenarioOverrides = field(default_factory=ScenarioOverrides)
    start: StartDistribution = field(default_factory=StartDistribution)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ConfigurationError("scenario name cannot be empty")
        object.__setattr__(self, "vehicle_path", Path(self.vehicle_path))
        object.__setattr__(self, "dock_path", Path(self.dock_path))

    def build(self) -> BuiltScenario:
        """Load the base specifications and apply the overrides.

        Raises :class:`ConfigurationError` (or a geometry validation error,
        which is also a ``ValueError``) if an override is invalid.
        """

        vehicle = load_vehicle_configuration(resolve_config_path(self.vehicle_path))
        dock = load_dock_configuration(resolve_config_path(self.dock_path))
        changes = self.overrides
        if changes.trailer_variant is not None:
            vehicle = vehicle.with_trailer_variant(changes.trailer_variant)
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
        return BuiltScenario(vehicle=vehicle, dock=dock)

    # ------------------------------------------------------- serialization

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "vehicle_path": self.vehicle_path.as_posix(),
            "dock_path": self.dock_path.as_posix(),
            "overrides": asdict(self.overrides),
            "start": {key: list(value) for key, value in asdict(self.start).items()},
        }

    @classmethod
    def from_dict(cls, data: Any) -> Scenario:
        if not isinstance(data, dict):
            raise ConfigurationError("scenario must be a mapping")
        allowed = {"name", "vehicle_path", "dock_path", "overrides", "start"}
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ConfigurationError(f"scenario has unknown fields {unknown}")
        try:
            return cls(
                name=str(data.get("name", "nominal")),
                vehicle_path=Path(data.get("vehicle_path", DEFAULT_VEHICLE_PATH)),
                dock_path=Path(data.get("dock_path", DEFAULT_DOCK_PATH)),
                overrides=ScenarioOverrides(**(data.get("overrides") or {})),
                start=StartDistribution(**(data.get("start") or {})),
            )
        except TypeError as error:
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
        fingerprints), the overrides, and the start distribution, but not the
        file paths, so the same scenario hashes identically from any working
        directory.
        """

        vehicle = load_vehicle_configuration(resolve_config_path(self.vehicle_path))
        dock = load_dock_configuration(resolve_config_path(self.dock_path))
        content = self.to_dict()
        content["vehicle_path"] = vehicle.source_fingerprint
        content["dock_path"] = dock.source_fingerprint
        serialized = json.dumps(content, sort_keys=True, separators=(",", ":"))
        return sha256(serialized.encode("utf-8")).hexdigest()


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
