"""Validate shared vehicle/dock configuration and derived geometry.

Run from the repository root with::

    python -m trailer_docking.validation.vehicle_spec
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from math import cos, isclose, sin
from pathlib import Path

from trailer_docking.core.config import (
    DockConfiguration,
    VehicleConfiguration,
    load_dock_configuration,
    load_vehicle_configuration,
)
from trailer_docking.core.geometry import polygon_area
from trailer_docking.core.state import Point2D, Pose2D

DEFAULT_VEHICLE_PATH = Path("configs/vehicles/compact_semitrailer_v1.yaml")
DEFAULT_DOCK_PATH = Path("configs/environments/nominal_dock_v1.yaml")


@dataclass(frozen=True, slots=True)
class ConformanceCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class ConformanceReport:
    vehicle_specification: str
    dock_specification: str
    vehicle_fingerprint: str
    dock_fingerprint: str
    checks: tuple[ConformanceCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def to_dict(self) -> dict[str, object]:
        return {
            "vehicle_specification": self.vehicle_specification,
            "dock_specification": self.dock_specification,
            "vehicle_fingerprint": self.vehicle_fingerprint,
            "dock_fingerprint": self.dock_fingerprint,
            "passed": self.passed,
            "checks": [asdict(check) for check in self.checks],
        }


def _check(name: str, passed: bool, detail: str) -> ConformanceCheck:
    return ConformanceCheck(name=name, passed=passed, detail=detail)


def run_conformance_checks(
    vehicle: VehicleConfiguration, dock: DockConfiguration
) -> ConformanceReport:
    """Run backend-independent invariants after strict loading succeeds."""

    geometry = vehicle.geometry
    neutral = geometry.neutral_state()
    residuals = geometry.constraint_residuals(neutral)
    tractor_area = polygon_area(geometry.tractor.footprint(neutral.tractor_pose))
    trailer_area = polygon_area(geometry.trailer.footprint(neutral.trailer_pose))

    target_yaw = dock.geometry.target_trailer_yaw
    target_rear = dock.geometry.target_position
    trailer_axle = Point2D(
        target_rear.x + geometry.trailer.rear_overhang * cos(target_yaw),
        target_rear.y + geometry.trailer.rear_overhang * sin(target_yaw),
    )
    target_pose = Pose2D(trailer_axle.x, trailer_axle.y, target_yaw)
    target_footprint = geometry.trailer.footprint(target_pose)
    margin = dock.geometry.collision_margin
    trailer_inside = all(
        margin <= point.x <= dock.geometry.interior_depth - margin
        and -dock.geometry.interior_width / 2.0 + margin
        <= point.y
        <= dock.geometry.interior_width / 2.0 - margin
        for point in target_footprint
    )

    checks = (
        _check(
            "rigid_hitch_constraint",
            max(residuals.values()) <= 1e-12,
            f"maximum neutral-pose residual={max(residuals.values()):.3e} m",
        ),
        _check(
            "tractor_footprint_area",
            isclose(
                tractor_area,
                geometry.tractor.body_length * geometry.tractor.body_width,
                abs_tol=1e-12,
            ),
            f"area={tractor_area:.6f} m^2",
        ),
        _check(
            "trailer_footprint_area",
            isclose(
                trailer_area,
                geometry.trailer.body_length * geometry.trailer.body_width,
                abs_tol=1e-12,
            ),
            f"area={trailer_area:.6f} m^2",
        ),
        _check(
            "timing_ratios",
            vehicle.timing.kinematic_substeps_per_control > 0
            and vehicle.timing.mujoco_substeps_per_control > 0,
            "kinematic="
            f"{vehicle.timing.kinematic_substeps_per_control}, "
            f"mujoco={vehicle.timing.mujoco_substeps_per_control} substeps/control",
        ),
        _check(
            "nominal_speed_within_limits",
            -vehicle.limits.reverse_speed_abs_max
            <= vehicle.command.nominal_target_speed
            <= vehicle.limits.forward_speed_max,
            f"nominal target={vehicle.command.nominal_target_speed:.3f} m/s",
        ),
        _check(
            "target_trailer_inside_dock",
            trailer_inside,
            "nominal trailer footprint evaluated with configured collision margin",
        ),
        _check(
            "semantic_site_coverage",
            set(vehicle.mujoco.required_names.sites)
            <= {site.name for site in vehicle.neutral_pose_sites},
            f"configured sites={len(vehicle.neutral_pose_sites)}",
        ),
    )
    return ConformanceReport(
        vehicle_specification=vehicle.specification_id,
        dock_specification=dock.geometry.specification_id,
        vehicle_fingerprint=vehicle.effective_fingerprint,
        dock_fingerprint=dock.source_fingerprint,
        checks=checks,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vehicle", type=Path, default=DEFAULT_VEHICLE_PATH)
    parser.add_argument("--dock", type=Path, default=DEFAULT_DOCK_PATH)
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    vehicle = load_vehicle_configuration(args.vehicle)
    dock = load_dock_configuration(args.dock)
    report = run_conformance_checks(vehicle, dock)
    if args.json_output:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"Vehicle specification: {report.vehicle_specification}")
        print(f"Dock specification: {report.dock_specification}")
        for check in report.checks:
            status = "PASS" if check.passed else "FAIL"
            print(f"[{status}] {check.name}: {check.detail}")
        print(f"Vehicle fingerprint: {report.vehicle_fingerprint}")
        print(f"Dock fingerprint: {report.dock_fingerprint}")
        print("Overall: PASS" if report.passed else "Overall: FAIL")
    return 0 if report.passed else 1


if __name__ == "__main__":  # pragma: no cover - exercised through CLI tests
    raise SystemExit(main())
