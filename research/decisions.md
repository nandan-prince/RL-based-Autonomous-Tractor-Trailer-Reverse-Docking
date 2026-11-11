# Research and engineering decisions

This log records decisions that affect reproducibility, interpretation, or the
eventual report. It complements the frozen design documents; it does not replace
them.

## D-001 — Versioned compact vehicle baseline

**Date:** 2026-11-11  
**Status:** accepted and implemented at the configuration/geometry layer

The first cross-simulator vehicle is `compact-semitrailer-v1`, based on the
NMPC paper's compact experimental scale. The nominal hitch is on the tractor
rear axle, while a signed hitch offset remains configurable to represent the
trajectory-state paper's more general geometry. Short, nominal, and long
trailer variants change hitch-to-axle distance through configuration rather
than separate code paths.

This is a development and literature-aligned baseline, not a permanent claim
that a compact vehicle is representative of a full-size highway combination.
A highway-scale vehicle can be added under the same schema with a new
specification ID.

## D-002 — Initial control authority

**Date:** 2026-11-11  
**Status:** accepted; plant propagation not yet implemented

The primary PPO-versus-NMPC comparison uses bounded steering rate as the only
controller-selected action and a fixed reverse-speed target of `-0.2 m/s`.
Steering angle remains vehicle state. The shared command retains `target_speed`
so scripted tests and later experiments can vary it without changing the
interface. Giving either controller velocity authority will be a separately
identified experiment.

The `0.3 rad/s` steering-rate limit is a project engineering choice, not a
reported parameter from either core paper, and should later receive a
sensitivity check.

## D-003 — Configuration as executable source of truth

**Date:** 2026-11-11  
**Status:** implemented

Vehicle and dock dimensions, limits, timing, semantic sites, and MuJoCo names
are loaded from strict versioned YAML. Unknown, missing, non-finite, or
inconsistent values fail explicitly. `schema_version` identifies the software
contract; `specification_id` identifies a parameter set. Normalized SHA-256
fingerprints will connect future experiments to exact inputs.

## D-004 — Primitive state with derived articulation

**Date:** 2026-11-11  
**Status:** implemented

The external state stores tractor pose, trailer pose, hitch position, steering,
speed, yaw rates, and time. Articulation angle and articulation rate are derived
from those primitives and cannot be independently written. Serialized derived
values are checked on load so contradictory state is rejected.

The state may contain redundant geometric measurements because MuJoCo will
observe multiple semantic sites. Geometry residuals quantify disagreement
instead of silently repairing it. The 2D model will construct states that
satisfy the rigid constraint exactly.

## D-005 — Living IEEE-format report

**Date:** 2026-11-11  
**Status:** implemented as a manuscript scaffold

The repository report uses generic IEEE conference formatting through
`IEEEtran`, without claiming a target venue. The report is maintained alongside
development and separates implemented, planned, and open work. The immediate
deliverable remains the course report; adaptation to a particular publication
template requires an explicit later decision.

## D-006 — MuJoCo model API and contributor handoff

**Date:** 2026-11-11  
**Status:** naming and documentation contract implemented; model planned

The version-1 MuJoCo interface freezes names for the free joint, hitch,
steering links and joints, every wheel body and spin joint, six semantic sites,
and four steering/drive actuators. The complete contract lives in the vehicle
YAML, strict configuration validation, frozen design specification, and the
MuJoCo contributor guide. This avoids coupling the future adapter to MJCF
element order or to undocumented names selected independently by the 3D team.

Collision geometry will use primitives that match the planar footprints;
decorative meshes remain visual-only. The vehicle XML, dock scene, adapter, and
cross-simulator tests are still planned and must not be reported as implemented.
