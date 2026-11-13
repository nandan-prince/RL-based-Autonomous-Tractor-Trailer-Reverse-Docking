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
**Status:** accepted; enforced by both plants and the primary Gymnasium action

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
**Status:** superseded by implemented model decision D-007

The version-1 MuJoCo interface freezes names for the free joint, hitch,
steering links and joints, every wheel body and spin joint, six semantic sites,
and four steering/drive actuators. The complete contract lives in the vehicle
YAML, strict configuration validation, frozen design specification, and the
MuJoCo contributor guide. This avoids coupling the adapter to MJCF
element order or to undocumented names selected independently by the 3D team.

Collision geometry uses primitives that match the planar footprints and
decorative meshes remain visual-only. D-007 records the subsequent vehicle,
dock, and adapter implementation; D-009 records the later direct comparison
with the project 2D simulator.

## D-007 — First MuJoCo vehicle, dock, and adapter

**Date:** 2026-11-11
**Status:** implemented; the later direct 2D comparison is recorded in D-009

The MJCF is generated from the vehicle and dock YAML by
`simulators/mujoco/builder.py`; the committed XML files are generator output
guarded by a drift test. `MujocoSimulator` exposes the shared command and state
contract at the `10 Hz` controller period with 50 physics steps per command.

Modeling choices made while validating the model, with the observation that
motivated each:

- *Collision masks.* Wheels collide only with the ground; chassis boxes only
  with dock walls. The front wheels lie inside the tractor collision box and
  are grandchildren of the tractor body, so MuJoCo's parent-child filter does
  not exclude that contact.
- *Planar wall collision.* Dock wall collision volumes extend above every
  chassis box regardless of the configured wall height, which only sets the
  visible wall. A wall lower than the trailer floor (`0.75 m`) otherwise let
  the trailer reverse `4 m` through the back wall with no contact reported.
- *Thin wheel contact geometry.* With full-width `0.12 m` cylinders the two rim
  contacts scrub whenever a wheel yaws. Steering then needed the full
  `200 N m` actuator limit and reached only about `0.04 rad/s` of the
  commanded `0.30 rad/s`. Collision now uses a `0.02 m` cylinder on the wheel
  centre plane while the full-width cylinder keeps the mass and inertia.
  Steering tracks the ramp within `0.003 rad` and holds within `0.0013 rad`.
- *Steering servo.* `kp = 20000 N m/rad`, joint damping `120 N m s/rad`,
  armature `1.0 kg m^2`, with mechanical stops at the Ackermann angles of the
  virtual limit. Measured overshoot past `0.50 rad` is at most `0.0012 rad`.
- *Drive.* Velocity servos with `kv = 1500 N m s/rad` and a rear differential
  term. With the differential, steady turning speed was `0.1993 m/s` for a
  `0.2 m/s` target, against `0.1984 m/s` without it.
- *Contact solver.* Elliptic friction cones with `impratio = 10` and the
  `implicitfast` integrator, chosen to suppress lateral tire creep.
- *Acceleration limit.* The speed target ramps from the last applied target,
  not from measured speed, so a lagging plant cannot stall the ramp.

Evidence from the automated tests (nominal variant, flat ground): neutral
semantic sites match the configuration to `1e-9 m`; an S-shaped `3 m` maneuver
at `+-0.2 m/s` ends within `0.009 m` and `0.001 rad` of an RK4 integration of
the specification's on-axle equations. These numbers come from the test
scenario only and are not a transfer result.

The MuJoCo-only tuning values live in `MujocoTuning` in code because the
strict version-1 YAML schema has no field for them. Moving them into the
vehicle configuration needs a schema revision.

## D-008 — MuJoCo alignment audit and verification scope

**Date:** 2026-11-12
**Status:** accepted

The first MuJoCo implementation was re-audited against the frozen vehicle and
dock contracts after merge. Geometry, semantic sites, names, signs, Ackermann
mapping, command limits, timing, configured physical values, dock outlines,
collision behavior, variants, deterministic stepping, and low-speed agreement
with the specified equations are covered by automated tests. The detailed
matrix is stored in `docs/design/mujoco-alignment-audit.md`.

At the time of this audit, direct parity with the project 2D simulator,
environment-level safety and success events, and controller transfer remained
unverified. D-009 records the subsequent plant-level parity work; event parity
and controller transfer still require higher layers. MuJoCo-only tuning remains
versioned in `MujocoTuning` rather
than in the YAML fingerprint. Until a configuration-schema revision, every
reported MuJoCo experiment must capture the Git revision and complete resolved
tuning values in its manifest.

Static analysis and branch-aware coverage now include the non-interactive
MuJoCo adapter, Ackermann mapping, and generator. Interactive GUI loops remain
outside the coverage gate and are checked through focused viewer tests and
manual smoke use.

## D-009 — Canonical two-dimensional kinematic backend

**Date:** 2026-11-12
**Status:** implemented and engineering-validated

The canonical 2D plant integrates the independent state
`[tractor_x, tractor_y, tractor_yaw, trailer_yaw, steering_angle, speed]` with
fixed-step classical RK4. At the frozen timing this means five `0.02 s`
substeps per `0.10 s` controller command. The simulator applies the shared
command supervisor once per command period, uses bounded acceleration to reach
the applied speed target, and holds the applied steering rate through the
substeps.

The trailer yaw equation includes both signed longitudinal and lateral tractor
hitch offsets. Hitch and trailer positions are reconstructed from shared rigid
geometry rather than integrated, preventing constraint drift. This keeps the
nominal on-axle case and configured off-axle variants in one implementation.

Automated tests cover analytic derivatives, forward and reverse motion,
constant steering, articulation response, angle wrapping, acceleration and
steering limits, exact rigid reconstruction, timestep convergence,
determinism, variants, and invalid input. Shared forward and reverse command
sequences also compare the 2D plant directly with MuJoCo. This is
plant-model validation, not evidence of controller transfer, docking success,
or physical fidelity.

## D-010 — Versioned Gymnasium docking task and teleoperation mode

**Date:** 2026-11-12
**Status:** implemented and engineering-validated

The primary `TrailerDocking-v0` action remains one normalized steering-rate
command at the configured fixed reverse target. A separately named `teleop`
mode adds normalized speed authority only for manual debugging and scripted
plant demonstrations; it does not change the planned PPO/NMPC comparison.

The task composes the canonical kinematic backend with goal-to-start reference
polylines, decreasing-progress preview sampling, chassis-footprint collision,
soft/hard articulation classification, docking-point success, workspace and
timeout events, a bounded 21-element path-relative observation, and a fully
decomposed reward. Episode events are evaluated independently from reward.
Scenario, preview, timeout, and reward values are fingerprinted from
`configs/environments/gymnasium_docking_v1.yaml`.

Validation includes Gymnasium's checker, deterministic and randomized seeded
resets, both action modes, each terminal/truncation path, invalid-action
atomicity, all trailer variants, a 1,000-step randomized stress rollout, RGB
rendering, and a complete default straight reverse. The latter reaches docking
success after 486 controller steps without collision. This establishes task
functionality, not controller performance or a comparative experiment.

## D-011 — Separate full-size service-center demonstration scenario

**Date:** 2026-11-12
**Status:** implemented for the interactive 2D environment

The compact paper-aligned vehicle remains the default `TrailerDocking-v0`
research baseline. A separately versioned service-center scenario is added for
human-scale inspection and teleoperation: a 5.8 m tractor, 14.0 m by 2.55 m
semitrailer, 18.0 m active loading zone, 20-door building façade, and a
200 m by 100 m maneuvering workspace. The building runs along the bottom
x-axis, with the active door centered at $x=100$ m. The interactive viewer
selects this scenario by default and can return to the compact baseline with
`--scenario compact`.

The active loading zone is a colored floor target rather than a parking slot
bounded by physical side walls. Its two longitudinal edges are absent from
both rendering and collision geometry; bay width still participates in the
complete-trailer containment success check. The service-center façade outside
the active door and rear boundary remain collidable. Manual limits are 2 m/s
forward and 1 m/s reverse so movement is observable across the larger yard. A
straight reverse acceptance rollout reaches docking success after 325 control
steps without collision. These values are demonstration configuration choices,
not identified real-vehicle parameters or controller results. Any future
algorithm experiment using this scenario must identify its three configuration
fingerprints separately from the compact baseline.

The full-size interactive scenario uses a $60^\circ$ terminal jackknife
threshold, while retaining its $35^\circ$ soft warning. This is deliberately
separate from the compact cross-simulator baseline, whose frozen hard threshold
remains $45^\circ$.

The service view is a rigid presentation transform of the dock-local state:
$(x_s,y_s,\psi_s)=(100-y_d,x_d,\psi_d+\pi/2)$. A regression test with unequal,
nonzero tractor and trailer headings requires the transformed body hitch points
and transformed hitch marker to coincide. This guards visualization integrity;
it does not alter the kinematic state or task calculations.
