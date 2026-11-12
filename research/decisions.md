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

## D-007 — First MuJoCo vehicle, dock, and adapter

**Date:** 2026-11-11
**Status:** implemented; validated against the specification's kinematic
equations, not yet against the project's 2D simulator

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

## D-008 — Learning-pipeline and control-UI skeleton

**Date:** 2026-11-11
**Status:** interfaces implemented and tested; reward, planner, and
observation scaling are placeholders; no trained result exists

The skeleton follows the proposal's pipeline: start state and dock, then a
path planner, then path-relative errors and preview points, then the policy,
then a bounded steering-rate command at the fixed reverse speed.

- *Simulator protocol.* `simulators/base.py` defines `reset`, `step`, `state`,
  `applied_command`, and `controller_period`. The environment, training, and
  UI depend only on it. MuJoCo is the default backend until the 2D simulator
  exists.
- *Reference path.* The path is the route of the trailer-axle midpoint, with
  the desired trailer heading at each waypoint, arc length, and curvature.
  This matches the tracked point of the trajectory-state paper (Yan et al.,
  2024) and of the NMPC paper's reference pose.
- *Planner.* `FixedDockingPlanner` is a predefined curve onto the bay axis
  followed by a straight reverse, the proposal's "predefined straight and
  curved docking paths". It is geometric only: it does not verify trailer
  feasibility or collisions. The trajectory-state paper uses Dubins paths; the
  proposal names Reeds-Shepp or Hybrid A* as later options.
- *Observation.* Trailer lateral error, trailer and tractor heading errors,
  articulation angle and rate, steering angle and rate, remaining distance,
  and three preview points at 1, 2, and 4 m. Each preview carries the error
  triple of the trajectory-state paper (lateral offset in the trailer frame,
  trailer and tractor heading errors). That paper does not state the preview
  count, spacing, sign convention, or normalization, so those are project
  choices still to be made and are plain constants for now.
- *Reward.* Progress along the path, lateral, heading, and articulation
  penalties, a small steering-rate penalty, a docking bonus, and a failure
  penalty, as listed in the proposal. All weights are untuned placeholders.
- *Termination.* Docking success (specification section 11.1), jackknife,
  planar wall collision, leaving the path corridor, and timeout, all computed
  from the shared planar state so they are identical across backends.
- *Scenario.* An editable scenario is the frozen base specifications plus
  explicit overrides and a start-pose distribution. The frozen YAML files are
  never modified. Mass and ground friction are marked physics-only because
  the 2D kinematic model has no counterpart for them.

Evidence so far is limited to software checks: the environment passes the
Gymnasium API checker, a zero-steer command docks from an aligned start, and a
30,000-step PPO smoke run completed, saved and reloaded a policy, and reached
0 % success. That run is a pipeline check, not an experiment.

Path generation, collision checking, and baseline control are led by Nandan in
the proposal's division of work; the planner, collision, and success modules
here are minimal stand-ins that define the interfaces and are expected to be
replaced or extended.
