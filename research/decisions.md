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

## D-012 — Learning pipeline reconciled onto the 2D Gymnasium task

**Date:** 2026-11-13
**Status:** implemented and software-tested; no registered experiment

The learning-pipeline and control-UI skeleton was first written against
MuJoCo, with its own reference path, tracker, observation, collision, success
rule, reward, and environment, before the canonical 2D backend (D-009) and
Gymnasium task (D-010) existed. The two lines were reconciled as follows.

- *One task definition.* `TrailerDockingEnv` and its path, tracking,
  observation, collision, jackknife, success, and reward modules are kept
  unchanged. The skeleton's duplicate environment, curve-then-straight
  planner, trailer-axle path type, tracker, observation layout, and
  docking-outcome module were removed.
- *Scenario wrapper.* `training/environment.py` wraps `TrailerDockingEnv` in
  `fixed_reverse` mode (D-002). It builds vehicle, dock, and task settings from
  an editable `Scenario` and samples each start pose from the scenario's start
  distribution. It adds no dynamics, observation, reward, or event logic.
- *Scenario provenance.* A scenario now names the task specification as well
  as the vehicle and dock, and its fingerprint includes all three.
- *Outcomes.* Training records and docking tests use the task's termination
  reasons: `success`, `collision`, `jackknife`, `out_of_bounds`, `time_limit`.
- *Backend.* At reconciliation, training and docking tests ran on the 2D
  kinematic backend only. D-013 adds the choice of simulator.
- *Reference route in viewers.* `paths.generators.docking_reference` is the
  environment's reference construction made callable, so the MuJoCo viewer and
  the UI draw the same route the task tracks.

Known limits: the scenario's default start ranges (6–9 m standoff) and the
training step limit (900) differ from the task file's defaults (4 m, 600), so
a run's manifest, not the task fingerprint alone, identifies its conditions.
Reward weights and start ranges are untuned. A 30,000-step run on this
pipeline completed, saved, and reloaded a policy; it is a pipeline check, not
an experiment, and its success rate is not reported as a result.

## D-013 — PPO training framework, simulator choice, and policy bundles

**Date:** 2026-11-13
**Status:** implemented and software-tested; no registered experiment

Requested by the user as the framework for running reinforcement-learning
training through one UI, on either simulator, with exportable results. It is
PPO-only; NMPC is implemented separately and plugs in through one hook.

- *Simulator choice.* `TrailerDockingEnv` takes `backend="2d"` (default,
  kinematic) or `"3d"` (MuJoCo). The simulator protocol's `reset` now accepts
  the articulated start pose both backends already supported. Observation,
  reward, and every terminal event are computed from the shared planar state
  in both cases. **Event parity between the backends is not validated**
  (the open gate of D-008), so 3D runs are engineering checks.
- *Scenario overrides.* Added tractor wheelbase and body width; trailer
  hitch-to-axle length, rear overhang, and body width; maximum steering angle
  and rate; and reverse speed. Footprints, wheel tracks, and neutral semantic
  sites are re-derived and pass the same validation as a YAML configuration;
  a trailer with any overridden dimension becomes the `custom` variant.
  `reverse_speed` changes the fixed reverse target of D-002 for that scenario
  only and is recorded in the scenario; the frozen files are untouched.
- *Domain randomization (proposal: start pose, trailer length, noise).* Start
  pose was already sampled. Trailer length can now be drawn per episode, which
  rebuilds the vehicle and simulator at each reset. Observation noise is
  zero-mean Gaussian on the normalized observation, clipped to its bounds. It
  is **not** a sensor model in physical units; the proposal's perfect-versus-
  noisy-state experiment needs that decision before it is run. The policy
  observation does not include trailer length (open question Q-001).
- *Campaigns.* A campaign repeats a run with seeds `seed`, `seed+1`, ...,
  tests each final policy for a fixed number of episodes from a separate
  evaluation seed (default 100000) on the chosen simulators, and selects the
  best run by success rate, then mean reward, on the first evaluation
  simulator. **Selecting on those test episodes makes them a validation set,
  not a held-out test set**; reported comparisons need separate seeds.
- *Bundles.* The selected policy is exported with its scenario, training
  configuration, evaluation summary, and provenance in one zip. Loading
  rebuilds the scenario and checks its fingerprint; the policy can then run in
  a freshly built 2D or 3D simulator.
- *Metrics.* Docking tests report the proposal's list: success, collision and
  jackknife rates, final errors, maximum articulation, maneuver time, path
  length, steering-rate RMS, and mean controller time per step.
- *Method switch.* `method="nmpc"` skips training and runs the controller
  returned by `controllers.nmpc.make_docking_controller(env)` on the same
  scenarios, seeds, simulators, and metrics. See
  `docs/design/nmpc-integration-handoff.md`.
- *Display.* The control UI is the only window needed: a top-down view for
  both simulators and an embedded offscreen MuJoCo rendering for 3D.

Hyperparameters are Stable-Baselines3 defaults except rollout length 1024 and
batch 256; the trajectory-state paper (Yan et al., 2024) does not report its
values. Not implemented: curriculum, parallel environments, resume,
TensorBoard. Evidence is software tests plus pipeline runs whose success rates
are not results.

### D-013 addendum — changes made after the adversarial audit (2026-11-13)

Three independent audits of this work confirmed defects that were fixed before
merge. The ones that change behaviour or interpretation:

- *Loading files from others.* Policy and bundle loading no longer calls
  Stable-Baselines3's metadata loader, which unpickles objects and could run
  code from a received file. Only the network weights are read
  (`weights_only`), into the default `MlpPolicy` architecture; a checkpoint
  with another architecture is rejected. The observation size recorded in a
  bundle is read from the weights.
- *Unsolvable scenarios are rejected.* `Scenario.build()` refuses a trailer
  that cannot satisfy the complete-containment success rule in its bay (this
  includes the `long` variant in the nominal 6 m bay, open question Q-006),
  both ends of a randomized trailer-length range, a fixed trailer length
  combined with a randomized one, and start ranges whose extremes begin an
  episode collided, out of bounds, or jackknifed.
- *Workspace bound (task change).* The outward out-of-bounds limit was
  `start x + 3.0 m` measured from the tractor rear axle, so a wheelbase of
  about 2.4 m or more started every episode out of bounds. It is now at least
  0.5 m ahead of the tractor nose. For the nominal tractor this is exactly
  the previous bound, so existing 2D results are unaffected.
- *Reference path (task change).* An offset start nearer the bay than the
  entry point (0.75 m outside the opening) used to get a path that doubled
  back out to that point, which a reversing vehicle cannot follow and which
  inflated the progress reward. Such starts now get a direct path. Default
  start ranges (4 m and 6-9 m standoff) are unaffected.
- *Wheel tracks.* Overriding a body width keeps the wheels the same distance
  inside the body sides instead of scaling the track.
- *Selection and failures.* Only complete tests are ranked; a failed run or
  failed test is recorded and the remaining runs still produce a result; the
  campaign manifest exists from the start and is rewritten after every run;
  every test episode is saved; campaign and bundle record the MuJoCo tuning
  whenever any phase ran in 3D; experiment IDs carry the method.
- *Provenance.* Manifests add the resolved PPO hyperparameters, interpreter,
  platform, command line, output checksums, and a hash of uncommitted changes.
  `cli export` records training settings as unknown unless given the run's
  manifest. Training and evaluation seed ranges may not overlap.

Not changed, and still open: bundles reference the base specification files
by path, so a scenario built on non-default files is portable only if those
files travel with it; the per-episode `vehicle_fingerprint` reported by the
task does not cover dimension or limit overrides (the scenario fingerprint
does, and is reported alongside it); a non-zero dock target yaw is not
supported consistently by the reference path and workspace.

## D-014 — Swappable route planners and route-following reward weights

**Date:** 2026-11-13
**Status:** implemented and software-tested; added after the D-013 audits and
not itself independently audited

Requested by the user: the planner should be a basic path-finding algorithm
that can be swapped in the UI, training should reward staying on the planned
route, and the route should be toggleable in the display.

- *Planner registry.* `paths/planners.py` maps a name to a planner.
  `TrailerDockingEnv(planner=...)` and `Scenario.planner` select one; the
  default `entry` is the route the task has always used, so existing behaviour
  is unchanged. The planner is part of the scenario fingerprint.
- *A\* planner.* `astar` searches an 8-connected 0.25 m grid for the trailer's
  docking point, with dock walls grown (as axis-aligned boxes) by the trailer
  half-width plus 0.15 m, shortens the grid path to line-of-sight corners, and
  ends with the straight centreline leg into the bay. With only the bay as an
  obstacle it differs from `entry` mainly for starts beside the bay. It treats
  the docking point as a point robot: it does **not** check tractor clearance,
  articulation, or that a reversing vehicle can follow the route. The
  proposal's requirement that generated paths be simulated and
  collision-checked before use is therefore not met by either planner, and
  Reeds-Shepp or Hybrid A\* remain the optional extensions it names.
- *Reward for staying on the route.* No new reward term was added. The task
  reward already contains the squared lateral and heading deviation from the
  reference path (weights 0.5 and 0.2 in `gymnasium_docking_v1.yaml`). The
  scenario can now override those two weights; an override is recorded in the
  scenario and its fingerprint, while the task fingerprint continues to
  describe the unmodified file.
- *Display.* One toggle shows or hides the route in the top-down view and in
  the embedded 3D picture, for the idle preview and for live runs.

Changing the planner or the weights changes the learning problem, so runs with
different settings are different experimental conditions, not repeats.

## D-015 — Static obstacles in both simulators

**Date:** 2026-11-13
**Status:** implemented and software-tested; added after the D-013 audits and
not itself independently audited; no experiment uses obstacles yet

The proposal's task includes avoiding obstacles as well as dock walls, and the
user asked for them in both simulators.

- *Model.* An obstacle is a rectangle in the dock frame (`core/obstacles.py`):
  centre, length, width, yaw, and height. A scenario holds a list of them;
  they are part of its fingerprint. The default is none, so the frozen task
  and the generated MuJoCo files are unchanged.
- *2D.* `evaluate_task` adds obstacle overlaps to the collision report
  (`trailer:obstacle_1`), using the existing separating-axis test on the
  chassis footprints. An obstacle collision terminates the episode with the
  same penalty as a wall collision.
- *3D.* The scene generator adds each obstacle as a box whose collision
  volume is extruded above both chassis (as for dock walls), so a low box
  cannot be driven over and the planar rule of the 2D task holds. Episode
  events still come from the shared planar check, not from MuJoCo contacts.
- *Planning.* The A* planner treats obstacles like walls. The clearance
  around walls and obstacles is now a cost penalty (factor 8) rather than a
  hard block, so a start that stands inside a clearance zone can still be
  planned; the real geometry remains impassable. Rotated obstacles are covered
  by their bounding boxes. The `entry` planner ignores obstacles.
- *What the policy sees.* Nothing: the observation is path-relative only. A
  policy avoids an obstacle only by tracking a route planned around it. This
  is a deliberate limit, consistent with the proposal's separation of path
  generation from tracking, and it means obstacle scenarios test route
  tracking accuracy rather than learned avoidance.
- *Validation.* A scenario is rejected when an obstacle overlaps the docked
  vehicle or an extreme of the start ranges. An obstacle between a start and
  the dock with the `entry` planner is accepted and simply makes the route
  collide; this is not detected.
- *UI.* Obstacles are entered as text, `x, y, length, width[, yaw[, height]]`
  separated by `;`, and are drawn in the top-down and 3D displays.

Dynamic obstacles remain a stretch goal and are not implemented.

