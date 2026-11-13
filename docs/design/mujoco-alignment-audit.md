# MuJoCo alignment audit

**Audit date:** 2026-11-12

**Vehicle:** `compact-semitrailer-v1`

**Dock:** `nominal-dock-v1`

**Baseline revision:** `690aa89` plus the verification/documentation changes
in this working tree

## Purpose

This audit checks the implemented MuJoCo vehicle, dock, generator, and adapter
against the frozen vehicle-core contract. It records what is verified now and
what must wait for applying the shared Gymnasium task to MuJoCo. It is not
a controller experiment or evidence of real-vehicle fidelity.

Authoritative inputs:

- [`vehicle-and-environment-specification.md`](vehicle-and-environment-specification.md)
- [`compact_semitrailer_v1.yaml`](../../configs/vehicles/compact_semitrailer_v1.yaml)
- [`nominal_dock_v1.yaml`](../../configs/environments/nominal_dock_v1.yaml)
- [`gymnasium_docking_v1.yaml`](../../configs/environments/gymnasium_docking_v1.yaml)

Implemented artifacts:

- `simulators/mujoco/builder.py`: deterministic MJCF generation
- `simulators/mujoco/adapter.py`: shared command/state translation
- `simulators/mujoco/ackermann.py`: virtual/physical steering conversion
- generated vehicle and nominal-dock MJCF files
- 3D and shared-state top-down viewers

## Contract alignment

| Contract item | Implementation and evidence | Status |
|---|---|---|
| SI units and `+x` forward, `+y` left, `+z` up | MJCF uses configured dimensions and gravity; sign tests cover forward/reverse motion and yaw | Verified |
| Articulation `wrap(tractor_yaw - trailer_yaw)` | Adapter reconstructs poses from semantic sites; positive, negative, and wrap-boundary cases are tested | Verified |
| Nominal tractor/trailer dimensions | Generator reads the typed vehicle configuration; collision-box footprint, height, and neutral-site tests match YAML | Verified |
| Short, nominal, and long trailer lengths | All variants use one parameterized generator and reproduce their derived semantic sites | Verified |
| Frozen body, joint, site, and actuator names | Adapter resolves every configured name; generator and compiled-model tests enforce coverage | Verified |
| Virtual steering and Ackermann wheels | Forward and inverse mappings round-trip; steering sign, rate, stops, and turning response are tested | Verified |
| Command `(steering_rate, target_speed)` | Adapter accepts `VehicleCommand`, clips rate/speed/angle, applies acceleration limiting, and reports the applied command | Verified |
| `10 Hz` control and `0.002 s` physics | One adapter step performs 50 physics steps and advances exactly `0.10 s` | Verified |
| Masses, gravity, friction, and torque limits | Generated/compiled MJCF values are compared with vehicle YAML | Verified |
| Dock dimensions and target | Generated wall outlines, target site, visible wall height, and collision behavior are checked against dock YAML | Verified |
| Primitive collision geometry | Chassis boxes and wheel cylinders are generated without decorative collision meshes | Verified |
| Straight reversing and deterministic stepping | Forward/reverse signs, speed tracking, rest stability, reset, and repeated-rollout equality are tested | Verified |
| Low-speed kinematic behavior | Scripted forward and reverse maneuvers agree with a test-local RK4 integration within documented tolerances | Provisionally verified |
| Soft/hard articulation and success events | The task computes these events from the shared state and can run on MuJoCo (`TrailerDockingEnv(backend="3d")`); scenarios comparing event boundaries across backends remain | Pending cross-backend event parity |
| Direct 2D/MuJoCo parity | Shared forward and reverse command sequences compare final planar pose and yaw within explicit tolerances | Verified at plant-trajectory level |
| NMPC/PPO transfer | No controller is implemented or transferred | Pending controllers |

## Deliberate MuJoCo-only modeling choices

These choices preserve the planar contract while adding a physical plant:

- Wheel contact uses a thin `0.02 m` cylinder on the wheel center plane; a
  full-width visual cylinder retains configured wheel mass and appearance.
- Wheels contact the ground, while chassis boxes contact dock walls. This
  matches the current planar collision-footprint definition. Steered wheels
  extending beyond the chassis footprint remain an open question.
- Dock collision volumes are vertically extruded above every chassis, while
  separate visual geoms retain the configured `1.00 m` wall height. Thus wall
  contact implements the 2D footprint rule instead of allowing a body to pass
  over a short wall.
- Rear drive targets include a differential term so the rear-axle midpoint
  tracks the shared target speed while turning.
- MuJoCo produces small measured overshoots despite bounded commands. The
  backend publishes a measured-state tolerance; the eventual environment rule
  for applying it remains open.

Exact gains, damping, armature, solver, and contact-width choices currently
live in the version-controlled `MujocoTuning` dataclass and are explained in
research decision D-007. They are therefore reproducible from a Git revision,
but they are not part of the vehicle YAML or its fingerprint. Before reporting
experiments, either move them into a versioned configuration schema or include
the complete resolved tuning values in every experiment manifest.

## Verification performed

From the repository root:

```bash
.venv/bin/ruff check src tests
.venv/bin/mypy
.venv/bin/pytest --cov --cov-report=term-missing -q
.venv/bin/python -m trailer_docking.validation.vehicle_spec
.venv/bin/python -m trailer_docking.simulators.mujoco.builder
```

In the current working tree and dependency set, all 266 tests passed. The
conformance report passed, and both committed MJCF files matched regenerated
output. Coverage is branch-aware and now includes the non-interactive MuJoCo
adapter, Ackermann mapping, and builder; interactive viewer loops remain
outside the coverage gate.

## Gate after kinematics work

The canonical 2D model now reuses the shared state, command, geometry, timing,
limits, supervisor, and configuration loaders. Shared forward and reverse
command sequences establish direct plant-trajectory parity. Shared path,
collision, jackknife, and goal logic exists in the verified Gymnasium task,
and since 2026-11-13 that task can also drive the MuJoCo plant. The remaining
parity gate is to compare event boundaries between the two backends under
shared scenarios.
