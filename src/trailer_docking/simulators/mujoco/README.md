# MuJoCo model and simulator contributor guide

This directory is the handoff contract for the 3D implementation. A contributor
should be able to build the first model from this guide without inventing
vehicle dimensions, signs, names, or controller semantics.

## Current state

The shared vehicle state, command, geometry, configuration loaders, safety
supervision, and conformance validation are implemented. The MJCF vehicle,
dock scene, Python adapter, and 2D/MuJoCo parity tests are not yet implemented.
Do not describe a 3D result as complete until the validation gates below pass.

## Read these sources in order

1. Repository [`AGENTS.md`](../../../../AGENTS.md) for development, evidence,
   and report-update rules.
2. The frozen
   [`vehicle-and-environment-specification.md`](../../../../docs/design/vehicle-and-environment-specification.md)
   for frames, equations, geometry, limits, hierarchy, and change control.
3. [`compact_semitrailer_v1.yaml`](../../../../configs/vehicles/compact_semitrailer_v1.yaml)
   for the machine-readable vehicle authority.
4. [`nominal_dock_v1.yaml`](../../../../configs/environments/nominal_dock_v1.yaml)
   for the machine-readable dock authority.
5. The repository [`architecture.md`](../../../../docs/design/architecture.md)
   for simulator boundaries and dependency direction.
6. `trailer_docking.core.state`, `core.geometry`, and `core.config` for the
   Python contracts the adapter must consume and produce.

The submitted proposal explains motivation and scope, but it does not override
the frozen specification. The YAML files are normative for exact values; the
design document is their required human-readable explanation. Resolve any
disagreement before modeling.

## Installation

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev,mujoco]'
.venv/bin/python -m trailer_docking.validation.vehicle_spec
```

The `mujoco` extra installs the official Python distribution. The package
contains the MuJoCo library, so a separate engine download is not normally
required. Use the official
[modeling guide](https://mujoco.readthedocs.io/en/stable/modeling.html) and
[MJCF reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html)
for engine behavior rather than copying unofficial XML patterns.

## Fixed cross-simulator conventions

- SI units: metres, seconds, radians, kilograms.
- Body axes: `+x` forward, `+y` left, `+z` up.
- World yaw: zero along `+x`, positive counterclockwise about `+z`.
- Positive steering turns left; positive speed moves forward.
- Articulation is `wrap(tractor_yaw - trailer_yaw)`.
- Shared steering is one virtual bicycle angle, not either wheel angle.
- Shared commands are `(steering_rate, target_speed)` at `10 Hz`.
- The primary PPO/NMPC experiment fixes target speed at `-0.20 m/s`; PPO
  selects only bounded steering rate.
- MuJoCo integrates at `0.002 s`, giving 50 physics steps per control period.
- Planar pose and metrics use semantic sites, never arbitrary body origins.

The adapter must map the virtual steering target to Ackermann left/right wheel
angles using the equation in the frozen specification. Rear-wheel velocity
targets use `omega = target_speed / wheel_radius`. Front-wheel and trailer-wheel
spin joints are passive in version 1.

## Frozen nominal dimensions

| Quantity | Value |
|---|---:|
| Tractor wheelbase | `1.90 m` |
| Tractor body length / width | `2.90 m / 1.00 m` |
| Front and rear track | `0.80 m` |
| Hitch offset from tractor rear axle | `0.00 m` longitudinal, `0.00 m` lateral |
| Hitch height | `0.75 m` |
| Trailer hitch-to-axle distance | `4.00 m` |
| Trailer body length / width | `5.50 m / 1.00 m` |
| Wheel radius / width | `0.30 m / 0.12 m` |
| Steering / steering-rate limits | `0.50 rad / 0.30 rad/s` |
| Hard articulation / mechanical hinge limits | `45 deg / 85 deg` |

Short (`2.00 m`) and long (`5.00 m`) hitch-to-axle variants are future
experiment inputs. The first model uses the nominal variant, but geometry must
be parameterized rather than copied into three unrelated models.

## Required MJCF API

The exact names below are frozen in `mujoco.required_names`. The Python loader
validates that the configuration retains them; the future adapter must validate
that the compiled `MjModel` contains them.

### Bodies

```text
tractor_body
trailer_body
front_left_steer_link
front_right_steer_link
front_left_wheel
front_right_wheel
rear_left_wheel
rear_right_wheel
trailer_left_wheel
trailer_right_wheel
```

### Joints

```text
tractor_free
hitch_yaw
front_left_steer
front_right_steer
front_left_wheel_spin
front_right_wheel_spin
rear_left_wheel_spin
rear_right_wheel_spin
trailer_left_wheel_spin
trailer_right_wheel_spin
```

`tractor_free` should be an MJCF `freejoint`. The hitch and steering axes are
local `+z`; wheel-spin axes are local `+y`. Confirm signs in simulation instead
of assuming XML nesting makes them correct.

### Semantic sites

```text
tractor_rear_axle
tractor_front_axle
hitch
trailer_axle
trailer_rear_face
docking_point
```

In the straight neutral pose with `tractor_rear_axle` at world `(0, 0)`:

| Site | Required world position |
|---|---:|
| `tractor_rear_axle` | `(0.00, 0.00, 0.30)` |
| `tractor_front_axle` | `(1.90, 0.00, 0.30)` |
| `hitch` | `(0.00, 0.00, 0.75)` |
| `trailer_axle` | `(-4.00, 0.00, 0.30)` |
| `trailer_rear_face` | `(-5.00, 0.00, 1.65)` |
| `docking_point` | `(-5.00, 0.00, 0.00)` |

### Actuators

```text
front_left_steer_position
front_right_steer_position
rear_left_wheel_velocity
rear_right_wheel_velocity
```

Actuator gains, damping, armature, inertias, and contact parameters may be tuned
while preserving this API and the configured physical limits. Record every
tuning change and the evidence that motivated it.

## Logical hierarchy

```text
tractor_body [tractor_free]
├── front_left_steer_link [front_left_steer]
│   └── front_left_wheel [front_left_wheel_spin]
├── front_right_steer_link [front_right_steer]
│   └── front_right_wheel [front_right_wheel_spin]
├── rear_left_wheel [rear_left_wheel_spin]
├── rear_right_wheel [rear_right_wheel_spin]
└── trailer_body [hitch_yaw]
    ├── trailer_left_wheel [trailer_left_wheel_spin]
    └── trailer_right_wheel [trailer_right_wheel_spin]
```

XML may be factored with `<include>`, but the compiled hierarchy and names must
remain equivalent. Put reusable vehicle MJCF in `models/`, complete dock/yard
worlds in `scenes/`, and optional appearance assets in `assets/`.

## Geometry and contact standards

- Use primitive boxes/cylinders for collision and separate visual geoms from
  collision geoms. Decorative meshes must never define experimental collision
  behavior.
- The planar projection of collision boxes must exactly match the configured
  tractor and trailer footprints.
- The hitch is a passive yaw joint with a mechanical range of `+-85 deg`.
  Episode safety still terminates at `+-45 deg`.
- Version 1 has no suspension. It has one tractor rear axle and one trailer
  axle represented by left/right wheels.
- The nominal initial physical values are `760 kg` tractor chassis,
  `1180 kg` trailer body, `10 kg` per wheel, gravity `[0, 0, -9.81]`, and
  friction `[1.0, 0.005, 0.0001]`. Treat these as baselines to validate, not
  literature-proven ground truth.
- If MJCF must repeat YAML numbers, keep YAML authoritative and add a parity
  test that detects drift. Prefer a small deterministic generator or model
  builder once multiple vehicle variants are required.

## Intended files

The first contribution should establish this layout:

```text
mujoco/
├── adapter.py                         # Shared command/state translation
├── models/
│   └── compact_semitrailer_v1.xml     # Reusable vehicle model
├── scenes/
│   └── nominal_dock_v1.xml            # Complete nominal docking scene
└── assets/                             # Optional visual-only assets
```

Put MuJoCo integration tests under `tests/integration/mujoco/`. Reusable Python
logic belongs in the package, not in one-off scripts or notebooks.

## Recommended implementation sequence

1. Build and compile the primitive vehicle skeleton with every required name.
2. Match all six neutral semantic-site positions and collision footprints.
3. Verify a free tractor and passive hitch settle without explosive contacts.
4. Add passive wheel spin, then Ackermann steering position actuators.
5. Add rear-wheel velocity actuation and verify straight forward/reverse signs.
6. Build the nominal dock from the environment YAML and verify wall contacts.
7. Implement the adapter from `VehicleCommand` to actuators and from named
   MuJoCo objects to `VehicleState`.
8. Add deterministic smoke tests and 2D/MuJoCo parity tests before controller
   integration or visual polishing.

## Required validation gates

- The MJCF compiles with the supported MuJoCo dependency and every required
  body, joint, site, and actuator resolves by name.
- Neutral-site positions and body dimensions match configuration within a
  documented numerical tolerance.
- Positive yaw, steering, steering rate, and articulation match shared signs.
- From yaw zero, positive speed moves toward world `+x` and negative speed
  moves toward world `-x`.
- Equal left/right steering targets produce straight wheels; virtual steering
  reconstructs from the physical angles consistently.
- The `10 Hz` command boundary is independent of the `0.002 s` physics step.
- Dock collision, hard jackknife termination, and docking error use the same
  geometry and thresholds as 2D.
- A fixed command sequence is repeatable and qualitatively agrees with the 2D
  backend. Exact trajectories are not expected because MuJoCo adds contacts
  and actuator dynamics.
- `ruff`, `mypy`, unit tests, MuJoCo integration tests, and the vehicle-spec
  conformance command pass before handoff.

## Contribution and evidence rules

Do not silently change a frozen sign, dimension, name, command, controller
period, success rule, or safety threshold. Such a change requires coordinated
updates to both YAML specifications, the frozen design document, 2D and MuJoCo
tests, the decision log, and the living paper. Keep screenshots and videos as
debugging evidence, but preserve machine-readable manifests and metrics for any
claim used in the report.

A MuJoCo handoff is complete only when another contributor can install the
extra, load the named scene, run the tests, reproduce the neutral pose, and
trace every model value back to a versioned repository source.
