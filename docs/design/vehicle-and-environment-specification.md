# Vehicle and environment specification

**Specification ID:** `compact-semitrailer-v1`  
**Status:** frozen; vehicle core, canonical 2D kinematic backend, and first
MuJoCo implementation verified; task-level geometry and Gymnasium pending

**Machine-readable sources:**
[`configs/vehicles/compact_semitrailer_v1.yaml`](../../configs/vehicles/compact_semitrailer_v1.yaml)
and
[`configs/environments/nominal_dock_v1.yaml`](../../configs/environments/nominal_dock_v1.yaml)

Both configuration files use `schema_version: 1`. The schema version identifies
the software contract; the specification ID identifies a particular vehicle or
dock. A new set of dimensions may use a new specification ID without changing
the schema or controller interfaces.

**Implementation status:** the typed state, command, geometry, configuration,
fingerprinting, command-supervision, and specification-conformance layers are
implemented and tested. Kinematic state propagation is intentionally deferred
to the next milestone.

Run the executable contract check from the repository root with:

```bash
python -m trailer_docking.validation.vehicle_spec
```

## 1. Purpose and scope

This document is the interface contract between the 2D kinematic simulator and
the 3D MuJoCo model. Both implementations must use the same semantic reference
frames, dimensions, signs, controller command, limits, and docking definition.
MuJoCo may contain additional physical state, but it must translate that state
to the planar contract defined here.

The nominal vehicle is deliberately a compact, research-scale conventional
semitrailer rather than a dimensionally accurate highway tractor. Its principal
dimensions follow the NMPC paper's experimental vehicle: a 1.9 m tractor
wheelbase, 4.0 m hitch-to-trailer-axle distance, 1.0 m vehicle width, 0.2 m/s
maximum reverse speed, and 0.5 rad steering limit. This keeps the first
experiments close to the literature while retaining all geometry needed for a
consistent 3D model.

Version 1 contains one front-steered tractor and one non-steered, single-axle
semitrailer. It excludes suspension, tire deformation, powertrain dynamics,
road slope, and multiple trailers. Those exclusions are modeling choices, not
missing interface definitions.

## 2. Normative conventions

The words **must**, **should**, and **may** indicate required, recommended, and
optional behavior respectively.

All quantities use SI units:

- distance: metres (`m`)
- time: seconds (`s`)
- angle: radians (`rad`)
- mass: kilograms (`kg`)
- force and torque: newtons (`N`) and newton-metres (`N m`)

Angles exposed through the shared state must be wrapped to `[-pi, pi)`. Internal
MuJoCo joint coordinates may use a different continuous representation, but
the adapter must convert them.

## 3. Coordinate systems and signs

### 3.1 World frame

The shared planar world frame is right-handed:

- `+x`: right in a conventional overhead plot
- `+y`: up in a conventional overhead plot
- `+z`: upward in MuJoCo
- yaw `0`: body forward axis aligned with world `+x`
- positive yaw: counterclockwise rotation about `+z`

Every body-local frame uses:

- `+x`: forward
- `+y`: left
- `+z`: up

Longitudinal speed is positive forward and negative in reverse. Steering angle
is positive for a left turn. The nominal reverse-speed target is therefore
`-0.2 m/s`.

### 3.2 Articulation

The shared articulation angle is

```text
articulation = wrap(tractor_yaw - trailer_yaw)
```

Positive articulation means the tractor heading is counterclockwise from the
trailer heading. This definition is normative even if a MuJoCo hinge joint has
the opposite native sign.

The articulation rate is

```text
articulation_rate = tractor_yaw_rate - trailer_yaw_rate
```

### 3.3 Steering

The shared steering angle is the virtual front-wheel angle of the kinematic
bicycle model. It is not either physical front wheel's angle. The MuJoCo
adapter must convert this virtual angle to left and right Ackermann angles.

For a nonzero virtual steering angle `delta`, define

```text
R = wheelbase / tan(delta)
delta_left  = atan(wheelbase / (R - front_track / 2))
delta_right = atan(wheelbase / (R + front_track / 2))
```

The expressions naturally exchange the inner and outer wheels for negative
steering. At zero steering, both physical steering angles are zero.

## 4. Semantic reference frames

Controllers and metrics must use semantic frames, not raw MuJoCo body origins.
The MuJoCo model should create named `site` elements for these frames.

| Frame/site | Definition |
|---|---|
| `tractor_rear_axle` | Ground-plane projection of the rear-axle midpoint |
| `tractor_front_axle` | Front-axle midpoint, `1.9 m` forward of the rear axle |
| `hitch` | Kingpin/yaw-joint center |
| `trailer_axle` | Trailer-axle midpoint |
| `trailer_rear_face` | Center of the trailer's rear vertical face |
| `docking_point` | Ground-plane projection of `trailer_rear_face` |

The planar tractor pose is the pose of `tractor_rear_axle`. The planar trailer
pose is the pose of `trailer_axle`. The final docking error is measured at
`docking_point`, not at the trailer axle or MuJoCo body center.

In MuJoCo, axle sites are placed at wheel-center height, `trailer_rear_face` is
placed at the geometric center of the rear face, and `docking_point` is placed
at ground height. The shared planar adapter discards site `z` coordinates.

## 5. Frozen nominal geometry

### 5.1 Tractor

The tractor rear axle is the local origin for all longitudinal dimensions.

| Parameter | Symbol | Value | Meaning |
|---|---:|---:|---|
| Wheelbase | `L_t` | `1.90 m` | Rear axle to front axle |
| Front overhang |  | `0.60 m` | Front axle to front face |
| Rear overhang |  | `0.40 m` | Rear axle to rear face |
| Total body length |  | `2.90 m` | `0.60 + 1.90 + 0.40` |
| Body width |  | `1.00 m` | Collision and visual body width |
| Front track |  | `0.80 m` | Front left/right wheel-center separation |
| Rear track |  | `0.80 m` | Rear left/right wheel-center separation |
| Body height |  | `1.40 m` | Initial MuJoCo visual/collision height |
| Body lower surface |  | `0.30 m` | Height above ground |
| Hitch longitudinal offset | `h` | `0.00 m` | Positive is forward of rear axle |
| Hitch lateral offset |  | `0.00 m` | Positive is left |
| Hitch height |  | `0.75 m` | MuJoCo yaw-joint height |

The nominal model is on-axle (`h = 0`). The implementation must nevertheless
accept a signed longitudinal hitch offset so later configurations can reproduce
the trajectory-state paper's off-axle model without changing interfaces.

Relative to `tractor_rear_axle`, the tractor collision footprint is
`x in [-0.40, 2.50]`, `y in [-0.50, 0.50]`. The nominal MuJoCo tractor box has
center `(1.05, 0.00, 1.00)` and full size `(2.90, 1.00, 1.40)` in the same
semantic frame.

### 5.2 Trailer

The hitch is the local origin for trailer longitudinal dimensions.

| Parameter | Symbol | Value | Meaning |
|---|---:|---:|---|
| Hitch-to-axle distance | `L_r` | `4.00 m` | Primary trailer-length parameter |
| Front overhang |  | `0.50 m` | Trailer body extends ahead of kingpin |
| Rear overhang |  | `1.00 m` | Axle to rear face |
| Total body length |  | `5.50 m` | `0.50 + 4.00 + 1.00` |
| Body width |  | `1.00 m` | Collision and visual body width |
| Axle track |  | `0.80 m` | Left/right wheel-center separation |
| Body height |  | `1.80 m` | Box body height |
| Body lower surface |  | `0.75 m` | Matches nominal kingpin/floor height |

At zero articulation, the trailer frame is aligned with the tractor frame. In
trailer-local coordinates:

```text
hitch              = ( 0.00, 0.00)
trailer_axle       = (-4.00, 0.00)
trailer_front_face = ( 0.50, 0.00)
trailer_rear_face  = (-5.00, 0.00)
```

Equivalently, relative to `trailer_axle`, the hitch is at `(4.00, 0.00)` and
the trailer collision footprint is `x in [-1.00, 4.50]`,
`y in [-0.50, 0.50]`. The nominal MuJoCo trailer box has center
`(1.75, 0.00, 1.65)` and full size `(5.50, 1.00, 1.80)` relative to the
`trailer_axle` semantic frame.

### 5.3 Wheels

| Parameter | Value |
|---|---:|
| Radius | `0.30 m` |
| Width | `0.12 m` |
| Tractor wheel count | `4` |
| Trailer wheel count | `2` |

Version 1 uses fixed axle locations with wheel-spin joints and front-wheel
steering joints. It does not model suspension.

### 5.4 Neutral assembled pose

The model's neutral pose has zero tractor yaw, trailer yaw, articulation, and
steering. With `tractor_rear_axle` at planar world `(0, 0)`, required site
positions are:

| Site | MuJoCo world position in neutral pose |
|---|---:|
| `tractor_rear_axle` | `(0.00, 0.00, 0.30)` |
| `tractor_front_axle` | `(1.90, 0.00, 0.30)` |
| `hitch` | `(0.00, 0.00, 0.75)` |
| `trailer_axle` | `(-4.00, 0.00, 0.30)` |
| `trailer_rear_face` | `(-5.00, 0.00, 1.65)` |
| `docking_point` | `(-5.00, 0.00, 0.00)` |

The 2D model uses the `x` and `y` components of these relationships. This
neutral pose is the first geometry parity fixture for both implementations.

## 6. Trailer-length variants

The primary length varied in experiments is `hitch_to_axle`. The following
named configurations reproduce the range emphasized by the trajectory-state
paper:

| Variant | Hitch-to-axle | Front overhang | Rear overhang | Total body length |
|---|---:|---:|---:|---:|
| `short` | `2.00 m` | `0.50 m` | `1.00 m` | `3.50 m` |
| `nominal` | `4.00 m` | `0.50 m` | `1.00 m` | `5.50 m` |
| `long` | `5.00 m` | `0.50 m` | `1.00 m` | `6.50 m` |

The first 2D and MuJoCo implementations must use the nominal variant. Variant
support should be parameterized; separate hard-coded models are not permitted.

## 7. Shared planar state

The external state returned by either simulator is:

```text
VehicleState
├── tractor_pose:     (x, y, yaw) at tractor_rear_axle
├── trailer_pose:     (x, y, yaw) at trailer_axle
├── hitch_position:   (x, y)
├── steering_angle:   virtual bicycle steering angle
├── steering_rate
├── longitudinal_speed  (at tractor_rear_axle, positive forward)
├── tractor_yaw_rate
├── trailer_yaw_rate
├── articulation_angle
├── articulation_rate
└── simulation_time
```

The 2D integrator may use a smaller internal state and derive redundant values.
The MuJoCo adapter must measure or calculate each field from named sites, body
velocities, and joints. Both backends must expose the same units and signs.

The 2D integrator's minimal state is

```text
[tractor_x, tractor_y, tractor_yaw, trailer_yaw, steering_angle,
 longitudinal_speed]
```

Longitudinal speed is a state because the configured acceleration limit is
applied explicitly. The hitch and trailer-axle positions are reconstructed
from geometry. Redundant positions are not independently integrated, so the
rigid hitch constraint cannot drift.

## 8. Shared controller command

The controller-facing command is

```text
VehicleCommand
├── steering_rate: rad/s
└── target_speed:  m/s
```

During the first PPO experiments, `target_speed` is fixed at `-0.2 m/s`; PPO
outputs only normalized steering rate. Keeping target speed in the common
command allows NMPC, scripted tests, and future experiments to command it
without changing the simulator interface.

The 2D backend accelerates toward target speed, integrates and clips steering
angle, then uses instantaneous speed in the kinematic equations. The MuJoCo
backend integrates the virtual steering
target, maps it to left/right wheel angles, and converts target speed to rear
wheel angular-velocity targets using `omega = speed / wheel_radius`.

## 9. Limits, rates, and timing

| Parameter | Nominal value |
|---|---:|
| Maximum steering magnitude | `0.50 rad` |
| Maximum steering-rate magnitude | `0.30 rad/s` |
| Maximum forward speed | `0.20 m/s` |
| Maximum reverse-speed magnitude | `0.20 m/s` |
| Nominal reverse-speed target | `-0.20 m/s` |
| Maximum longitudinal acceleration | `0.40 m/s^2` |
| Soft articulation threshold | `0.610865 rad` (`35 deg`) |
| Hard/jackknife threshold | `0.785398 rad` (`45 deg`) |
| MuJoCo mechanical hinge limit | `1.483530 rad` (`85 deg`) |
| Controller period | `0.10 s` (`10 Hz`) |
| 2D integration substep | `0.02 s` |
| MuJoCo physics timestep | `0.002 s` |
| Nominal command delay | `0.00 s` |

The safety supervisor terminates an episode at the hard articulation threshold.
The wider MuJoCo hinge limit is a mechanical guard only; it must not replace the
shared safety threshold.

The nominal 2D actuator is ideal apart from rate, angle, speed, and acceleration
limits. Optional first-order steering and speed response parameters are reserved
for transfer-robustness experiments and default to disabled.

## 10. Planar kinematic model

The implementation supports general signed longitudinal and lateral hitch
offsets. The nominal validation case is the on-axle specialization. With
tractor rear-axle pose
`(x_t, y_t, psi_t)`, trailer yaw `psi_r`, virtual steering `delta`, tractor
wheelbase `L_t`, trailer hitch-to-axle distance `L_r`, and longitudinal speed
`v`, the nominal on-axle equations are

```text
x_t_dot   = v cos(psi_t)
y_t_dot   = v sin(psi_t)
psi_t_dot = v tan(delta) / L_t
psi_r_dot = v sin(psi_t - psi_r) / L_r
delta_dot = steering_rate
v_dot     = longitudinal_acceleration
```

For signed longitudinal hitch offset `a`, lateral offset `b`, tractor yaw rate
`omega = v tan(delta) / L_t`, and articulation `alpha = psi_t - psi_r`, the
implemented general trailer equation is

```text
psi_r_dot = ((v - b omega) sin(alpha) + a omega cos(alpha)) / L_r
```

The trailer axle is

```text
x_r = x_t - L_r cos(psi_r)
y_r = y_t - L_r sin(psi_r)
```

The backend uses classical fixed-step fourth-order Runge--Kutta (RK4) with the
configured `0.02 s` substep, five substeps per controller command. Automated
tests cover timestep convergence, forward and reverse motion, constant
steering, articulation, angle wrapping, rigid reconstruction, limit handling,
invalid inputs, and deterministic replay.

## 11. Nominal loading dock

The dock has its own right-handed frame:

- origin: center of the closed rear wall/door plane
- `+x`: outward from the rear wall toward the yard
- `+y`: left when looking outward from the wall
- `+z`: up

The nominal bay is:

| Parameter | Value |
|---|---:|
| Interior width | `1.50 m` |
| Interior depth | `6.00 m` |
| Wall thickness | `0.10 m` |
| Visible wall height in MuJoCo | `1.00 m` |
| Collision margin | `0.03 m` |

The interior occupies `x in [0, 6]` and `y in [-0.75, 0.75]`. Side walls
occupy the bay edges and the back wall/door plane lies at `x = 0`. The opening
to the yard is at `x = 6`.

At the target pose, the trailer points outward (`yaw = 0`) and its rear face is
near the back wall:

```text
target docking point = (0.25, 0.00)
target trailer yaw   = 0.00 rad
```

This leaves `0.25 m` nominal clearance between the trailer rear face and the
back wall and places the nominal trailer front face at `x = 5.75 m`, inside the
`6.0 m` bay.

### 11.1 Docking success

Success requires all of the following at the same control step:

- docking-point longitudinal error no greater than `0.10 m`
- docking-point lateral error no greater than `0.10 m`
- trailer heading error no greater than `0.052360 rad` (`3 deg`)
- no tractor, trailer, wall, or obstacle collision
- articulation magnitude below the hard threshold
- the complete trailer footprint inside the nominal bay interior, excluding the
  configured collision margin

A low final speed is not required in version 1 because the core PPO task uses a
fixed reverse speed and terminates when the target tolerance is entered.

## 12. MuJoCo model contract

### 12.1 Required body and joint hierarchy

The implemented MuJoCo model follows this logical hierarchy; generated XML may
be factored differently in a later compatible revision.

```text
tractor_body [tractor_free: free joint]
├── front_left_steer_link [front_left_steer: hinge z]
│   └── front_left_wheel [front_left_wheel_spin: hinge y]
├── front_right_steer_link [front_right_steer: hinge z]
│   └── front_right_wheel [front_right_wheel_spin: hinge y]
├── rear_left_wheel [rear_left_wheel_spin: hinge y]
├── rear_right_wheel [rear_right_wheel_spin: hinge y]
└── trailer_body [hitch_yaw hinge z at hitch]
    ├── trailer_left_wheel [trailer_left_wheel_spin: hinge y]
    └── trailer_right_wheel [trailer_right_wheel_spin: hinge y]
```

The hitch is yaw-only in version 1. Pitch and roll articulation are intentionally
excluded so the 3D model implements the same planar mechanism as the 2D model.

### 12.2 Required names

The XML must expose at least these stable names:

```text
body: tractor_body
body: trailer_body
body: front_left_steer_link
body: front_right_steer_link
body: front_left_wheel
body: front_right_wheel
body: rear_left_wheel
body: rear_right_wheel
body: trailer_left_wheel
body: trailer_right_wheel
joint: tractor_free
joint: hitch_yaw
joint: front_left_steer
joint: front_right_steer
joint: front_left_wheel_spin
joint: front_right_wheel_spin
joint: rear_left_wheel_spin
joint: rear_right_wheel_spin
joint: trailer_left_wheel_spin
joint: trailer_right_wheel_spin
site: tractor_rear_axle
site: tractor_front_axle
site: hitch
site: trailer_axle
site: trailer_rear_face
site: docking_point
actuator: front_left_steer_position
actuator: front_right_steer_position
actuator: rear_left_wheel_velocity
actuator: rear_right_wheel_velocity
```

Renaming one of these items requires a specification-version change or an
explicit compatibility alias in the MuJoCo adapter.

The machine-readable `mujoco.required_names` block in the vehicle YAML is the
normative list used by validation code. This section is its human-readable
counterpart and must be updated in the same change.

### 12.3 Initial physical parameters

These are initial MuJoCo parameters, not inputs to the 2D kinematic equations:

| Parameter | Value |
|---|---:|
| Tractor chassis mass | `760 kg` |
| Trailer body mass | `1180 kg` |
| Wheel mass | `10 kg` each |
| Ground gravity | `[0, 0, -9.81] m/s^2` |
| Contact friction | `[1.0, 0.005, 0.0001]` |
| Rear-wheel actuator torque limit | `400 N m` per wheel |
| Steering actuator torque limit | `200 N m` per wheel |

Initial inertias may be inferred from simple box and cylinder geometry. Mass,
inertia, contact, damping, and actuator gains may be tuned for stable simulation
without changing the shared vehicle contract. The current resolved tuning is
versioned in the `MujocoTuning` dataclass and recorded in research decision
D-007, because schema version 1 does not contain those fields. Before a reported
experiment, the complete tuning must be copied into its manifest or moved into
a new machine-readable configuration schema; a vehicle YAML fingerprint alone
does not identify it.

### 12.4 Collision geometry

The implemented model uses simple box collision geometry for the tractor and
trailer and cylinders for wheels. Decorative meshes are visual-only. The
planar projection of the chassis collision boxes matches Section 5. Wheel
contact uses a thin cylinder on each wheel center plane to avoid rigid-rim scrub;
a separate full-width cylinder carries wheel mass and visual width.

Dock collision implements the planar wall rule: collision volumes retain the
configured wall outline and are vertically extruded above every chassis. A
separate visual geom retains the configured wall height. This prevents a body
from passing over a low wall while keeping 2D and 3D footprint collision
semantics comparable.

## 13. Cross-simulator parity checks

Before either controller is transferred, the following checks must pass:

1. All named semantic sites are present in MuJoCo.
2. A zero pose produces matching 2D and MuJoCo reference-point coordinates.
3. Positive yaw, steering, articulation, and steering-rate signs agree.
4. Straight reverse commands move both backends in world `-x` from yaw zero.
5. The MuJoCo adapter reports the same virtual steering angle used by 2D.
6. Trailer axle and docking-point transforms match the frozen dimensions.
7. Both backends apply controller commands at `10 Hz`.
8. Both backends trigger the same soft and hard articulation thresholds.
9. Both backends evaluate docking error at `docking_point`.
10. A low-speed deterministic command sequence produces qualitatively matching
    planar trajectories before PPO or NMPC integration begins.

Exact trajectory equality is not required because MuJoCo includes contact and
actuator dynamics. Coordinate, geometry, command, and event equality is
required.

Items 1--7 are covered by automated tests. Direct project-backend trajectory
comparison for item 10 now covers low-speed forward and reverse command
sequences within documented tolerances. Full equality for safety, collision,
and docking events remains pending until the shared task modules and Gymnasium
environment exist. The evidence and remaining gates are tracked in the
[MuJoCo alignment audit](mujoco-alignment-audit.md).

## 14. Change control

The following are frozen cross-team interfaces:

- coordinate and sign conventions
- semantic frame definitions and required MuJoCo names
- nominal planar dimensions
- articulation definition and safety thresholds
- controller command fields and units
- controller frequency
- docking-point and success definitions

Changing one of these requires updating this specification, both YAML files,
2D tests, MuJoCo parity tests, and the specification ID.

The following may be tuned independently without an interface change:

- MuJoCo mass distribution and inertias
- contact parameters
- joint damping and armature
- actuator gains within the frozen command and physical-limit contract
- visual meshes, colors, lights, cameras, and textures
- PPO and NMPC hyperparameters

