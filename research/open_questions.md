# Open research questions

These are questions to investigate after the nominal 2D controllers and shared
evaluation pipeline work. They are not established contributions or mandatory
experiments.

## Q-001 — Geometry generalization

Can one trajectory-relative PPO policy generalize across held-out
hitch-to-trailer-axle distances, or does reliable performance require
geometry-conditioned observations or retraining? Compare this fairly with NMPC
using correct and deliberately mismatched model parameters.

## Q-002 — Steering-rate sensitivity

How sensitive are PPO and NMPC performance, smoothness, and feasibility to the
project-selected `0.3 rad/s` steering-rate limit? This value is not inherited
from the two core papers and should be treated as a controlled sensitivity
factor before drawing broad conclusions.

## Q-003 — Simulation-to-simulation transfer

Direct low-speed forward and reverse trajectory checks now align the verified
2D kinematic plant with MuJoCo within explicit test tolerances, and the 2D task
events are implemented. Which remaining
differences—actuator response, wheel contact, slip, hitch behavior, or
numerical timing—most affect zero-shot controller transfer? Investigate only
after the same collision, safety, docking-event, observation, and controller
interfaces are applied to MuJoCo and pass parity checks.

## Q-004 — Limit checks against a physical plant

`state_limit_violations` defaults to a `1e-12` tolerance, which suits the
exact 2D plant. In MuJoCo, with every command inside its limit, measured
steering angle overshoots `0.50 rad` by about `0.001 rad`, measured steering
rate reached `0.349 rad/s` against `0.30`, and measured speed `0.214 m/s`
against `0.20`. The function now accepts a `MeasurementTolerance`, and the
MuJoCo backend publishes `MUJOCO_MEASUREMENT_TOLERANCE`. Still to decide: the
environment-level rule, that is whether termination uses measured state with
a per-backend tolerance or only commanded quantities, so that 2D and MuJoCo
episodes end under the same conditions.

## Q-007 — Steered front wheels outside the collision footprint

Collision is defined on the chassis footprints in both simulators. At full
steering lock the inner front wheel of the nominal vehicle extends about
`0.11 m` beyond the body side, so that corner can overlap a dock wall with no
collision reported. It only matters when the tractor itself is beside or
inside the bay. Decide whether to accept this, widen the tractor footprint, or
add the steered wheels to the planar collision test.

## Q-005 — Rigid pitch coupling at the hitch

The version-1 hitch is yaw-only and there is no suspension, so tractor and
trailer form one body in pitch resting on three axles. Axle loads are then set
by contact compliance rather than by a fifth-wheel load path; a static estimate
puts roughly 30 % of total weight on the front axle instead of only its share
of the tractor. This did not visibly affect the planar kinematics at
`0.2 m/s`, but it changes steering and traction loads and should be revisited
before any dynamics-sensitive transfer claim.

## Q-006 — Long trailer and the nominal bay

With the docking point at `x = 0.25 m`, the `long` variant's front face is at
`x = 6.75 m`, outside the `6.0 m` bay, so the "complete trailer inside bay"
success rule cannot be met. The conformance check only evaluates the nominal
variant. Decide on a deeper bay, a variant-specific rule, or dropping the rule
for length-generalization experiments.

Since 2026-11-13 the scenario layer used for training and docking tests
rejects a trailer that cannot meet the rule in its bay, so the `long` variant
requires a scenario with a deeper bay there. The underlying decision for the
frozen task and the conformance check is still open.

## Q-008 — MuJoCo tuning provenance

The vehicle YAML fingerprints geometry, limits, timing, masses, friction, and
actuator force limits, but not `MujocoTuning` gains, damping, armature, contact
width, solver choices, or rear-differential setting. Before reported MuJoCo
experiments, decide whether to introduce a new configuration schema or require
the experiment runner to serialize the complete resolved tuning object into
every manifest. Git revision alone is reproducible but makes controlled tuning
studies and configuration comparison less explicit.
