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

Which differences between the verified 2D kinematic model and MuJoCo—actuator
response, wheel contact, slip, hitch behavior, or numerical timing—most affect
zero-shot controller transfer? Investigate only after coordinate, geometry,
command, and event parity checks pass.

## Q-004 — Limit checks against a physical plant

`state_limit_violations` compares measured state with a `1e-12` tolerance,
which suits the exact 2D plant. In MuJoCo the measured virtual steering angle
overshoots the `0.50 rad` limit by about `0.001 rad` at full lock, so an
environment that terminates on that check would end MuJoCo episodes the 2D
environment would not. Decide whether measured-state checks get a documented
tolerance or whether only commanded quantities are checked.

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
