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
