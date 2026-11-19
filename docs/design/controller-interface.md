# Shared controller interface

This document is the controller-integration contract for the deterministic
tracker, NMPC, and PPO. It prevents independently developed controllers from
silently solving different tasks.

## Frozen primary comparison conventions

- Environment: `TrailerDocking-v0`, using the compact vehicle, nominal dock,
  and `gymnasium-docking-v1` task unless an experiment explicitly selects
  another configuration.
- Controller period: `0.10 s`.
- Speed: the vehicle configuration's fixed nominal reverse target
  (`-0.20 m/s` for the compact baseline).
- Action: one `numpy.float32` value with shape `(1,)`, bounded to `[-1, 1]`.
- Action meaning: normalized steering rate. The physical command is
  `action[0] * steering_rate_abs_max`; it is not a steering angle.
- Positive steering and steering rate turn left under the frozen body/world
  sign convention.
- Reference path: ordered from the dock target outward toward the start.
  Reverse docking moves toward decreasing arc length and succeeds near
  progress zero.
- Tracking point: the center of the trailer rear face (the docking point), not
  the tractor pose or trailer axle.
- Episode events: success, collision, jackknife, workspace exit, and timeout
  are evaluated by the environment, independently of controller cost/reward.
- Reset/evaluation: methods must use the same explicit reference path, initial
  tractor pose, trailer heading, trailer variant, configuration fingerprints,
  and seed for a paired comparison.

The environment accepts an explicit `ReferencePath` either at construction or
through `reset(options={"reference_path": path})`. This path is independent of
the randomized initial state, which is essential for measuring recovery from
path errors.

## Runtime controller protocol

`ControllerContext` contains the normalized observation, shared physical state,
reference path, path-relative tracking state, vehicle configuration, and step
index. Every inference adapter returns the same normalized one-element action.

- LQR and NMPC may use the shared state, path, model, and tracking fields.
- The primary PPO policy must use only the normalized Gymnasium observation.
  Feeding privileged context fields to PPO changes the observation contract and
  must be a separately named experiment.
- Stable-Baselines3 can train directly on `TrailerDockingEnv`. Its deployment
  adapter should implement the shared `Controller` protocol so evaluation uses
  the same runner as LQR and NMPC.

The normalized observation schema and stable element names are defined by
`ObservationBuilder`. Do not reorder or independently reconstruct them inside
the PPO code. The action space exposed by the environment is the authority for
shape, dtype, and bounds.

## Implemented baseline scope

`TrajectoryLQRController` numerically linearizes the verified RK4 kinematic
plant about straight reverse motion. Its error state is trailer docking-point
lateral error, tractor heading error, trailer heading error, and steering
angle. A discrete Riccati solution produces a physical steering-rate command.

`BasicNMPCController` solves a 40-step nonlinear receding-horizon problem with
the configured fixed reverse speed and steering rate as the only decision
input. It tracks sampled docking-point position and trailer heading while
penalizing tractor heading, articulation, steering angle, and steering rate.
It enforces steering, steering-rate, and articulation limits. Collision is
still evaluated by the environment but is not yet an NMPC prediction
constraint.

These controllers are first vertical-slice baselines. Their successful straight
tracking checks do not establish robustness, optimality, or comparative
performance.

## PPO integration status

The merged PPO infrastructure now:

1. Uses `action_mode="fixed_reverse"` for the primary policy.
2. Produces a `(1,)` `float32` action in `[-1, 1]`.
3. Does not independently control speed in the primary comparison.
4. Uses the existing observation without reordering fields.
5. Uses seeded Gymnasium reset and records all three configuration fingerprints.
6. Can save, load, and run deterministically for evaluation.
7. Uses `PPOController`, a thin observation-only adapter implementing the
   shared `Controller` protocol.
8. Is evaluated on the same explicit paths and reset options as LQR and NMPC.
9. Keeps training-only wrappers, normalization statistics, and reward changes
   versioned and available during evaluation.
10. Keeps optional recording in the evaluation layer rather than inside the
    policy/controller implementation.
