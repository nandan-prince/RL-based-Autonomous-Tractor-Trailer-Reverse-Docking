# NMPC integration handoff

**Audience:** whoever implements the NMPC baseline (Nandan, and any AI
assistant working with him). Read this before writing
`src/trailer_docking/controllers/nmpc.py`.

**Status on 2026-11-13:** the runner, command line, and control UI can already
select NMPC. The controller itself does not exist yet, so selecting it reports
"NMPC is not implemented yet". Nothing else needs to change for it to work:
provide one function and it runs everywhere PPO runs.

## What you provide

One function in `src/trailer_docking/controllers/nmpc.py`:

```python
def make_docking_controller(env) -> Callable[[np.ndarray], np.ndarray]:
    ...
```

- It is called **once at the start of every episode**, immediately after the
  environment has been reset. Build or reset the solver here (warm start,
  horizon, reference for this episode).
- It returns a callable that is called **once per control step** with the
  21-element policy observation. NMPC may ignore the observation and read the
  shared state from `env` instead.
- The callable returns **one normalized steering-rate action**: shape `(1,)`,
  value in `[-1, 1]`. Any array-like of one number is accepted. The
  environment multiplies it by `limits.steering_rate_abs_max`.

NMPC is not trained. It solves its optimization online at every step, so
there is no training phase: with NMPC selected, **Start** simply runs the
episodes.

## What `env` gives you

`env` is a `trailer_docking.training.environment.ScenarioDockingEnv`.

| You need | Where it is |
|---|---|
| Current shared vehicle state | `env.state` (`VehicleState`: tractor and trailer pose, hitch position, steering angle and rate, speed, yaw rates, articulation angle, simulation time) |
| Reference path for this episode | `env.reference_path` (`ReferencePath`: docking-point polyline ordered **goal to start**, with `length`, `sample(progress)`, `nearest(point)`, `previews(...)`) |
| Vehicle geometry and limits | `env.built.vehicle` (`.geometry`, `.limits`, `.timing`, `.command`) |
| Dock geometry, target, tolerances | `env.built.dock.geometry` |
| Task settings (previews, reward weights) | `env.built.task` |
| Step limit actually applied to this run | `env.task_env.settings.max_episode_steps` (the runner overrides the task file's 600 with 900 by default) |
| Path-relative tracking the task computed | `env.task_env.tracking` |
| Which simulator is running | `env.backend` (`"2d"` kinematic or `"3d"` MuJoCo) |

Read these each step; do not cache `env.state`, because a new object is
produced every step. With trailer-length randomization enabled the vehicle is
rebuilt at each reset, which is why the hook runs per episode: read
`env.built.vehicle` inside it, not at import time.

## The plant contract you are controlling

These are fixed by decision D-002 and the frozen specification, and they are
the same for PPO, so the comparison stays fair:

- One control input: bounded steering **rate**. Steering angle is state.
- Reverse speed is fixed at `vehicle.command.nominal_target_speed`
  (`-0.2 m/s` nominal). The controller does not choose speed in the primary
  comparison. If NMPC is later given speed authority, that is a separately
  named experiment.
- Control period `vehicle.timing.controller_period` (`0.10 s`).
- Limits: `limits.steering_angle_abs_max`, `limits.steering_rate_abs_max`,
  `limits.articulation_soft_abs_max`, `limits.articulation_hard_abs_max`.
- The prediction model should be the specification's kinematic equations
  (section 10), implemented in
  `trailer_docking.simulators.kinematic.KinematicModel`. Reuse it rather than
  re-deriving constants. In 3D the same controller then runs against MuJoCo,
  which is the model-mismatch case the proposal asks about.
- Success, collision, jackknife, out-of-bounds, and time-limit events are
  evaluated by the task, not by the controller. Do not add a second success
  rule.

## How to run it once the hook exists

Command line:

```bash
# 20 episodes in 2D, then the same seeds in 3D
.venv/bin/python -m trailer_docking.training.cli test --nmpc --episodes 20
.venv/bin/python -m trailer_docking.training.cli test --nmpc --episodes 20 --backend 3d

# full evaluation run with saved records (no training happens)
.venv/bin/python -m trailer_docking.training.cli train --method nmpc \
    --eval-episodes 100 --eval-backends 2d,3d
```

Control UI (`python -m trailer_docking.ui.app`): on the **Training** tab set
**Method** to `nmpc` and press **Start**, or on the **Docking test** tab choose
**NMPC controller**. Scenario parameters (truck dimensions, dock size, start
ranges, limits) apply to NMPC exactly as they do to PPO.

A run with `--method nmpc` writes `results/<id>/manifest.yaml` and
`summary.json` with success, collision and jackknife rates, final errors,
maximum articulation, maneuver time, path length, steering-rate RMS, and mean
controller time per step. Controller time is measured around your callable, so
it is the online compute cost the proposal lists as a metric.

For a like-for-like comparison, test PPO and NMPC with the same scenario file
and the same `--seed`; episode `i` uses seed `seed + i` in both.

## Minimal skeleton

```python
"""CasADi/IPOPT nonlinear model-predictive controller."""

import numpy as np


def make_docking_controller(env):
    vehicle = env.built.vehicle
    path = env.reference_path
    rate_limit = vehicle.limits.steering_rate_abs_max
    solver = ...  # build or warm-start the NLP for this episode

    def act(observation):
        state = env.state
        steering_rate = ...  # solve, in rad/s
        return np.array([np.clip(steering_rate / rate_limit, -1.0, 1.0)], np.float32)

    return act
```

## Tests that already cover the seam

`tests/unit/training/test_campaign_and_bundle.py`:

- `test_nmpc_is_reported_as_not_implemented_until_the_hook_exists` will need
  its first assertion removed once the hook exists; that is expected.
- `test_a_controller_factory_is_rebuilt_from_the_environment_each_episode`
  shows the calling convention with a stand-in controller.
- `test_nmpc_campaign_skips_training_and_only_runs_the_controller` shows what
  an NMPC run records.

Add NMPC's own tests (solver feasibility, constraint handling, timing) next to
the controller.

## Please do not change without talking first

- The hook name and signature, or `training/methods.py`.
- The action contract (one normalized steering rate) and the fixed speed.
- `TrailerDockingEnv` events, observation, or reward.

CasADi is not yet a declared dependency. Add it as its own optional extra in
`pyproject.toml` (for example `nmpc = ["casadi>=3.6"]`) and import it inside
`make_docking_controller`, so the package still installs and the 2D tests
still run without it.

## Background an assistant should have before starting

### Where the code is

This framework is on branch `chris/rl-ui-reconcile` and reaches `main` through
its pull request. Until that is merged, `main` has the 2D task but not
`training/`, `ui/`, or this document. Branch NMPC work from a `main` that
contains `src/trailer_docking/training/methods.py`; if it does not exist yet,
wait for the merge or branch from `chris/rl-ui-reconcile`.

| Path | What it is |
|---|---|
| `AGENTS.md` | Project rules: scope, fairness, documentation, experiment records. Read first |
| `docs/design/vehicle-and-environment-specification.md` | Frozen conventions, geometry, state and command contracts, limits, timing, section 10 kinematic equations, section 11.1 docking success |
| `configs/vehicles/compact_semitrailer_v1.yaml`, `configs/environments/nominal_dock_v1.yaml`, `configs/environments/gymnasium_docking_v1.yaml` | Frozen vehicle, dock, and task values. Never edit; override through a `Scenario` |
| `src/trailer_docking/core/` | `VehicleState`, `VehicleCommand`, `Pose2D`, `Point2D`, geometry, strict config loading |
| `src/trailer_docking/simulators/kinematic/model.py` | `KinematicModel(geometry)` with `derivative(state, control)` and `rk4_step(...)` over `PlanarKinematicState` and `KinematicInput` |
| `src/trailer_docking/environments/docking_env.py`, `task.py` | The Gymnasium task: observation, reward, events. `evaluate_task` is the single success and failure definition |
| `src/trailer_docking/paths/` | `ReferencePath`, `docking_reference(...)`, `tracking_state(...)` |
| `src/trailer_docking/safety/` | Footprint collision, jackknife classification, command supervisor |
| `src/trailer_docking/training/` | Scenario environment, test runner, campaigns, bundles, method switch, CLI |
| `src/trailer_docking/controllers/nmpc.py` | **Yours.** Currently a one-line stub |
| `research/papers/nonlinear-mpc-local-planner.pdf` | Moradi and Mehrandezh (2022), the NMPC reference the proposal names |
| `research/decisions.md` | D-002 (control authority), D-009 (2D plant), D-010 (task), D-013 (this framework) |
| `research/open_questions.md` | Q-002 steering-rate sensitivity, Q-004 limit checks against MuJoCo, Q-006 long trailer and bay depth |

### Conventions that are easy to get wrong

- SI units, radians. `+x` forward, `+y` left, yaw counterclockwise.
- The dock frame has its origin at the rear-wall centre with `+x` pointing
  **out of the bay**. Reversing into the bay moves toward decreasing `x`.
- Reverse speed is **negative**: the nominal target is `-0.2 m/s`.
- Articulation is `wrap(tractor_yaw - trailer_yaw)`.
- Positive steering turns left when driving forward. In reverse the trailer
  swings the opposite way to the steering input, and the open-loop system is
  unstable, which is the whole difficulty.
- The reference path is the route of the **docking point** (trailer rear-face
  centre), not the trailer axle and not the tractor. It is ordered goal to
  start, so progress along it **decreases** as the maneuver proceeds, and the
  path tangent points outward (the direction the vehicle faces, not the
  direction it moves).
- The nominal hitch is on the tractor rear axle, but both signed hitch offsets
  are configurable and the kinematic model includes them. Do not hard-code the
  on-axle case; use `vehicle.geometry`.
- The hard jackknife threshold (`articulation_hard_abs_max`, 45 degrees
  nominal) ends the episode. The soft threshold (35 degrees) is a warning the
  controller can use as its own constraint.

### Behaviour of the runner you should know

- The environment **rejects** an action outside `[-1, 1]` or with a non-finite
  value by raising `ValueError`; it does not clip. Clip in the controller.
- The steering angle saturates at `steering_angle_abs_max` inside the plant.
  The command supervisor clips the rate so the angle cannot leave its limit,
  so the applied rate can be smaller than the commanded one. The rate actually
  applied in the last step is `env.task_env.simulator.applied_command`.
- The plant ramps speed from rest at `longitudinal_acceleration_abs_max`
  (`0.4 m/s^2`), so the first half second is slower than the target. Use the
  measured `state.longitudinal_speed` in the prediction, not the target.
- An episode starts at rest with zero steering angle at a pose sampled from the
  scenario's start distribution. Default ranges are 6 to 9 m standoff,
  plus or minus 0.4 m lateral, 0.1 rad trailer yaw, 0.05 rad articulation.
- The step limit of a test is 900 control steps by default (90 s simulated).
- A scenario is rejected up front if its trailer cannot be completely inside
  its bay or if an extreme of its start ranges begins already collided, out of
  bounds, or jackknifed. If a scenario builds, every start is controllable.
- The reference path comes from the scenario's planner: `entry` (default,
  at most three points) or `astar` (a polyline with a few more corners). Both
  are piecewise linear with heading discontinuities at the corners, and
  neither is checked for trailer feasibility, so treat the path as a guide to
  track, not a trajectory the vehicle can follow exactly.
- Each episode's `info` includes `scenario_fingerprint`, which covers every
  override and range; the task's own `vehicle_fingerprint` does not.
- Simulated time does not wait for the controller. A slow solver does not
  cause a missed deadline in simulation; its time is only measured and
  reported. The real-time budget to compare against is the 0.10 s period.
- In 3D the plant is MuJoCo, not the prediction model. Measured steering rate
  and speed overshoot their commanded limits slightly (open question Q-004),
  and there is tire and actuator lag. This is the intended model-mismatch
  test, but 3D event agreement with 2D is not yet validated.
- Everything is deterministic for a given seed in 2D. Keep the controller
  deterministic too (fixed solver options, no wall-clock-dependent logic) so
  runs can be reproduced.

### Working on it day to day

Setup and the gates every pull request must pass (on Windows create the
environment with `py -3.12 -m venv .venv` and replace `.venv/bin/` with
`.venv\Scripts\` in every command in this document):

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev,mujoco,viz,train]'
.venv/bin/ruff check src tests
.venv/bin/mypy
.venv/bin/pytest --cov --cov-report=term-missing
.venv/bin/python -m trailer_docking.validation.vehicle_spec
.venv/bin/python -m trailer_docking.simulators.mujoco.builder
```

`mypy` runs in strict mode on the listed packages; add
`src/trailer_docking/controllers` to `[tool.mypy] files` and
`trailer_docking.controllers` to the coverage sources in `pyproject.toml` when
the controller has real code, and add `casadi` to the mypy
`ignore_missing_imports` override.

The fastest feedback loop is a script that builds the environment directly:

```python
import numpy as np
from trailer_docking.controllers.nmpc import make_docking_controller
from trailer_docking.evaluation.scenarios import Scenario, StartDistribution
from trailer_docking.training.environment import ScenarioDockingEnv

scenario = Scenario(start=StartDistribution((4, 4), (0.3, 0.3), (0.05, 0.05), (0, 0)))
env = ScenarioDockingEnv(scenario, max_episode_steps=900)
observation, info = env.reset(seed=0)
act = make_docking_controller(env)
done = False
while not done:
    observation, reward, terminated, truncated, info = env.step(
        np.asarray(act(observation), np.float32).reshape(1)
    )
    done = terminated or truncated
print(info["termination_reason"], info["docking_error_lateral"], info["docking_error_yaw"])
```

`info` carries every reward term, the collision contacts, the articulation
status, and the docking errors each step, which is usually enough to see why
an episode failed. To watch it, open the control UI, pick **NMPC controller**
on the **Docking test** tab, and run one episode in 2D or 3D.

A straight reverse with zero steering is the sanity baseline: it docks from an
aligned start in 486 steps in both simulators and collides from the default
offset starts. A working NMPC should do no worse than it anywhere.

### Rules that apply to results

- Tuning NMPC weights, horizon, and tolerances is expected. Record the final
  values in a versioned file under `configs/`, not only in code, and note the
  tuning effort, because the proposal compares tuning cost as well as runtime.
- Do not tune on the seeds used for the reported comparison. Campaign tests
  start at evaluation seed 100000 by default; pick a different, documented
  seed range for the final held-out comparison and use it for both methods.
- A result is report-ready only when it has an experiment ID with a manifest
  (AGENTS.md section 6.1). `train --method nmpc` writes one. Results written
  during development go under an output root outside the repository or stay
  uncommitted; commit only runs that are meant as evidence.
- Keep failed and infeasible runs. A solver failure is a result to categorize,
  not an episode to drop.
- If NMPC cannot be made reliable in time, the proposal's fallback is a
  geometric or LQR tracker. It plugs into the same hook; say so in the
  decision log and the paper rather than labelling it NMPC.

## Open points to settle together

- Numerical-failure handling: what the controller returns when IPOPT does not
  converge, and whether that is counted as its own failure category.
- Whether NMPC uses the task's polyline reference directly or smooths it.
- The 3D backend's event parity with 2D has not been validated, so 3D numbers
  for either controller are engineering checks until it is.
