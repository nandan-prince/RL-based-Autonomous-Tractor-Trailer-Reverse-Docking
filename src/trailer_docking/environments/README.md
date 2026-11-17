# Gymnasium docking environment

`TrailerDockingEnv(backend="3d")` runs the same task on the MuJoCo plant:
observation, reward, and events are computed from the shared planar state in
both cases, but agreement of episode events between the two plants has not
been validated. Everything below describes the default 2D backend.

`TrailerDockingEnv` composes the verified kinematic plant, reference path,
path-relative observation, dock collision geometry, jackknife detection,
docking success, reward, timeout, and rendering. It is registered as
`TrailerDocking-v0` when `trailer_docking.environments` is imported.

## Interactive keyboard control

```bash
.venv/bin/python -m trailer_docking.environments.viewer
```

The viewer defaults to the full-size service-center scenario: a 5.8 m tractor,
14.0 m semitrailer, a 200 m by 100 m paved yard, and 20 building doors along
the bottom x-axis. The active 18 m loading zone is indicated by its floor
color; it has no physical or collidable side walls. The building/rear boundary
remains collision geometry. Manual limits are 2 m/s forward and 1 m/s reverse.
The service-center terminal jackknife threshold is 60 degrees, with the soft
warning retained at 35 degrees.
Use `--scenario compact` to display the compact research baseline instead.

Use Up/Down for forward/reverse, Left/Right to change the desired steering
angle, Space to stop, and R or Backspace to reset. Add `--demo` for the tested
straight reverse that reaches docking success, `--random-start` for a seeded
offset, or `--variant short|nominal|long` to change trailer length.

The registered `TrailerDocking-v0` still defaults to the compact baseline so
adding the visual service-center scenario does not silently change controller
experiments. Pass the three service-center configuration paths explicitly when
an algorithm experiment is intended to use the full-size scenario.

## Gymnasium API

```python
import gymnasium as gym
import trailer_docking.environments  # registers TrailerDocking-v0

env = gym.make("TrailerDocking-v0")
observation, info = env.reset(seed=0)
observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

The default `fixed_reverse` mode has one normalized steering-rate action and
uses the frozen `-0.2 m/s` target. This is the PPO/NMPC comparison contract.
The viewer uses `action_mode="teleop"`, whose second normalized action controls
speed for manual debugging. Both modes use the same plant and event logic.

The 21-element float32 observation is bounded to `[-1, 1]` and contains path
lateral error, sine/cosine heading errors, articulation and rate, steering
state, speed, remaining distance, goal-pose error, and three local preview
points. `info` reports every reward component, applied command, collision
contacts, articulation state, trailer containment, docking errors, progress,
success, and termination reason.

Episodes terminate on docking success, collision, jackknife, or leaving the
workspace. Reaching the step limit is a Gymnasium truncation. Event logic is
independent of reward. Settings are versioned in
[`gymnasium_docking_v1.yaml`](../../../../configs/environments/gymnasium_docking_v1.yaml).

## Verification

```bash
.venv/bin/pytest tests/unit/paths tests/unit/safety -q
.venv/bin/pytest tests/unit/environments tests/unit/observations -q
.venv/bin/pytest tests/integration/gymnasium -q
```

The suite runs Gymnasium's checker, seeded reset comparisons, fixed and
teleoperation actions, every terminal outcome, invalid actions, every trailer
variant, 1,000 random steps, RGB rendering, and a complete straight reverse to
collision-free docking success.
