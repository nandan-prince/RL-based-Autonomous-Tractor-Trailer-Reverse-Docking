# Canonical 2D kinematic backend

This package is the deterministic planar plant used before the Gymnasium task
and controllers are composed. It reads the same typed vehicle configuration as
MuJoCo and returns the shared `VehicleState`; it does not define observations,
rewards, collisions, docking success, or episode termination.

`PlanarKinematicState` integrates tractor rear-axle position/yaw, trailer yaw,
virtual steering angle, longitudinal speed, and simulation time. The public
command remains `VehicleCommand(steering_rate, target_speed)`. The shared
supervisor applies the frozen steering, speed, acceleration, and timing limits.

`KinematicSimulator.step()` advances one `0.10 s` controller period using five
fixed `0.02 s` classical RK4 substeps. Speed accelerates toward the supervised
target; steering rate is held over the command period. `KinematicModel`
includes signed longitudinal and lateral hitch offsets. Only independent
coordinates are integrated; `VehicleGeometry` reconstructs the hitch and
trailer state so the rigid connection cannot accumulate numerical drift.

The frozen equations and parameters are in
[`docs/design/vehicle-and-environment-specification.md`](../../../../docs/design/vehicle-and-environment-specification.md).
Do not introduce separate geometry constants here.

## Verification

```bash
.venv/bin/pytest tests/unit/simulators/kinematic -q
.venv/bin/pytest tests/integration/test_kinematic_mujoco_parity.py -q
.venv/bin/ruff check src tests
.venv/bin/mypy
```

The tests cover equations, forward/reverse and constant-steering motion,
articulation, wrapping, rigid reconstruction, convergence, limits,
determinism, variants, invalid input, and direct MuJoCo trajectory parity.
Task-event parity awaits shared collision, jackknife, and docking logic.
