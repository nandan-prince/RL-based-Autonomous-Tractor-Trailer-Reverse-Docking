# Tests

Unit tests verify the 2D dynamics equations, timestep convergence, rigid
geometry reconstruction, deterministic integration, forward/reverse motion,
steering and acceleration limits, angle wrapping, articulation behavior, and
invalid-input rejection. Unit suites now also verify reference interpolation,
signed tracking errors, previews, polygon collision, dock containment,
jackknife thresholds, goal evaluation, scenario construction, and normalized
observations. Integration tests verify direct low-speed 2D/MuJoCo trajectory
parity and the complete Gymnasium API and 2D task behavior.

The current vehicle-core suite verifies strict schema handling, invalid-input
rejection, golden neutral geometry, coordinate-transform round trips, footprint
invariants, trailer variants, off-axle geometry, serialization, configuration
fingerprints, command supervision, dependency boundaries, and the executable
vehicle/dock conformance report. Run it with:

```bash
.venv/bin/ruff check src tests
.venv/bin/mypy
.venv/bin/pytest --cov --cov-report=term-missing
.venv/bin/python -m trailer_docking.validation.vehicle_spec
```

The MuJoCo suite under `tests/integration/mujoco/` covers those gates for the
first model, plus dock geometry, wall contact, plant limits, reset behavior,
trailer variants, measured-state tolerances, and agreement with a test-local
integration of the specification's kinematic equations.

The canonical 2D tests live under `tests/unit/simulators/kinematic/`. The
cross-backend command-sequence check is
`tests/integration/test_kinematic_mujoco_parity.py` and skips cleanly if MuJoCo
is unavailable.

The environment suite under `tests/integration/gymnasium/` runs Gymnasium's
checker, reproducible seeded resets, both action modes, success/collision/
jackknife/timeout events, invalid-action atomicity, all trailer variants, long
random rollouts, RGB rendering, and a full collision-free straight docking.
The service-center acceptance tests additionally validate the full-size
dimensions, exact 200 m by 100 m rendered workspace, 20-door layout, absence of
loading-zone side collision walls, collidable building façade, maneuvering
motion, yard rendering, the 60-degree terminal jackknife boundary, rigid
dock-to-yard rendering of an articulated hitch, and full-size collision-free
docking.

MuJoCo tests live under `tests/integration/mujoco/` and skip cleanly when the
optional dependency is absent, and the training tests that need
Stable-Baselines3 skip without the `train` extra. The full verification
command should install `.[dev,mujoco,viz,train]`; otherwise a passing run may
omit 3D and training tests. Tests compile
the MJCF, resolve every configured name, check neutral semantic-site positions,
verify sign conventions and command timing, and exercise deterministic
straight-reverse and steering cases. A viewer screenshot is useful for
debugging but is not a substitute for numeric assertions.

The training-framework suites cover scenario overrides and their rejection
rules (`tests/unit/evaluation/`), the task on both simulators with domain
randomization (`tests/integration/test_backends_and_randomization.py`),
campaigns, the PPO/NMPC method switch, policy bundles, and safe policy loading
(`tests/unit/training/`), and the control-UI logic (`tests/unit/ui/`). The
Tkinter window itself is exercised only by its `--smoke-test` option and by
hand.
