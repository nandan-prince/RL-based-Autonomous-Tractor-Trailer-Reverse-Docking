# Tests

Unit tests verify the 2D dynamics equations, timestep convergence, rigid
geometry reconstruction, deterministic integration, forward/reverse motion,
steering and acceleration limits, angle wrapping, articulation behavior, and
invalid-input rejection. Future tests will add collision and reference
tracking. Integration tests currently verify direct low-speed 2D/MuJoCo
trajectory parity; Gymnasium compliance and event parity remain planned.

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

MuJoCo tests live under `tests/integration/mujoco/` and skip cleanly when the
optional dependency is absent. The full verification command should install
`.[dev,mujoco,viz]`; otherwise a passing run may omit 3D tests. Tests compile
the MJCF, resolve every configured name, check neutral semantic-site positions,
verify sign conventions and command timing, and exercise deterministic
straight-reverse and steering cases. A viewer screenshot is useful for
debugging but is not a substitute for numeric assertions.
