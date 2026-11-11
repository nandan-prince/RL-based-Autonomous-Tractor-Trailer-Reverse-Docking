# Tests

Unit tests will additionally verify dynamics equations, timestep convergence,
collision, reference tracking, and deterministic integration. Integration
tests will verify Gymnasium compliance, controller/simulator contracts,
reproducible seeding, and 2D/MuJoCo state-action parity.

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
first model, plus dock geometry, wall contact, and agreement with a test-local
integration of the specification's kinematic equations.

MuJoCo tests should live under `tests/integration/mujoco/` and skip cleanly when
the optional dependency is absent. Once the model exists, tests must compile
the MJCF, resolve every configured name, check neutral semantic-site positions,
verify sign conventions and command timing, and exercise deterministic
straight-reverse and steering smoke cases. A viewer screenshot is useful for
debugging but is not a substitute for numeric assertions.
