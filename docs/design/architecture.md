# Repository architecture

The project is organized around a simulator-independent control loop. Vehicle
state, reference-path tracking, observations, safety checks, controllers, and
evaluation code are shared. Only state propagation and actuator/state adapters
differ between the lightweight 2D model and MuJoCo.

```text
reference path ──> observation builder ──> controller
       ^                                      │
       │                                      v
vehicle state <── simulator adapter <── safety supervisor
```

## Directory map

```text
configs/                         Versioned experiment configuration
docs/design/                     Architecture and mathematical specifications
scripts/                         Thin CLI entry points for training/evaluation
src/trailer_docking/
├── core/                        Shared state, action, geometry, and configuration types
├── paths/                       Reference paths, generators, and path-relative tracking
├── observations/                Controller-facing observation construction
├── safety/                      Collision, jackknife, limits, and command supervision
├── simulators/
│   ├── base.py                  Common simulator protocol
│   ├── kinematic/               Fast deterministic 2D state propagation
│   └── mujoco/                  MuJoCo adapter, XML models, scenes, and assets
├── environments/                Gymnasium API wrapper around the 2D stack
├── controllers/                 Geometric, NMPC, and PPO controller adapters
└── evaluation/                  Scenarios, metrics, rollout runner, and reports
tests/
├── unit/                        Mathematics and geometry tests
└── integration/                 Environment, controller, and simulator-contract tests
```

## Dependency direction

- `core` has no dependency on Gymnasium, Stable-Baselines3, CasADi, or MuJoCo.
- `paths`, `observations`, and `safety` operate on shared core types.
- simulator implementations conform to the protocol in `simulators/base.py`.
- controllers conform to the protocol in `controllers/base.py`.
- the Gymnasium environment composes the 2D simulator and shared modules; it
  does not own a separate vehicle model.
- evaluation drives controllers and simulators through their common protocols.
- MuJoCo XML, meshes, and scenes remain isolated from the 2D implementation.

This boundary is what allows a PPO policy or NMPC controller developed in 2D
to be evaluated against MuJoCo without rewriting its observation or evaluation
logic.

## Implementation order

1. Freeze state, action, coordinate, geometry, timing, and tolerance definitions
   in the [vehicle and environment specification](vehicle-and-environment-specification.md).
2. Implement and test the 2D kinematic simulator.
3. Implement paths, geometry, collision, jackknife, and preview calculations.
4. Compose those modules into the Gymnasium environment.
5. Add a simple controller, then NMPC and PPO.
6. Implement the MuJoCo model and validate simulator-interface parity.
7. Run shared held-out scenarios through the evaluation runner.
