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
│   ├── kinematic/               Implemented deterministic 2D state propagation
│   └── mujoco/                  Implemented adapter, generator, MJCF, and assets
├── environments/                Gymnasium task over the 2D or MuJoCo plant
├── training/                    Scenario environment, PPO runs and campaigns,
│                                docking tests, policy bundles, method switch, CLI
├── ui/                          Desktop control UI (parameters, training, tests)
├── controllers/                 Geometric, NMPC, and PPO controller adapters
├── evaluation/                  Scenarios, metrics, rollout runner, and reports
└── visualization/               Local 3D and top-down viewers of the shared state
tests/
├── unit/                        Mathematics and geometry tests
└── integration/                 Environment, controller, and simulator-contract tests
```

## Dependency direction

- `core` has no dependency on Gymnasium, Stable-Baselines3, CasADi, or MuJoCo.
- `paths`, `observations`, and `safety` operate on shared core types.
- simulator implementations conform to the runtime-checkable protocol in
  `simulators/base.py`.
- the Gymnasium environment composes a simulator backend (2D by default,
  MuJoCo on request) and the shared modules; it does not own a vehicle model.
- `training/evaluate.py` runs a policy or a model-based controller on that
  environment and scores the episodes. A model-based controller enters through
  the single hook described in the
  [NMPC integration handoff](nmpc-integration-handoff.md).
- `controllers/base.py`, `controllers/geometric.py`, `controllers/ppo.py`,
  `evaluation/runner.py`, and `evaluation/metrics.py` are placeholders with no
  code yet.
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
6. Implement the MuJoCo model and validate its side of the simulator contract.
7. Run shared held-out scenarios through the evaluation runner.

Steps 1--4 and the MuJoCo portion of step 6 are implemented. Direct low-speed
trajectory parity is tested for forward and reverse maneuvers. For step 5, the
PPO training framework exists and runs on either simulator, but no policy has
been tuned or evaluated; the simple tracker and NMPC are not implemented. The
task runs on MuJoCo, but agreement of its episode events with 2D has not been
validated, and step 7 has not started.
