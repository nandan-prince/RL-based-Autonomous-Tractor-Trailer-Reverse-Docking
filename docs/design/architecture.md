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
│                                curricula, comparisons, bundles, method switch, CLI
├── ui/                          Desktop control UI (parameters, training, tests)
├── controllers/                 Geometric, NMPC, and PPO controller adapters
├── evaluation/                  Scenarios, resolved episodes, metrics, runners
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
- `evaluation/episodes.py` resolves every random task choice once. The
  comparison runner replays the same fingerprinted episode objects across
  controllers and 2D/3D backends.
- strict controller settings, experiment suites, and curricula live under
  `configs/`; formal comparisons serialize their resolved inputs and compact
  outputs through `training/comparison.py`.
- `controllers/base.py` defines the shared inference contract; implemented LQR
  and NMPC controllers use it. PPO training produces an observation-only policy
  that is evaluated through the same normalized action boundary.
- MuJoCo XML, meshes, and scenes remain isolated from the 2D implementation.

This boundary is what allows a PPO policy or NMPC controller developed in 2D
to be evaluated against MuJoCo without rewriting its observation or evaluation
logic.

The exact controller handoff—observation access, normalized steering-rate
action, fixed reverse speed, path ordering, reset pairing, and PPO restrictions—is
defined in the [shared controller interface](controller-interface.md).

## Implementation order

1. Freeze state, action, coordinate, geometry, timing, and tolerance definitions
   in the [vehicle and environment specification](vehicle-and-environment-specification.md).
2. Implement and test the 2D kinematic simulator.
3. Implement paths, geometry, collision, jackknife, and preview calculations.
4. Compose those modules into the Gymnasium environment.
5. Add a simple controller, then NMPC and PPO.
6. Implement the MuJoCo model and validate its side of the simulator contract.
7. Run shared held-out scenarios through the evaluation runner.

Steps 1--4, the straight-path LQR and basic NMPC portion of step 5, the PPO
training, gated straight/curved curriculum, controlled reward-study
infrastructure, formal paired-evidence writer, and the MuJoCo portion of step 6
are implemented.
Direct low-speed trajectory parity is tested for forward and reverse maneuvers,
and the 2D task has verified collision, jackknife, docking, boundary, and
timeout events. No PPO policy has been tuned or registered as an experiment,
and the controllers have not undergone curved/randomized paired evaluation.
MuJoCo task-event parity remains a transfer gate. See the
[MuJoCo alignment audit](mujoco-alignment-audit.md) for current status.
