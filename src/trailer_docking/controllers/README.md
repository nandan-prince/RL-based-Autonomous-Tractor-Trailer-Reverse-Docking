# Controllers

The controller package implements the common interface in
[`docs/design/controller-interface.md`](../../../docs/design/controller-interface.md).

Implemented controllers:

- `TrajectoryLQRController`: a deterministic trajectory-error LQR controller
  derived from a numerical linearization of the verified kinematic backend.
- `BasicNMPCController`: a CasADi/IPOPT nonlinear receding-horizon tracker with
  fixed reverse speed and bounded steering-rate authority.
- `PPOController`: the observation-only adapter used to execute a trained PPO
  policy through the same normalized steering-rate interface.

All three methods use the controller contract in `base.py`. LQR and NMPC are
rebuilt for each resolved episode so they receive that episode's geometry and
reference path. PPO receives the normalized observation and returns the same
one-element action in `[-1, 1]`. Versioned LQR/NMPC settings live in
`configs/controllers/baseline_controllers_v1.yaml`; PPO optimization settings
live in `TrainingConfig` and the run manifest.

Install the NMPC dependency with:

```bash
.venv/bin/pip install -e '.[control]'
```

Run the closed-loop demonstrations interactively:

```bash
.venv/bin/python -m trailer_docking.evaluation.runner --controller lqr
.venv/bin/python -m trailer_docking.evaluation.runner --controller nmpc
.venv/bin/python -m trailer_docking.evaluation.runner --controller lqr --scenario service-center
.venv/bin/python -m trailer_docking.evaluation.runner --controller nmpc --scenario service-center
```

Run a saved PPO checkpoint, or compare methods on identical resolved episodes:

```bash
.venv/bin/python -m trailer_docking.training.cli test \
  --policy PATH/policy_final.zip --backend 2d --episodes 5

.venv/bin/python -m trailer_docking.training.cli compare \
  --controllers lqr,nmpc,ppo \
  --policy PATH/policy_final.zip \
  --backends 2d,3d --episodes 5 --seed 100000
```

The executable capability matrix is documented in
[`control-ui-capability-matrix.md`](../../../docs/design/control-ui-capability-matrix.md).
Passing it proves that each adapter/backend combination executes; it is not a
claim of successful curved docking or equal 2D/3D performance.

Use `--no-render` for a headless acceptance run. The default scenario is the
compact baseline; `--scenario service-center` selects the full-size yard. Both
use an explicit straight reference path, a `0.35 m` lateral offset, and a
`3 degree` initial heading offset by default.

Every live controller view displays the active method in a prominent title.
Optional MP4 export belongs to the evaluation runner, not the controller:

```bash
.venv/bin/pip install -e '.[video]'
.venv/bin/python -m trailer_docking.evaluation.runner \
  --controller nmpc \
  --scenario service-center \
  --record artifacts/controller_videos/nmpc_service_center.mp4 \
  --speed 4 \
  --frame-stride 2
```

`--speed` controls playback rate in the exported file. `--frame-stride` reduces
rendering cost by capturing every Nth controller step. The terminal frame is
held briefly so the final status remains readable.
