# Controllers

The controller package implements the common interface in
[`docs/design/controller-interface.md`](../../../docs/design/controller-interface.md).

Implemented controllers:

- `TrajectoryLQRController`: a deterministic trajectory-error LQR controller
  derived from a numerical linearization of the verified kinematic backend.
- `BasicNMPCController`: a CasADi/IPOPT nonlinear receding-horizon tracker with
  fixed reverse speed and bounded steering-rate authority.

The PPO module remains intentionally untouched for parallel development. PPO
training may use the Gymnasium environment directly; its evaluation adapter
should implement `Controller` and return the same normalized one-element action.

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
