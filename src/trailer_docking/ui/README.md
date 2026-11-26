# Training control UI

The desktop UI is an operator interface over the same scenario, training, and
evaluation APIs used by the CLI. It does not implement a separate simulator or
controller path.

## Start it

From the repository root:

```bash
.venv/bin/pip install -e '.[dev,mujoco,viz,train,control]'
.venv/bin/python -m trailer_docking.ui.app \
  --scenario configs/scenarios/curved_docking_mixed_v2.yaml
```

Use `--output-root PATH` to keep exploratory `results/` and `artifacts/` away
from the checkout. On macOS this application runs through ordinary Python;
`mjpython` is needed for the native interactive MuJoCo viewer, not the UI's
offscreen 3D panel.

## Workflows

Pass `--initial-policy PATH/policy_final.zip` when opening the app to
warm-start training from a saved PPO policy. The run starts a new optimizer
and records the starting checkpoint path and checksum.
Use `--timesteps`, `--learning-rate`, and `--purpose` to prefill the live
training form for a controlled run. Review the settings before pressing Start.

### Train PPO

- **Run** selects the 2D or 3D plant, budget, seed, repetitions, episode limit,
  checkpoint interval, PyTorch device, and post-training evaluation.
- **PPO** exposes every optimizer/rollout argument passed to
  Stable-Baselines3. The engineering starting learning rate is `0.0003`.
- **Rewards** exposes the effective task coefficients. Version 2 includes
  trailer-tail distance improvement (`5.0`), a worsening multiplier (`2.0`),
  and heading improvement (`1.0`). These are unvalidated hypotheses.

Periodic and final raw checkpoints are written under
`artifacts/<run-id>/checkpoints/`. A completed campaign creates
`artifacts/<campaign-id>/policy_bundle.dockpolicy.zip`; **Save portable model**
copies that bundle to a selected destination. Stop is graceful and may take
until the current environment step returns.

### Models & Evaluate

The model table scans `artifacts/*/checkpoints/*.zip`, including checkpoints
created during earlier application sessions. Select a checkpoint or open one
elsewhere. A portable model contains its training scenario and adopts that
scenario when loaded. Evaluation explicitly selects one of:

- selected PPO checkpoint;
- loaded portable PPO model;
- NMPC;
- LQR; or
- no-steering baseline.

Each can execute in 2D or 3D. Execution is an integration check; successful
learning, controller equivalence, and 2D-to-3D transfer require registered
experiments.

### Advanced scenario

This page edits dock/vehicle overrides, limits, start distributions, planner
sets, domain randomization, MuJoCo-only physics values, and obstacles. Apply
validates and rebuilds the scenario without modifying the frozen base YAML.
Save a scenario under a new versioned name before using it for a significant
run.

## Verification

```bash
# Headless view/controller logic.
.venv/bin/pytest tests/unit/ui/test_ui_controller.py \
  tests/unit/ui/test_ui_framework.py -q

# Real Tk window; requires a desktop session.
.venv/bin/pytest tests/unit/ui/test_ui_window.py -q

# Opens and closes the application automatically.
.venv/bin/python -m trailer_docking.ui.app --smoke-test 2
```

Long PPO curricula and controlled multi-seed reward studies remain CLI
workflows. The UI currently runs a single campaign and does not provide exact
optimizer-state resume, parallel environments, TensorBoard, or curriculum
visualization.
