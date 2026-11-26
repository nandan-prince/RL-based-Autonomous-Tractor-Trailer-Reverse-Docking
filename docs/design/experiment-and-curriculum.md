# Controlled experiments and PPO curriculum

This document defines how controller settings, dataset splits, formal
comparisons, and staged PPO training are represented. It complements the
[shared controller interface](controller-interface.md); it does not change the
plant action, observation, or task-event contracts.

## Versioned inputs

- `configs/controllers/baseline_controllers_v1.yaml` contains every LQR and
  NMPC weight, horizon, numerical tolerance, and iteration limit used by the
  implemented baselines. Its fingerprint belongs in comparison provenance.
- `configs/experiments/compact_docking_suite_v1.yaml` owns disjoint training,
  validation, and held-out test seeds. Validation may guide choices. The test
  split must not be used for reward shaping, controller tuning, curriculum
  gates, or policy selection.
- `configs/training/ppo_curriculum_v1.yaml` resolves the PPO hyperparameters,
  experiment suite, stage scenarios, budgets, and advancement thresholds.
- `configs/training/ppo_curved_curriculum_v1.yaml` shortens the aligned warm-up
  and then trains on one and finally three smooth predefined curve templates.
- `configs/training/ppo_curved_reward_study_v1.yaml` defines a multi-seed
  reward study on one frozen mixed-curve scenario distribution.

All loaders are strict and reject unknown fields, invalid values, duplicate or
overlapping split seeds, and inconsistent training ownership.

## Formal paired comparison

`training.cli compare` resolves the selected suite split once, then replays
the same geometry, start state, steering state, and reference path across each
controller/backend cell. With `--save-results`, it creates:

```text
configs/experiments/<experiment-id>.yaml
results/<experiment-id>/
  manifest.yaml
  metrics.csv
  summary.json
  notes.md
```

The generated experiment configuration contains the full scenario, suite,
controller settings, policy checksum when applicable, and every serialized
resolved episode. A second invocation receives a different immutable ID and
does not overwrite the first.

Validation example:

```bash
.venv/bin/python -m trailer_docking.training.cli compare \
  --scenario PATH/scenario.yaml \
  --suite configs/experiments/compact_docking_suite_v1.yaml \
  --split validation \
  --controllers lqr,nmpc,ppo \
  --policy PATH/policy_final.zip \
  --controller-config configs/controllers/baseline_controllers_v1.yaml \
  --backends 2d \
  --save-results \
  --purpose validation-controller-comparison
```

Selecting `--split test` is reserved for a frozen, final condition. Merely
having a test command available does not authorize repeated tuning against its
outcomes.

## Curriculum execution

The version-1 curriculum contains three gates:

1. aligned straight reverse;
2. perturbed straight reverse; and
3. randomized docking with start and trailer-length variation.

These budgets and thresholds are engineering starting points, not reported
optima. Each stage trains in a new `TrainingSession`, validates on the suite's
validation range, and advances only if its success-rate threshold is met. The
top-level curriculum result records each child training experiment and every
validation episode fingerprint.

Run it with:

```bash
.venv/bin/python -m trailer_docking.training.cli curriculum \
  --config configs/training/ppo_curriculum_v1.yaml
```

To restart at a later stage after an interruption or deliberate stop:

```bash
.venv/bin/python -m trailer_docking.training.cli curriculum \
  --config PATH/curriculum.yaml \
  --start-stage 1 \
  --initial-policy PATH/policy_final.zip
```

This is a **policy warm start**, not an exact optimizer resume. Network weights
are loaded through the safe weights-only loader; optimizer moments, rollout
buffer, learning-rate schedule, and timestep counters restart at each stage.
The manifest records this limitation. Do not describe the resulting stages as
one uninterrupted PPO optimization run.

The curriculum CLI prints training progress but is headless. The existing UI
continues to support live single-condition 3D training; curriculum visualization
and exact optimizer-state resume remain future interface work.

## Open-dock training condition

The `configs/scenarios/straight_open_dock_v1.yaml` scenario matches the
aligned straight warm-up and selects a painted bay with no physical dock
walls. Jackknife, workspace, timeout, and geometric docking criteria still
apply. Select the nominal dock to restore the walls.

The recovery preset in `configs/training/ppo_open_dock_recovery_v1.yaml`
keeps the bay open and samples small lateral and heading offsets. Its
`centerline` reference places all path points on the dock axis, so lateral
offsets remain tracking errors at reset. The preset supports a weights-only
warm start and validates on 20 fixed seeds after training.

## Predefined curved paths

The registered `curve_gentle`, `curve_early`, and `curve_late` planners are
sampled cubic paths ordered from the dock goal outward to the episode start.
Their endpoint tangents align with the dock axis, while their fixed control
fractions move the lateral transition to different parts of the maneuver. A
scenario can list them under `randomization.reference_planners`; the selected
name and complete sampled path are frozen into every resolved episode.

These templates are controlled trajectory-tracking inputs, not automatic
articulated-vehicle planners. They do not prove collision clearance,
jackknife avoidance, or reverse kinematic feasibility. Those properties are
still measured by the task during rollout.

Run the curved curriculum with:

```bash
.venv/bin/python -m trailer_docking.training.cli curriculum \
  --config configs/training/ppo_curved_curriculum_v1.yaml
```

## Quarter-arc path-observation pilot

The monitored pilot compares
`configs/scenarios/open_dock_arc_guided_v1.yaml` and
`configs/scenarios/open_dock_arc_goal_only_v1.yaml`. The open bay
approach pairs a 90-degree arc of radius 10--11 m with a tangent straight
leg into the dock. Starts are sampled on both mirror sides with up to one
degree of heading perturbation about the arc tangent. Both scenarios use
the same seed-driven physical starts, reference paths, and path-based reward.

Launch each configuration through `trailer_docking.ui.app --scenario` and
train from scratch with seed 0, 400,000 requested steps, the default PPO
settings, 2,000 steps per episode, 20,000 steps per checkpoint, and 20
deterministic 2D validation episodes. Keep both scenario configurations
fixed while collecting the pilot. Use validation to select checkpoints
before evaluating a final test split. The final policy and its automatic
evaluation are saved by the UI.

Both observation conditions retain the same vector size and normalize the
goal position by workspace bounds. Goal-only masks the path-relative
errors, remaining reference distance, and preview coordinates. The path
still supplies reward feedback during training. For each checkpoint record
docking success, terminal failures, arc coverage, and mean and peak lateral
tracking error on the arc. Arc coverage is needed when a policy terminates
before reaching the straight final leg.

### Single-direction feasibility run

The single-direction preset retains the guided observation, full quarter arc,
reward, and positive-side start ranges. It disables the start reflection.
Train from scratch with the task's 2,000-step limit:

~~~bash
.venv/bin/python -m trailer_docking.training.cli train \
  --scenario configs/scenarios/open_dock_arc_one_direction_v1.yaml \
  --backend 2d --timesteps 400000 --max-episode-steps 2000 \
  --runs 1 --seed 0 --purpose arc-one-direction \
  --eval-episodes 20 --eval-backends 2d
~~~

Use `reports/phase1/check_arc_checkpoints.py` with the training run ID to
replay the same validation starts across checkpoints. It reports projected
arc progress, progress before the first $0.5$ m lateral deviation, lateral
error, and outcome. Compare against a mirrored run with the same budget and
episode limit before attributing differences to the choice of turn sides.

### Short one-sided arc

`configs/scenarios/open_dock_arc_30_one_direction_v1.yaml` uses a
30-degree sweep at the same 10--11 m radius, with a tangent straight leg.
LQR docked in 775--818 steps on all 20 validation starts. For live PPO
training, open the preset in `trailer_docking.ui.app --scenario` and use
2D, seed 0, 400,000 requested training steps, a 2,000-step episode limit,
20,000-step checkpoints, and 20 evaluation episodes. Start from scratch
and keep the reward and PPO settings at their defaults. The arc checkpoint
diagnostic accepts both the short arc and the quarter arc.

## Controlled reward studies

`training.cli reward-study` trains every named reward condition on every
selected training seed, then evaluates it on the suite-owned validation range.
The study schema permits only reward coefficient changes. Vehicle, dock,
start distribution, reference-path distribution, PPO settings, budgets, and
validation inputs therefore remain shared. The runner verifies that every
condition resolves identical physical validation inputs even though reward
changes necessarily give each scenario and episode a different full
fingerprint.

```bash
.venv/bin/python -m trailer_docking.training.cli reward-study \
  --config configs/training/ppo_curved_reward_study_v1.yaml \
  --initial-policy PATH/policy_final.zip
```

The optional initial policy is applied identically to every cell through the
weights-only loader. Outputs include an immutable configuration under
`configs/experiments/`, a copy named `study.yaml`, per-cell validation CSV
files, `cells.csv`, `summary.json`, and a manifest with
child experiment IDs, checkpoints, checksums, physical episode fingerprints,
software versions, and Git state. Reward-study validation is tuning evidence,
not a held-out result. After choosing a reward condition, freeze it in a new
configuration before using the final test split.
