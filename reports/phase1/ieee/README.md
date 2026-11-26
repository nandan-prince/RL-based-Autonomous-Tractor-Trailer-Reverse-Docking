# Cumulative IEEE / Overleaf course report

This is the existing IEEE report, extended in place with Phase 2 PPO progress.
It retains the Phase 1 environment and classical-controller evidence.

## Compile in Overleaf

1. Update the existing Overleaf project with the contents of this directory.
2. Set `main.tex` as the main document.
3. Use the pdfLaTeX compiler. The project uses the standard `IEEEtran` class
   and BibTeX; no custom class or local package is required.

The included `figures/` directory is sufficient to compile the report. The
large MP4 files are deliberately not part of the Overleaf project.

## Evidence provenance

The compact controller montages were extracted from the labelled videos in
`artifacts/controller_videos/`. Each montage selects the
perturbed start, the maximum-articulation step, and terminal success. The
tracking plot and numerical metrics are regenerated through the public
controller contract by:

```bash
.venv/bin/python reports/phase1/make_controller_evidence.py
```

If the ignored local videos are absent, recreate them first:

```bash
.venv/bin/python -m trailer_docking.evaluation.runner --controller lqr  --scenario compact        --record artifacts/controller_videos/lqr_compact.mp4        --speed 4 --frame-stride 2
.venv/bin/python -m trailer_docking.evaluation.runner --controller nmpc --scenario compact        --record artifacts/controller_videos/nmpc_compact.mp4       --speed 4 --frame-stride 2
```

The plant and MuJoCo artifacts remain reproducible with
`../make_evidence.py`.

## Scope of the evidence

The Phase 2 montage replays the
[farther-start PPO run](../../../results/20261126-0014_ppo_ppo-ui-run-1-of-1/manifest.yaml)
at validation seed 100000. To regenerate it from the repository root:

~~~bash
.venv/bin/python reports/phase1/make_ppo_progress_evidence.py
~~~

The other Phase 2 results come from the
[aligned run](../../../results/20261125-2331_ppo_ppo-ui-campaign/summary.json),
[100k recovery run](../../../results/20261125-2346_ppo-curriculum_ppo-open-dock-recovery-v1/summary.json),
and [200k recovery run](../../../results/20261125-2359_ppo_ppo-ui-campaign/summary.json).

For the gentle-curve section, see the
[training manifest](../../../results/20261126-1213_ppo_ppo-ui-campaign/manifest.yaml),
[sampled training episodes](../../../results/20261126-1213_ppo_ppo-ui-run-1-of-1/metrics.csv),
and [final evaluation](../../../results/20261126-1213_ppo_ppo-ui-campaign/evaluation_run1_2d.csv).
The selected 100k checkpoint and LQR were replayed on the same 20 nearby
starts. The [paired summary](../../../results/20261126-1223_comparison_gentle-curve-nearby-diagnostic/summary.json),
[episode metrics](../../../results/20261126-1223_comparison_gentle-curve-nearby-diagnostic/metrics.csv),
and [resolved configuration](../../../configs/experiments/20261126-1223_comparison_gentle-curve-nearby-diagnostic.yaml)
record the exploratory diagnostic and checkpoint selection.

Regenerate the gentle-curve replay and training plot from the repository root:

~~~bash
.venv/bin/python reports/phase1/make_ppo_progress_evidence.py --gentle-curve
~~~

Replay the aligned 2D-trained policy in MuJoCo with:

~~~bash
.venv/bin/python -m trailer_docking.training.cli test \
  --bundle artifacts/20261125-2331_ppo_ppo-ui-campaign/policy_bundle.dockpolicy.zip \
  --backend 3d --episodes 1 --seed 100000
~~~

The single-direction quarter-arc pilot is recorded in the
[campaign manifest](../../../results/20261126-1304_ppo_arc-one-direction-campaign/manifest.yaml)
and [summary](../../../results/20261126-1304_ppo_arc-one-direction-campaign/summary.json).
See the [training outcomes](../../../results/20261126-1304_ppo_arc-one-direction-run-1-of-1/metrics.csv)
and arc replays for the
[140k checkpoint](../../../results/20261126-1304_ppo_arc-one-direction-run-1-of-1/arc_replay_140k.csv)
and [final checkpoint](../../../results/20261126-1304_ppo_arc-one-direction-run-1-of-1/arc_replay_final.csv).
The four-start screen and 20-start replay overlap, so the 140k checkpoint
result is exploratory validation.

The shorter one-sided arc uses the
[training log](../../../results/20261126-1316_ppo_ppo-ui-run-1-of-1/metrics.csv),
[campaign manifest](../../../results/20261126-1316_ppo_ppo-ui-campaign/manifest.yaml),
and [paired checkpoint replays](../../../results/20261126-1316_ppo_ppo-ui-run-1-of-1/arc_replay_120k_and_final.csv).
The 120k checkpoint was selected on a subset of the validation starts.

The gated-progress fine-tune has a
[campaign manifest](../../../results/20261126-1327_ppo_ppo-ui-campaign/manifest.yaml),
[training outcomes](../../../results/20261126-1327_ppo_ppo-ui-run-1-of-1/metrics.csv),
and [paired arc and approach replays](../../../results/20261126-1327_ppo_ppo-ui-run-1-of-1/arc_and_approach_replays.csv).
The table compares the original initialization, the 140k fine-tuned
checkpoint selected by a four-start screen, and the final policy.

The next matched comparison uses the gated preset and the
`configs/scenarios/open_dock_arc_30_recovery_reward_v1.yaml` intervention.
Both start from the original 120k checkpoint from campaign
`20261126-1316_ppo_ppo-ui-campaign`, with a 60,000-step budget and
learning rate 0.0001.

The recovery-reward arm is documented in the
[campaign summary](../../../results/20261126-1348_ppo_arc30-recovery-campaign/summary.json)
and [matched checkpoint replays](../../../results/20261126-1348_ppo_arc30-recovery-run-1-of-1/arc_and_recovery_replays.csv).
The replay CSV records whether the trailer returned within 0.3 m after
exceeding 0.5 m of lateral error in the straight-approach interval. The
matched [gated-only control summary](../../../results/20261126-1358_ppo_arc30-recovery-control-campaign/summary.json)
and [control replay CSV](../../../results/20261126-1358_ppo_arc30-recovery-control-run-1-of-1/arc_and_recovery_replays.csv)
use the same 20 physical starts. The [intermediate intervention checkpoints](../../../results/20261126-1348_ppo_arc30-recovery-run-1-of-1/arc_and_recovery_mid_checkpoints.csv)
complete the 20k, 40k, 60k, and final comparison.

The two Phase 1 controller runs are deterministic, perturbed straight-path
acceptance tests. They prove end-to-end closed-loop integration and do not
constitute the planned randomized controller comparison. The report states
this distinction explicitly.
