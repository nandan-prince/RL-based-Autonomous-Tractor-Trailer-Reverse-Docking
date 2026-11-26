# Phase 2 progress report

The cumulative submission is now the [existing IEEE course report](../phase1/ieee/main.tex).
This directory holds supplementary Phase 2 notes and reproduction commands.

Upload [main.tex](main.tex) to an IEEE conference project in Overleaf and
select it as the main document. This report uses only standard LaTeX packages.

## Evidence

- Aligned open bay: [run manifest](../../results/20261125-2331_ppo_ppo-ui-run-1-of-1/manifest.yaml),
  [training episodes](../../results/20261125-2331_ppo_ppo-ui-run-1-of-1/metrics.csv),
  [deterministic evaluation](../../results/20261125-2331_ppo_ppo-ui-campaign/evaluation_run1_2d.csv).
- Perturbed centerline recovery: [curriculum manifest](../../results/20261125-2346_ppo-curriculum_ppo-open-dock-recovery-v1/manifest.yaml),
  [training episodes](../../results/20261125-2346_ppo_ppo-open-dock-recovery-v1-stage-1-centerline-recovery/metrics.csv),
  [validation episodes](../../results/20261125-2346_ppo-curriculum_ppo-open-dock-recovery-v1/validation_stage1.csv).
- Longer centerline recovery: [campaign manifest](../../results/20261125-2359_ppo_ppo-ui-campaign/manifest.yaml),
  [training episodes](../../results/20261125-2359_ppo_ppo-ui-run-1-of-1/metrics.csv),
  [validation episodes](../../results/20261125-2359_ppo_ppo-ui-campaign/evaluation_run1_2d.csv).

## Repeat the recovery run

Run from the repository root using the local checkpoint from the aligned
campaign. The runner saves a new experiment ID, resolved configuration,
checkpoint hash, and per-episode validation CSV.

~~~bash
.venv/bin/python -m trailer_docking.training.cli curriculum \
  --config configs/training/ppo_open_dock_recovery_v1.yaml \
  --initial-policy artifacts/20261125-2331_ppo_ppo-ui-run-1-of-1/checkpoints/policy_final.zip
~~~

A run below the 70% validation gate exits with status 1 and still saves the
completed training checkpoint and validation results. Validation seeds are
available for diagnosis. The final test split remains reserved for evaluation.

To watch the same recovery condition in the desktop UI:

~~~bash
.venv/bin/python -m trailer_docking.ui.app \
  --scenario configs/scenarios/open_dock_centerline_recovery_v1.yaml \
  --initial-policy artifacts/20261125-2331_ppo_ppo-ui-run-1-of-1/checkpoints/policy_final.zip
~~~

Under **Train PPO → Run**, select **2D**, **100000** timesteps, seed **0**,
one run, a **900** step episode limit, **20000** checkpoint interval, and
**20** evaluation episodes in 2D. Keep the PPO and reward defaults. Select
the top-down view, then click **Start training**. The UI runs a single
campaign and shows its final evaluation without applying a curriculum gate.
