# Living LaTeX report

`main.tex` is the source of truth for the course report and the starting point
for any later research manuscript. It uses the generic IEEE conference layout
provided by `IEEEtran`; no specific IEEE venue is claimed. The content remains
intentionally neutral: planned work is marked as planned, and results are
included only when they are traceable to saved evidence.

The current manuscript records the proposal, literature basis, frozen vehicle
and environment contract, controller comparison, experimental protocol, and
implemented vehicle core, 2D kinematic plant and Gymnasium task, MuJoCo plant,
direct low-speed plant-trajectory parity, and the separately configured
full-size interactive service-center demonstration. A trajectory-error LQR and
basic straight-path NMPC now provide closed-loop vertical slices through the
same controller interface. The PPO training/evaluation infrastructure,
parameterized optimization and reward settings, and fingerprinted paired
episode replay are implemented and software-tested. Versioned classical
controller settings, disjoint experiment splits, formal evidence output, and a
gated policy-warm-start curriculum are also implemented. Smooth predefined
curves, a curved curriculum, and a controlled reward-study runner are
software-tested. The current version-2 PPO task adds dense trailer-tail and
heading-improvement reward terms; their learning benefit remains untested and
version 1 is retained for reproducibility. A full curriculum or reward-study
run, tuned PPO policy,
broader NMPC validation, MuJoCo task-event parity, and comparative controller
results remain explicitly labeled as planned until their evidence exists.

## Overleaf

Upload or synchronize the complete `paper/` directory and set `main.tex` as the
main document. In Overleaf, select a TeX Live version containing `IEEEtran`.
Repository edits do not automatically synchronize an Overleaf project;
configure and operate that workflow explicitly.

## Local build

When a LaTeX distribution is installed:

```bash
cd paper
latexmk -pdf main.tex
```

The report intentionally uses common packages available in standard Overleaf
installations.

## Updating the report

- Update methods and modeling text in the same task as substantive verified
  code changes.
- When a frozen 2D/MuJoCo interface changes, update the corresponding design
  document, configuration, tests, decision log, and manuscript together.
- Keep unimplemented items labeled `Planned`.
- Link every reported experimental number or generated figure to immutable
  experiment IDs under `results/`.
- Keep raw trajectories, videos, checkpoints, and TensorBoard data outside this
  directory.
- Do not state novelty, superiority, or statistical significance without saved
  evidence and a supporting literature review.
