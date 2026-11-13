# Living LaTeX report

`main.tex` is the source of truth for the course report and the starting point
for any later research manuscript. It uses the generic IEEE conference layout
provided by `IEEEtran`; no specific IEEE venue is claimed. The content remains
intentionally neutral: planned work is marked as planned, and results are
included only when they are traceable to saved evidence.

The current manuscript records the proposal, literature basis, frozen vehicle
and environment contract, controller comparison, experimental protocol, and
implemented vehicle core and MuJoCo plant. Kinematic propagation, Gymnasium,
NMPC, PPO, direct 2D/MuJoCo parity, and controller results remain explicitly
labeled as planned until their code and evidence exist.

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
