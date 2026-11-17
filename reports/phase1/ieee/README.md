# Phase 1 IEEE / Overleaf report

This folder is a self-contained Overleaf project for the Phase 1 submission.
Its controller claims and figures correspond to the implemented LQR and basic
NMPC controllers.

## Compile in Overleaf

1. Upload the contents of this `ieee/` directory as a new Overleaf project.
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

The two controller runs are deterministic, perturbed straight-path
acceptance tests. They prove end-to-end closed-loop integration and do not
constitute the planned randomized controller comparison. The report states
this distinction explicitly.
