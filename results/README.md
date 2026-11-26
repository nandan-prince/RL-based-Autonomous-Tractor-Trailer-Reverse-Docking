# Experiment results

This tracked directory holds compact, publication-relevant evidence. It is not
a dumping ground for checkpoints, raw videos, or temporary training output.

Each significant run receives an immutable ID such as
`YYYYMMDD-HHMM_<method>_<short-purpose>` and a matching configuration under
`configs/experiments/`. Store its compact evidence as:

```text
results/<experiment-id>/
├── manifest.yaml   # provenance, versions, command, status, and artifact paths
├── metrics.csv     # per-episode or aggregated metrics when reasonably sized
├── summary.json    # machine-readable headline results
├── notes.md        # failures, interpretation, validity, and caveats
└── figures/        # figures generated reproducibly from this run
```

Large or regenerable outputs belong under ignored
`artifacts/<experiment-id>/` paths. Record their locations and checksums in the
manifest when they are needed for reproduction.

Controlled PPO reward studies additionally store a top-level `study.yaml`, one
validation CSV per condition/seed cell, `cells.csv`, and `summary.json`. Their
validation results are tuning evidence and must not be relabeled as held-out
test results.

For MuJoCo runs, the manifest must include the complete resolved
`MujocoTuning` values as well as vehicle/dock fingerprints and the Git revision.
The tuning values are not currently included in the vehicle YAML fingerprint.

Never overwrite a completed result directory. Preserve meaningful failed or
negative runs and label invalid implementation runs clearly. Every manuscript
number, table, and experimental figure must identify the experiment IDs from
which it was produced. No formal controller experiments have been recorded
yet in the tracked evidence set; generated LQR/NMPC demonstration videos and
unregistered training campaigns are engineering artifacts, not reportable
comparative evidence.
