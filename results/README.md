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

Never overwrite a completed result directory. Preserve meaningful failed or
negative runs and label invalid implementation runs clearly. Every manuscript
number, table, and experimental figure must identify the experiment IDs from
which it was produced. No controller experiments have been recorded yet.
