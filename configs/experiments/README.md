# Experiment configurations

Each reported experiment must have an immutable configuration containing at
least:

- experiment ID and purpose
- source revision and vehicle/dock specification fingerprints
- simulator and controller configuration
- complete resolved `MujocoTuning` values for every MuJoCo run
- scenario distribution and held-out split
- random seeds
- trailer-length and dock-dimension variations
- observation/noise settings
- training budget and evaluation episode count
- requested metrics and artifact paths

Use IDs of the form `YYYYMMDD-HHMM_<method>_<short-purpose>`, optionally with a
repeat or seed suffix. A rerun with changed inputs receives a new ID rather
than overwriting the old record.

Future runners should write outputs beneath `results/<experiment_id>/` with a
copy of the resolved configuration, manifest, per-episode metrics, aggregate
metrics, and logs. Large trajectories, videos, checkpoints, and TensorBoard
data belong in ignored artifact storage; report-ready tables and figures must
retain the originating experiment ID. Never add measured values to the paper
unless the corresponding evidence is preserved.
