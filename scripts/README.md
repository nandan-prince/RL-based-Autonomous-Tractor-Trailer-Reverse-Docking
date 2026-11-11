# Scripts

No training or evaluation scripts are implemented yet. This directory is
reserved for thin command-line entry points such as:

- training PPO
- evaluating one or more controllers
- rendering a deterministic rollout
- validating 2D/MuJoCo interface parity

Reusable implementation belongs under `src/trailer_docking`, not in scripts.
Scripts should resolve versioned configurations, print or save their
fingerprints, accept an explicit seed, and write evidence under a unique
experiment ID. They must not contain a second implementation of dynamics,
observations, rewards, controllers, or metrics.

The currently implemented conformance entry point lives in the package and can
be run directly:

```bash
.venv/bin/python -m trailer_docking.validation.vehicle_spec
```
