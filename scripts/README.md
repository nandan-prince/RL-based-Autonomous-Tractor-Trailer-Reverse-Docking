# Scripts

Reusable closed-loop evaluation is now implemented as a package entry point.
This directory remains reserved for future thin wrappers such as:

- training PPO
- evaluating one or more controllers
- rendering a deterministic rollout
- validating 2D/MuJoCo interface parity

Reusable implementation belongs under `src/trailer_docking`, not in scripts.
Scripts should resolve versioned configurations, print or save their
fingerprints, accept an explicit seed, and write evidence under a unique
experiment ID. They must not contain a second implementation of dynamics,
observations, rewards, controllers, or metrics.

Currently implemented utility entry points live in the package and can be run
directly:

```bash
.venv/bin/python -m trailer_docking.validation.vehicle_spec
.venv/bin/python -m trailer_docking.simulators.mujoco.builder
.venv/bin/python -m trailer_docking.visualization.viewer --mode 2d
.venv/bin/python -m trailer_docking.evaluation.runner --controller lqr
.venv/bin/python -m trailer_docking.evaluation.runner --controller nmpc
```

The evaluation runner also supports explicit MP4 generation with `--record`.
Recording is kept outside controller implementations and writes regenerable
media under the ignored `artifacts/` tree by convention.

See the root README for platform-specific MuJoCo 3D viewer commands.
