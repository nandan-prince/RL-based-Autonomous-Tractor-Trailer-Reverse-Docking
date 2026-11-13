# Scripts

No scripts live here yet; training and evaluation are run through package
entry points listed below. This directory is
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

Currently implemented utility entry points live in the package and can be run
directly:

```bash
.venv/bin/python -m trailer_docking.validation.vehicle_spec
.venv/bin/python -m trailer_docking.simulators.mujoco.builder
.venv/bin/python -m trailer_docking.visualization.viewer --mode 2d
.venv/bin/python -m trailer_docking.environments.viewer
.venv/bin/python -m trailer_docking.ui.app
.venv/bin/python -m trailer_docking.training.cli --help
```

The last two need the `train` extra. `training.cli` trains PPO campaigns, runs
docking tests, and exports policy bundles; it writes each run under a unique
experiment ID.

See the root README for platform-specific MuJoCo 3D viewer commands.
