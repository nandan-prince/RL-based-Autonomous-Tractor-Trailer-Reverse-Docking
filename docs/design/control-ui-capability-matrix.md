# Control UI and backend capability matrix

This is the executable contract for the desktop control window. It separates
training capability from controller evaluation so the UI does not present a
classical controller as a trainable algorithm.

## Algorithms and simulators

| Algorithm/source | Train in 2D | Train in 3D | Test in 2D | Test in 3D |
|---|---:|---:|---:|---:|
| PPO | Yes | Yes | Yes | Yes |
| NMPC | No | No | Yes | Yes |
| LQR | No | No | Yes | Yes |
| Scripted zero-steering baseline | No | No | Yes | Yes |

The **Train PPO** tab therefore exposes PPO only. NMPC and LQR are selected
under **Models & Evaluate**. They calculate steering-rate commands online but
do not optimize parameters or produce checkpoints. A PPO test uses a selected
checkpoint or loaded portable model; there is no implicit model fallback.

The automated capability test executes all PPO-policy/LQR/NMPC by 2D/3D
evaluation cells for a short rollout and performs a real tiny PPO optimization
on both simulator backends. Passing means the interfaces execute; it is not
evidence of successful docking or comparable performance.

## Display controls

- **Top-down (2D and 3D runs)** displays the shared planar state for every
  algorithm and backend.
- **3D view (MuJoCo)** displays an idle preview and live frames only when the
  active run uses the 3D backend and that workflow's live-render option was
  enabled before the run started.
- **Show planned route** affects both displays and can be changed during a
  run.
- MuJoCo images preserve aspect ratio and refit to the display panel.
- Rendering is deliberately unavailable for a 2D run; the top-down display is
  the correct view in that case.

## Training controls

- **Run** controls the experience simulator, experiment name, budget, seed,
  independent repetitions, episode limit, checkpoint interval, device, and
  post-training evaluation.
- **PPO** exposes every PPO argument passed to Stable-Baselines3. Values are
  validated by the same immutable `TrainingConfig` used by the CLI.
- **Rewards** exposes all ten effective trajectory/task reward coefficients.
  Starting a run applies and validates these values in the scenario before
  creating the campaign, so the manifest and portable model retain them.
- PPO may finish at the next complete rollout boundary after the requested
  timestep budget.
- Stop requests a clean stop. **Save portable model** becomes available only
  after a PPO campaign has produced a bundle. Raw periodic and final
  checkpoints are saved automatically.
- Training controls are locked while training or testing is active.

The gated curriculum and reward-study runners remain command-line workflows;
the desktop window runs one ordinary campaign at a time.

## Model and evaluation controls

- The model browser discovers `artifacts/*/checkpoints/*.zip` on startup and
  refresh, including models written by previous application sessions.
- A raw checkpoint can also be opened directly. A portable model loads its
  saved scenario and training metadata.
- Controller selection is explicit: selected PPO model, loaded portable model,
  NMPC, LQR, or no-steering baseline. No missing checkpoint silently becomes
  the baseline.
- Simulator selects the plant used for the complete evaluation episode.
- Episodes and seed define the deterministic evaluation sequence.
- NMPC and LQR are rebuilt after each reset from that episode's resolved
  vehicle geometry and reference path.
- Stop interrupts the test at its next control step.

## Scenario controls

Vehicle geometry, limits, dock dimensions, start distributions, reference
paths, reward weights, obstacles, and observation noise feed the shared task
definition and therefore affect 2D and 3D. Tractor/trailer mass and ground
friction are explicitly labeled **3D only** and have no effect on the
kinematic plant.

The single-planner selector includes `entry`, `astar`, `curve_gentle`,
`curve_early`, and `curve_late`. The optional comma-separated path set samples
one registered planner per episode. The selected path is frozen for replay.

## Evidence and limits

The capability matrix is covered by software tests. It does not establish that
NMPC or LQR succeeds on arbitrary curves, that PPO has learned docking, or that
2D and 3D controller performance is equivalent. Those require registered
experiments on the frozen validation and held-out suites.
