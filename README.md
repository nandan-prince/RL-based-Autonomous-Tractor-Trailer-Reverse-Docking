# RL-Based Autonomous Tractor–Trailer Reverse Docking

![Overhead view of a tractor-trailer reversing toward a loading dock](docs/assets/reverse-docking.png)

A simulation-based research project for safely reversing a tractor–trailer into
a logistics-center loading dock. The project compares trajectory-aware PPO and
NMPC in a shared 2D Gymnasium task, then evaluates transfer to a dimensionally
matched 3D MuJoCo model. A successful, reproducible 2D comparison is the core
deliverable; 3D evaluation is the planned extension.

## Project goal

The core research question is whether a PPO controller using path-relative
preview information can reliably reverse a tractor–trailer into a loading dock
and how it compares with NMPC under varied starting poses, trailer lengths,
dock dimensions, and observation noise.

The six-week core scope is intentionally focused on:

- A verified low-speed tractor–trailer model
- Fixed or simply generated docking references
- A classical trajectory-tracking baseline
- A trajectory-aware PPO controller
- Collision and jackknife detection
- Quantitative held-out evaluation
- One reproducible 3D reverse-docking demonstration

LiDAR mapping, visual dock-number recognition, advanced global planning, and dynamic obstacles are stretch goals rather than critical-path requirements.

## Repository structure

```text
.
├── configs/                    # Vehicle, task, controller, and experiment settings
├── docs/
│   ├── assets/
│   │   └── reverse-docking.png
│   ├── design/                 # Architecture and mathematical specifications
│   └── proposal/               # Submitted and expanded project proposals
├── research/
│   ├── README.md
│   └── papers/                 # Local reading copies; see copyright note
├── results/                    # Compact, traceable experiment evidence
├── paper/                      # Living IEEE-format LaTeX report
├── scripts/                    # Reserved for thin entry points (none yet)
├── src/trailer_docking/
│   ├── core/                   # Shared state, dynamics, geometry, and configuration
│   ├── paths/                  # Reference paths, tracking errors, and previews
│   ├── observations/           # Controller-facing observation construction
│   ├── safety/                 # Collision, jackknife, and command supervision
│   ├── simulators/
│   │   ├── kinematic/          # Fast deterministic 2D backend
│   │   └── mujoco/             # MuJoCo adapter, models, scenes, and assets
│   ├── environments/           # Gymnasium task over the 2D or MuJoCo plant
│   ├── evaluation/             # Resolved episodes, suites, metrics, and runners
│   ├── training/               # PPO campaigns, curricula, studies, bundles, CLI
│   ├── ui/                     # Control UI: parameters, training, tests, display
│   ├── visualization/          # Top-down view, MuJoCo viewer, offscreen 3D frames
│   ├── validation/             # Executable specification conformance report
│   └── controllers/            # Shared interface plus PPO, LQR, and NMPC adapters
└── tests/                      # Unit and integration tests
```

Start with the [architecture document](docs/design/architecture.md) for module
responsibilities and implementation order. A 2D contributor should then read
the [vehicle and environment specification](docs/design/vehicle-and-environment-specification.md).
A 3D contributor should use the self-contained
[MuJoCo contributor guide](src/trailer_docking/simulators/mujoco/README.md).
The verified contract coverage and remaining 2D-parity gates are recorded in
the [MuJoCo alignment audit](docs/design/mujoco-alignment-audit.md).

The frozen cross-simulator dimensions, coordinate conventions, state and
command contracts, dock geometry, and MuJoCo naming requirements are defined
in the [vehicle and environment specification](docs/design/vehicle-and-environment-specification.md).

## Proposal

- [Full project proposal](docs/proposal/full-proposal.md)
- [Submitted proposal PDF](docs/proposal/MSML642_Nandan_Prince_Chris_Pedretti.pdf)
- [Proposal presentation](docs/proposal/Group20_ProposalPresentation.pptx.pdf)
- [Six-week milestones and work division](docs/proposal/full-proposal.md#12-parallel-work-plan)

## Academic foundation

The project is informed by four principal works:

1. H. Yan et al., “Trajectory State Model-based Reinforcement Learning for Truck-trailer Reverse Driving,” ICRCA 2024. [IEEE DOI](https://doi.org/10.1109/ICRCA60878.2024.10649133)
2. B. Moradi and M. Mehrandezh, “A Non-linear MPC Local Planner for Tractor-Trailer Vehicles in Forward and Backward Maneuvering,” 2022. [arXiv](https://arxiv.org/abs/2212.11427)
3. H. Ma et al., “A Hierarchical Approach for Tractor-trailer Motion Planning Using Graph Search and Reinforcement Learning,” ECC 2025. [MERL publication](https://www.merl.com/publications/TR2025-093)
4. C. Menz and A. Wenzel, “Edge-based Imitation Learning for Autonomous Tasks: A Case Study on Tractor-Trailer Positioning,” 2026. [Publisher DOI](https://doi.org/10.1515/auto-2025-0081)

See the [research index](research/README.md) for summaries, relevance to this project, and additional technical references.

## Related industry-scale work

This problem is already receiving significant industrial investment and deployment:

- [Outrider](https://www.outrider.ai/press-releases/outrider-builds-first-in-industry-safety-system-for-driverless-yard-operations/) develops autonomous yard trucks that move, hitch, and back trailers in mixed-traffic logistics yards. Its system also integrates trailer brake-line connection and yard-management software. [Its 2024 Series D announcement](https://www.outrider.ai/press-releases/outrider-closes-62-million-series-d-financing-to-drive-growth-of-yard-automation/) reported a $62 million round and more than $250 million in total equity funding.
- [ISEE and TICO](https://www.isee.ai/news/business-wire-isee-and-tico-announce-strategic-partnership-to-deliver-industry-first-fully-integrated-autonomous-yard-trucks-to-customer-operations) are deploying OEM-integrated autonomous yard trucks at active logistics sites. Their system addresses mixed traffic, trailer movement, and production-scale yard operation.
- [Embotech, Terberg, and APM Terminals](https://www.terbergspecialvehicles.com/en/news/apm-terminals-maasvlakte-ii-embotech-and-terberg-expand-automated-terminal-tractor-fleet-in-rotterdam/) are scaling Level-4 automated terminal tractors in Rotterdam. Reported validation includes mixed-traffic driving, narrow-lane maneuvering, reverse parking, and centimeter-level motion planning.
- [FERNRIDE and HHLA](https://fernride.com/news/fernride-hhla-driverless-machinery-directive) are operating autonomous terminal tractors in Estonia following safety certification, with an emphasis on gradual driverless deployment in live port logistics.
- [Mitsubishi Electric Research Laboratories](https://www.merl.com/publications/TR2025-093) conducts directly related corporate research in tractor–trailer planning, reinforcement learning, optimization, and collision avoidance. Its [2026 adaptive planning work](https://www.merl.com/publications/TR2026-118) reports simulation and field-test validation.
- [Einride](https://www.einride.tech/press/einride-operates-first-ever-fully-autonomous-heavy-duty-truck-on-a-public-road-in-belgium-at-the-port-of-antwerp-bruges) and [Volvo Autonomous Solutions](https://www.volvoautonomoussolutions.com/en-en/autona-freight.html) are developing broader autonomous freight and terminal ecosystems. Their focus extends beyond dock backing but demonstrates the commercial direction of autonomous logistics.

These systems reinforce the project’s modular design: perception and localization supply a structured world state, planning produces a feasible maneuver, control executes it, and an independent safety layer monitors the vehicle.

## Current status

- Project proposal completed
- Six-week work plan defined
- Literature set collected and indexed
- Initial 2D/MuJoCo repository architecture created
- Frozen version-1 vehicle and dock specification created
- Shared vehicle state, command, geometry, strict configuration, provenance,
  command supervision, and conformance validation implemented
- First MuJoCo vehicle, nominal dock scene, and shared-contract adapter
  implemented and generated from the YAML specifications; validated against the
  specification and direct low-speed 2D simulator trajectories
- Canonical 2D kinematic backend implemented with generalized signed hitch
  offsets, acceleration-limited speed, steering-rate control, fixed-step RK4,
  rigid geometry reconstruction, and a common simulator protocol
- Reference paths, preview observations, collision, jackknife, docking success,
  deterministic scenarios, decomposed rewards, and Gymnasium API implemented
- Keyboard-controlled 2D Gymnasium and MuJoCo 3D/top-down viewers available
- Full-size 2D service-center yard available with a 200 m by 100 m paved area,
  20 loading doors along the bottom x-axis, a 14 m semitrailer, and an open
  color-highlighted active loading zone without collidable side walls; its
  soft/terminal articulation thresholds are 35/60 degrees
- Shared controller interface and explicit fixed-reference reset support added
- Straight-path trajectory-error LQR and basic fixed-speed NMPC controllers
  implemented with a common headless/interactive rollout runner
- PPO training framework on the shared Gymnasium task: one control UI that
  sets truck dimensions, limits, dock size, physics, start ranges, domain
  randomization, static obstacles, and the route planner, runs repeated
  training on the 2D or 3D simulator, tests each run, and exports the best
  policy as a portable bundle that can be loaded in a fresh simulator
- NMPC selectable through the same training CLI and UI integration hook; PPO,
  LQR, and NMPC share the fixed-reverse steering-rate plant contract
- Fingerprinted resolved episodes freeze geometry, start state, and reference
  path once for paired controller/backend comparisons
- PPO rollout and optimization hyperparameters plus every reward term are
  explicit validated configuration values and are recorded with each run
- The current version-2 PPO task adds dense trailer-tail and heading-error
  improvement rewards while preserving the original version-1 task for
  reproducible comparison; whether it learns faster remains untested
- LQR/NMPC settings, train/validation/test seed ownership, formal comparison
  evidence, and a gated three-stage PPO curriculum are versioned and tested
- Three smooth predefined curve templates, deterministic mixed-path sampling,
  a shortened straight-to-curved PPO curriculum, and a controlled multi-seed
  reward-study runner are implemented and software-tested
- These are working infrastructure and vertical slices, not comparative
  results: the full curved curriculum, reward study, curved/randomized
  validation, and frozen held-out experiment remain to be run
- Living IEEE-format LaTeX report updated through the combined controller and
  learning-pipeline milestone

## Teammate handoff: install and test

Pull this branch, enter the repository root, and create a clean environment:

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev,mujoco,viz,train,control]'
```

Run the high-value acceptance checks first:

```bash
.venv/bin/python -m trailer_docking.validation.vehicle_spec
.venv/bin/pytest tests/integration/test_control_capability_matrix.py -q
.venv/bin/pytest tests/integration/gymnasium \
  tests/integration/test_backends_and_randomization.py -q
.venv/bin/pytest tests/integration/mujoco \
  tests/integration/test_kinematic_mujoco_parity.py -q
```

Then open the desktop workflow with the current curved scenario:

```bash
.venv/bin/python -m trailer_docking.ui.app \
  --scenario configs/scenarios/curved_docking_mixed_v2.yaml
```

Use the **Train PPO** tab for one controlled campaign, **Models & Evaluate** to
run a saved PPO policy, LQR, or NMPC in either plant, and **Advanced scenario**
for geometry/start/path changes. Begin policy diagnosis in 2D, keep PPO's
learning rate at `0.0003`, and disable live MuJoCo rendering for long 3D runs.
The version-2 reward is an unvalidated hypothesis; do not report a smoke run as
evidence that it improves learning.

Training writes checkpoints and portable bundles under ignored `artifacts/`.
It also writes compact run records under `results/`; keep exploratory result
directories local unless the run has a deliberate immutable experiment ID,
configuration, notes, and valid interpretation. See [tests/README.md](tests/README.md)
for the complete acceptance sequence and
[results/README.md](results/README.md) before committing outputs.

## Vehicle-core verification

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev,mujoco,viz,train]'
.venv/bin/ruff check src tests
.venv/bin/mypy
.venv/bin/pytest --cov --cov-report=term-missing
.venv/bin/python -m trailer_docking.validation.vehicle_spec
.venv/bin/python -m trailer_docking.simulators.mujoco.builder
```

The conformance command loads the versioned YAML specifications, validates all
fields and cross-field constraints, reconstructs the neutral assembly, checks
rigid geometry and dock fit, and prints configuration fingerprints suitable for
future experiment manifests.

The `mujoco` extra currently admits MuJoCo 3.14 releases through the next major
version, and `viz` adds Matplotlib for the top-down viewer. See the MuJoCo
contributor guide before creating or renaming any MJCF element.

## Demo

This section is the walkthrough for seeing the project run. Each numbered step
is marked **Available** or **Planned**. Keep the step order and update the
status and commands in place rather than adding a separate guide.

### Setup (once)

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev,mujoco,viz,train,control]'
```

On Windows use `py -3.12 -m venv .venv` and replace `.venv/bin/` with
`.venv\Scripts\` in every command below. Run all commands from the repository
root.

On macOS, the 3D window requires MuJoCo's launcher. Because this repository
path contains spaces, invoke that launcher through the virtual-environment
Python:

```bash
.venv/bin/python .venv/bin/mjpython -m trailer_docking.visualization.viewer --mode 3d
```

The native MuJoCo and Matplotlib macOS event loops cannot currently share the
same process, so `--mode both` is not supported on macOS. Run `--mode 3d` and
`--mode 2d` in separate terminals. Those processes have independent simulator
instances; use one at a time when exact synchronized playback matters. On
platforms without that event-loop restriction, the commands below can use
`.venv/bin/python` directly.

### 1. Drive the Gymnasium 2D environment — Available

```bash
.venv/bin/python -m trailer_docking.environments.viewer
```

The window shows the kinematic tractor--trailer, loading-zone/building geometry,
reference path, preview points, driven trajectory, reward, tracking errors, and
episode status. The service-center renderer uses a tested rigid transform from
the dock frame, so tractor, hitch, and trailer geometry remain connected while
turning.
Use Up/Down to drive, Left/Right to steer, Space to stop, and R or Backspace to
reset. It defaults to the full-size service-center yard; use
`--scenario compact` for the compact research baseline. A complete tested
straight docking and a randomized start are available:

```bash
.venv/bin/python -m trailer_docking.environments.viewer --demo --speed 4
.venv/bin/python -m trailer_docking.environments.viewer --random-start --seed 7
.venv/bin/python -m trailer_docking.environments.viewer --scenario compact
```

See the [2D environment guide](src/trailer_docking/environments/README.md) for
the Gymnasium API, action modes, observation, events, and focused tests.

### Run the implemented controllers

The first controllers use the compact research baseline and the same explicit
straight reference path from a perturbed start. They do not record media.

```bash
# Deterministic trajectory-error LQR
.venv/bin/python -m trailer_docking.evaluation.runner --controller lqr

# Basic fixed-speed nonlinear MPC (requires the control extra)
.venv/bin/pip install -e '.[control]'
.venv/bin/python -m trailer_docking.evaluation.runner --controller nmpc

# Run either controller in the full-size service-center yard
.venv/bin/python -m trailer_docking.evaluation.runner --controller lqr --scenario service-center
.venv/bin/python -m trailer_docking.evaluation.runner --controller nmpc --scenario service-center

# Headless acceptance run
.venv/bin/python -m trailer_docking.evaluation.runner --controller lqr --no-render
```

Both methods return the shared one-dimensional normalized steering-rate action.
The alignment contract for parallel PPO development is documented in the
[controller interface](docs/design/controller-interface.md).

The controller name is displayed prominently above every interactive run.
Generate an H.264 MP4 without opening a GUI using:

```bash
.venv/bin/pip install -e '.[video]'
.venv/bin/python -m trailer_docking.evaluation.runner \
  --controller lqr \
  --scenario service-center \
  --record artifacts/controller_videos/lqr_service_center.mp4 \
  --speed 4 \
  --frame-stride 2
```

Video output is a regenerable artifact and remains ignored by Git.

### 2. See the generated MuJoCo dock and vehicle — Available

```bash
# Linux/Windows: both windows from one process
.venv/bin/python -m trailer_docking.visualization.viewer --mode both

# macOS terminal 1: MuJoCo 3D
.venv/bin/python .venv/bin/mjpython -m trailer_docking.visualization.viewer --mode 3d

# macOS terminal 2: shared-state top-down view
.venv/bin/python -m trailer_docking.visualization.viewer --mode 2d
```

In combined mode, two windows open and are driven by one simulation:

- **3D (MuJoCo):** the tractor, trailer, and loading bay generated from the
  YAML specifications.
- **2D (top-down):** the same scene drawn only from the shared planar state a
  controller receives, with speed, steering, articulation, and docking error.

Use `--mode 3d` or `--mode 2d` for one window only.

### 3. Drive MuJoCo yourself — Available

Click either window to give it keyboard focus.

| Key | Action |
|---|---|
| Up / Down | Drive forward / reverse at the speed limit |
| Left / Right | Steer left / right in small steps |
| Space | Stop |
| Backspace | Reset to the start pose |

Reversing the trailer into the bay by hand is a good way to feel why the task
is hard: the trailer swings opposite to the steering input and jackknifes if
left uncorrected.

### 4. Watch a scripted MuJoCo reverse — Available

```bash
# Linux/Windows
.venv/bin/python -m trailer_docking.visualization.viewer --mode both --demo --speed 4

# macOS 3D demo
.venv/bin/python .venv/bin/mjpython -m trailer_docking.visualization.viewer --mode 3d --demo --speed 4
```

The vehicle reverses straight into the bay and stops when the trailer touches
the back wall. This is a fixed command, not a controller.

### 5. Change the MuJoCo scenario from the command line — Available

| Option | Effect |
|---|---|
| `--variant short\|nominal\|long` | Trailer hitch-to-axle length of 2, 4, or 5 m |
| `--no-dock` | Empty yard without the loading bay |
| `--speed N` | Play N times faster than real time |
| `--vehicle PATH`, `--dock PATH` | Load a different vehicle or dock YAML |

Any vehicle or dock YAML that passes validation and that the MuJoCo backend
supports (for example, zero command delay) can be viewed this way; copy
one of the files under `configs/`, change a value such as the bay width, and
pass the new path.

### 6. See the reference route in the MuJoCo viewer — Available

Every docking episode starts by building a reference route for the trailer's
docking point from the start pose to the dock target; the controller's job is
to follow it. The MuJoCo viewer from steps 2–4 draws that route as a green
line in both windows. Press `P` to show or hide it.

The route is the same one the Gymnasium task tracks: a straight line along the
bay centreline for an aligned start, or a two-segment line through an entry
point just outside the bay for an offset start. It is not checked for trailer
feasibility; Reeds–Shepp or Hybrid A* planners remain optional extensions.

### 7. Open the control UI — Available

One window sets the parameters, runs either simulator, and displays the run.

```bash
.venv/bin/pip install -e '.[dev,mujoco,viz,train]'
.venv/bin/python -m trailer_docking.ui.app
```

The left side is organized by workflow:

1. **Train PPO** sets the run, PPO optimizer, and reward configuration.
2. **Models & Evaluate** finds checkpoints from earlier sessions and evaluates
   a selected PPO model, NMPC, LQR, or the no-steering baseline.
3. **Advanced scenario** edits geometry, paths, start distributions,
   randomization, physics, and obstacles.

The right side is the display, with two tabs:

- **Top-down** draws the shared planar state, so it shows 2D and 3D runs alike.
- **3D view (MuJoCo)** shows the rendered 3D scene: a preview of the current
  scenario when idle, and the live episode while a 3D run is in progress.
  Live rendering is a separate option for training and evaluation. Leave it
  off during long 3D training runs for substantially better throughput.

Only PPO appears in **Train PPO**, because NMPC and LQR do not learn parameters.
All three controllers can be run on both simulators from **Models & Evaluate**. The
complete supported/unsupported matrix and control semantics are recorded in
the [control UI capability matrix](docs/design/control-ui-capability-matrix.md).

The **Advanced scenario** tab edits a scenario: the frozen vehicle, dock, and task
specifications plus a set of overrides. **Apply** rebuilds the scene;
**Save scenario…** writes it to its own YAML file. The frozen specification
files are never modified.

| Group | Parameters | Affects |
|---|---|---|
| Dock | Interior width and depth | 2D and 3D |
| Route | Route planner (`entry`, `astar`, or a predefined smooth curve); reward weights for staying on the route and aligning to it | 2D and 3D |
| Reward | Progress, time, articulation, steering, success, collision, jackknife, and boundary weights | 2D and 3D |
| Vehicle | Trailer variant; tractor wheelbase and body width; trailer hitch-to-axle length, rear overhang, and body width | 2D and 3D |
| Limits | Maximum steering angle and steering rate; reverse speed | 2D and 3D |
| Physics (3D only) | Tractor and trailer mass, ground friction | 3D only |
| Start pose | Standoff, lateral offset, trailer yaw, articulation ranges | 2D and 3D |
| Randomization | Trailer length drawn per episode; observation noise; configured path-template sets | 2D and 3D |
| Obstacles | Static boxes in the yard | 2D and 3D |

The **route planner** produces each episode's reference route from the start
pose to the dock. `entry` is the default: straight along the bay centreline,
joined through a point just outside the opening. `astar` is a basic
path-finding planner that routes the docking point around dock walls.
`curve_gentle`, `curve_early`, and `curve_late` are sampled cubic templates
with aligned endpoint tangents and different transition locations. A scenario
may sample a configured set of those templates once per episode. All planners
are geometric and do not prove that the articulated vehicle can follow the
route safely in reverse; further planners register in
`paths/planners.py` and then appear in this list. The reward already
penalizes leaving the route (squared lateral and heading deviation from it),
and the two **Route** fields set how strongly; blank keeps the task file's
weights (0.5 and 0.2). The separate **Reward** group exposes every remaining
term. Zero deliberately disables a term for a controlled reward ablation.

**Obstacles** are static boxes. Tick **Click the map to place** and click the
top-down view where each box should go; its length, width, height, and turn
come from the fields beside it. A box can also be added by typing its `x` and
`y` (metres out from the bay's rear wall, and to the left), and the list
below removes one box or all of them. Each change is applied at once.
Touching one ends the episode as a collision
in both simulators, and they appear in both displays. The policy does not
sense obstacles; it avoids them only by following a route planned around
them, so choose the `astar` planner when obstacles lie near the route (the
`entry` planner ignores them). An obstacle that overlaps a start pose or the
docked vehicle is refused.

**Show planned route** above the display draws or hides the route in both the
top-down and the 3D view, before a run and while one is in progress.

Rest the pointer on any field, button, or option for a short explanation of
what it does and its default.

A blank field keeps the base value; a blank randomization field leaves that
randomization off. **Apply** refuses a scenario that could never succeed or
that would start an episode already failed, and says why: for example a
trailer that cannot be completely inside the bay (the `long` variant needs a
bay deeper than the nominal 6 m), or a start range that begins in a wall.

### 8. Train and get a finished result — Available

On **Train PPO**, the **Run** page selects the simulator, budget, seed,
checkpoint interval, device, evaluation settings, and number of independent
runs. The **PPO** page exposes every Stable-Baselines3 setting passed by this
project: learning rate, rollout size, batch size, optimization epochs,
discount, GAE, policy/value clipping, advantage normalization, entropy/value
coefficients, gradient clipping, and target KL. The **Rewards** page exposes
the effective path, progress, time, articulation, steering, success, collision,
jackknife, and workspace coefficients. These displayed values are the values
saved with the scenario and run; they are not hidden library defaults.

The current `gymnasium-docking-v2` reward also exposes **Tail moves toward
path**, **Tail deviation penalty multiplier**, and **Heading improvement**.
The defaults reward normalized tail-distance reduction by `5.0`, charge
worsening at `2.0` times that rate, and reward reduction in absolute trailer
heading error by `1.0`. The original absolute lateral/heading and articulation
penalties remain active. These values are an initial reward-design hypothesis,
not a demonstrated optimum.

The core run controls are:

| Setting | Meaning |
|---|---|
| Simulator | `2d` (kinematic, fast) or `3d` (MuJoCo) for training |
| Independent runs | How many times to repeat training, with consecutive seeds |
| Timesteps | PPO budget of each run |
| Save every N steps | Checkpoint interval; the final policy is always saved |
| Evaluate after training, 2D / 3D | After each run the final policy is tested on the selected simulators, on the same start poses for every run |

Press **Start training**. The display shows the episodes as they happen, and the status
panel reports the run in progress, recent success rate, and each finished
run's test results. **Stop** ends the work cleanly.

When the runs finish, the best one (highest test success rate, then mean
reward, on the first ticked simulator; the first run if none was tested) is
written as a single portable file,
`artifacts/<campaign-id>/policy_bundle.dockpolicy.zip`. **Save portable model…**
copies it wherever you choose. The bundle contains the policy and the scenario
it was trained on. Loading a bundle or policy file reads only its network
weights and plain-text metadata, never pickled objects, so a file received
from someone else cannot run code.

The same thing without the UI:

```bash
.venv/bin/python -m trailer_docking.training.cli train \
    --runs 3 --timesteps 200000 --backend 2d --eval-episodes 20 --eval-backends 2d,3d
```

The PPO settings are not hidden library defaults. They can be changed from the
CLI and are written to the run manifest, for example:

```bash
.venv/bin/python -m trailer_docking.training.cli train \
    --timesteps 200000 --learning-rate 0.0003 \
    --n-steps 1024 --batch-size 256 --n-epochs 10 \
    --gamma 0.99 --gae-lambda 0.95 --clip-range 0.2 \
    --entropy-coefficient 0.0 --value-coefficient 0.5 \
    --max-grad-norm 0.5 --device cpu
```

Run paired controller cells on exactly the same resolved episodes with:

```bash
# LQR and NMPC on one frozen 2D episode set
.venv/bin/python -m trailer_docking.training.cli compare \
    --controllers lqr,nmpc --backends 2d --episodes 20 --seed 100000

# Add a trained PPO policy and replay the same set in both plants
.venv/bin/python -m trailer_docking.training.cli compare \
    --controllers lqr,nmpc,ppo --policy PATH/policy_final.zip \
    --backends 2d,3d --episodes 20 --seed 100000
```

The command prints the stable episode fingerprints used by every cell. To
create a formal validation evidence package with the versioned controller
settings and experiment suite:

```bash
.venv/bin/python -m trailer_docking.training.cli compare \
    --scenario PATH/scenario.yaml \
    --suite configs/experiments/compact_docking_suite_v1.yaml \
    --split validation \
    --controllers lqr,nmpc,ppo \
    --policy PATH/policy_final.zip \
    --controller-config configs/controllers/baseline_controllers_v1.yaml \
    --backends 2d \
    --save-results \
    --purpose validation-controller-comparison
```

This writes an immutable configuration under `configs/experiments/` and the
manifest, per-episode metrics, summary, and caveats under `results/`. Reserve
`--split test` for the final frozen condition; do not tune against it.

Run the configured PPO curriculum with:

```bash
.venv/bin/python -m trailer_docking.training.cli curriculum \
    --config configs/training/ppo_curriculum_v1.yaml
```

It trains aligned straight reverse, then perturbed reverse, then randomized
docking, advancing only when the validation success gate passes. To continue
from a chosen checkpoint at a later stage:

```bash
.venv/bin/python -m trailer_docking.training.cli curriculum \
    --config configs/training/ppo_curriculum_v1.yaml \
    --start-stage 1 \
    --initial-policy PATH/policy_final.zip
```

This continuation loads policy weights safely but resets PPO optimizer and
schedule state. It is a warm start, not an exact interrupted-run resume. The
curriculum CLI is headless; use the control UI for live single-condition 3D
training. See the
[experiment and curriculum contract](docs/design/experiment-and-curriculum.md).

To move quickly from the straight warm-up into mixed predefined curves:

```bash
.venv/bin/python -m trailer_docking.training.cli curriculum \
    --config configs/training/ppo_curved_curriculum_v1.yaml
```

If you already have an acceptable straight-policy checkpoint, skip the warm-up
and begin at the first curved stage:

```bash
.venv/bin/python -m trailer_docking.training.cli curriculum \
    --config configs/training/ppo_curved_curriculum_v1.yaml \
    --start-stage 1 \
    --initial-policy PATH/policy_final.zip
```

For a single exploratory PPO campaign instead of the gated curriculum:

```bash
.venv/bin/python -m trailer_docking.training.cli train \
    --scenario configs/scenarios/curved_docking_mixed_v2.yaml \
    --timesteps 150000 --runs 1 --seed 0
```

To compare named reward variants on the same curved-path distribution,
training seeds, PPO settings, and validation inputs:

```bash
.venv/bin/python -m trailer_docking.training.cli reward-study \
    --config configs/training/ppo_curved_reward_study_v1.yaml \
    --initial-policy PATH/policy_final.zip
```

The initial policy is optional, but supplying the same straight-policy
checkpoint to every condition makes the reward study a controlled warm-start
experiment. Edit a copied, newly named study YAML to change reward conditions;
do not overwrite a configuration already associated with results.

Every run writes its own `results/<id>/manifest.yaml` and `metrics.csv`, and
the campaign writes `results/<campaign-id>/manifest.yaml`, `summary.json`, and
one `evaluation_run<k>_<simulator>.csv` per test with every episode. With the
default output root these land in the tracked `results/` directory; commit
only runs meant as evidence, or pass `--output-root` (CLI and UI) to write
elsewhere.
Checkpoints and the bundle go under the ignored `artifacts/` directory.

**Read this before interpreting anything:** the reward weights and the
scenario's start ranges are untuned, the training step limit overrides the
task's, and the 3D simulator's episode events have not been validated against
2D. Treat every run as a pipeline check until it is registered as an
experiment with a held-out evaluation.

_Not implemented: parallel environments, exact optimizer-state resume,
curriculum visualization in the UI, and TensorBoard._

### 9. Load a result and run it in a fresh simulator — Available

The **Models & Evaluate** tab automatically lists raw PPO checkpoints under
`artifacts/*/checkpoints/`, including models produced before the UI was
opened. Select one, or use **Open checkpoint…** for a checkpoint elsewhere.
**Open portable model…** loads a `.dockpolicy.zip` bundle and adopts its saved
scenario. Choose the controller, simulator, episode count, and seed, then
press **Run test**. A policy trained in 2D can be run in 3D and the reverse.

The same tab tests a selected PPO model, portable bundle, no-steering baseline,
NMPC, or LQR. Each episode is
listed with its outcome (success, collision, jackknife, out of bounds, time
limit) and final errors.

From the command line:

```bash
.venv/bin/python -m trailer_docking.training.cli test --bundle PATH/result.dockpolicy.zip --backend 3d --episodes 20
.venv/bin/python -m trailer_docking.training.cli test --episodes 20          # untrained baseline
.venv/bin/python -m trailer_docking.training.cli test --nmpc --episodes 20
.venv/bin/python -m trailer_docking.training.cli test --lqr --episodes 20
```

The NMPC baseline plugs in through the shared per-episode controller factory;
see the [NMPC integration handoff](docs/design/nmpc-integration-handoff.md).

### What the demo currently does not show

The Gymnasium environment is functional and interactive, the PPO pipeline
trains and tests policies in 2D and 3D, and basic NMPC and LQR vertical slices
are implemented. No trained PPO policy or controller comparison has been
registered as an experiment. The 2D `--demo` is a
constant reverse command used to validate the complete success path; it is not
a controller-performance result. The 3D task uses the same observation,
reward, and event definitions as 2D, but agreement of its episode events with
2D has not been validated.

## Team workflow

Development should use short feature branches and reviewed pull requests:

```bash
git switch -c your-name/kinematic-model
# make and test a focused change
git add .
git commit -m "Implement tractor-trailer kinematic model"
git push -u origin your-name/kinematic-model
gh pr create --fill
```

## Copyright note

The papers used for study remain the property of their respective authors and
publishers. Local study copies are currently tracked under `research/papers/`;
their presence in this working repository is not permission to redistribute
them. Audit each license and remove any copy that cannot be shared before making
the repository public. The research index provides stable publisher or
author-hosted links for citation and sharing.
