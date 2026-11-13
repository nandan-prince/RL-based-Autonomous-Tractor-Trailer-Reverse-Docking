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
│   ├── evaluation/             # Editable scenarios (metrics and runner are placeholders)
│   ├── training/               # PPO runs and campaigns, tests, bundles, CLI
│   ├── ui/                     # Control UI: parameters, training, tests, display
│   ├── visualization/          # Top-down view, MuJoCo viewer, offscreen 3D frames
│   ├── validation/             # Executable specification conformance report
│   └── controllers/            # Placeholders; NMPC plugs in here
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
- PPO training framework on the shared Gymnasium task: one control UI that
  sets truck dimensions, limits, dock size, physics, start ranges, and domain
  randomization, runs repeated training on the 2D or 3D simulator, tests each
  run, and exports the best policy as a portable bundle that can be loaded and
  run in a fresh simulator. It is a working pipeline, not a result: reward
  weights and start ranges are untuned and no run is registered as an
  experiment
- NMPC selectable in the same UI and command line through a single controller
  hook; the controller itself is not implemented yet
- Living IEEE-format LaTeX report updated through the Gymnasium implementation
- Simple closed-loop tracker and NMPC remain planned

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
.venv/bin/pip install -e '.[dev,mujoco,viz]'
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

The left side has three tabs (**Scenario**, **Training**, **Docking test**).
The right side is the display, with two tabs:

- **Top-down** draws the shared planar state, so it shows 2D and 3D runs alike.
- **3D view (MuJoCo)** shows the rendered 3D scene: a preview of the current
  scenario when idle, and the live episode while a 3D run is in progress.
  Untick **Render 3D runs** to train in 3D at full speed without pictures.

The **Scenario** tab edits a scenario: the frozen vehicle, dock, and task
specifications plus a set of overrides. **Apply** rebuilds the scene;
**Save scenario…** writes it to its own YAML file. The frozen specification
files are never modified.

| Group | Parameters | Affects |
|---|---|---|
| Dock | Interior width and depth | 2D and 3D |
| Route | Route planner (`entry` or `astar`); reward weights for staying on the route and aligning to it | 2D and 3D |
| Vehicle | Trailer variant; tractor wheelbase and body width; trailer hitch-to-axle length, rear overhang, and body width | 2D and 3D |
| Limits | Maximum steering angle and steering rate; reverse speed | 2D and 3D |
| Physics (3D only) | Tractor and trailer mass, ground friction | 3D only |
| Start pose | Standoff, lateral offset, trailer yaw, articulation ranges | 2D and 3D |
| Randomization | Trailer length drawn per episode; observation noise | 2D and 3D |

The **route planner** produces each episode's reference route from the start
pose to the dock. `entry` is the default: straight along the bay centreline,
joined through a point just outside the opening. `astar` is a basic
path-finding planner: an A* grid search that routes the trailer's docking
point around the dock walls. Both are geometric and do not check that the
truck can follow the route in reverse; further planners register in
`paths/planners.py` and then appear in this list. The reward already
penalizes leaving the route (squared lateral and heading deviation from it),
and the two **Reward** fields set how strongly; blank keeps the task file's
weights (0.5 and 0.2).

**Show planned route** above the display draws or hides the route in both the
top-down and the 3D view, before a run and while one is in progress.

A blank field keeps the base value; a blank randomization field leaves that
randomization off. **Apply** refuses a scenario that could never succeed or
that would start an episode already failed, and says why: for example a
trailer that cannot be completely inside the bay (the `long` variant needs a
bay deeper than the nominal 6 m), or a start range that begins in a wall.

### 8. Train and get a finished result — Available

On the **Training** tab choose:

| Setting | Meaning |
|---|---|
| Method | `ppo` trains a policy. `nmpc` has nothing to train, so it only runs the tests (available once the NMPC controller is implemented) |
| Simulator | `2d` (kinematic, fast) or `3d` (MuJoCo) for training |
| Number of runs | How many times to repeat training, with consecutive seeds |
| Timesteps per run | PPO budget of each run |
| Test episodes per run, 2D / 3D | After each run the final policy is tested on the ticked simulators, on the same start poses for every run |

Press **Start**. The display shows the episodes as they happen, and the status
panel reports the run in progress, recent success rate, and each finished
run's test results. **Stop** ends the work cleanly.

When the runs finish, the best one (highest test success rate, then mean
reward, on the first ticked simulator; the first run if none was tested) is
written as a single portable file,
`artifacts/<campaign-id>/policy_bundle.dockpolicy.zip`. **Export result…**
copies it wherever you choose. The bundle contains the policy and the scenario
it was trained on. Loading a bundle or policy file reads only its network
weights and plain-text metadata, never pickled objects, so a file received
from someone else cannot run code.

The same thing without the UI:

```bash
.venv/bin/python -m trailer_docking.training.cli train \
    --runs 3 --timesteps 200000 --backend 2d --eval-episodes 20 --eval-backends 2d,3d
```

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

_Not implemented: curriculum, parallel environments, resuming a run, and
TensorBoard._

### 9. Load a result and run it in a fresh simulator — Available

On the **Docking test** tab press **Load result…** and pick a bundle. The UI
adopts the scenario stored in it. Choose the simulator (`2d` or `3d`), the
number of episodes, and press **Run test**: a new simulator is built and the
policy drives it. A policy trained in 2D can be run in 3D and the reverse.

The same tab also tests the latest checkpoint, a raw `.zip` policy, the
untrained straight-reverse baseline, or the NMPC controller. Each episode is
listed with its outcome (success, collision, jackknife, out of bounds, time
limit) and final errors.

From the command line:

```bash
.venv/bin/python -m trailer_docking.training.cli test --bundle PATH/result.dockpolicy.zip --backend 3d --episodes 20
.venv/bin/python -m trailer_docking.training.cli test --episodes 20          # untrained baseline
.venv/bin/python -m trailer_docking.training.cli test --nmpc --episodes 20   # once NMPC exists
```

The NMPC baseline plugs in through one function; see the
[NMPC integration handoff](docs/design/nmpc-integration-handoff.md).

### What the demo currently does not show

The Gymnasium environment is functional and interactive, and the PPO pipeline
trains and tests policies on it in 2D and 3D, but NMPC is not implemented and
no trained policy has been evaluated as an experiment. The 2D `--demo` is a
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
