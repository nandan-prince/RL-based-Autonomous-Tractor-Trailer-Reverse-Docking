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
├── scripts/                    # Training, evaluation, and rendering entry points
├── src/trailer_docking/
│   ├── core/                   # Shared state, dynamics, geometry, and configuration
│   ├── paths/                  # Reference paths, tracking errors, and previews
│   ├── observations/           # Controller-facing observation construction
│   ├── safety/                 # Collision, jackknife, and command supervision
│   ├── simulators/
│   │   ├── kinematic/          # Fast deterministic 2D backend
│   │   └── mujoco/             # MuJoCo adapter, models, scenes, and assets
│   ├── environments/           # Gymnasium task wrapper
│   ├── controllers/            # Geometric, NMPC, and PPO controllers
│   └── evaluation/             # Scenarios, metrics, and rollout runner
└── tests/                      # Unit and integration tests
```

Start with the [architecture document](docs/design/architecture.md) for module
responsibilities and implementation order. A 2D contributor should then read
the [vehicle and environment specification](docs/design/vehicle-and-environment-specification.md).
A 3D contributor should use the self-contained
[MuJoCo contributor guide](src/trailer_docking/simulators/mujoco/README.md).

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
  specification's kinematic equations, not yet against the 2D simulator
- Local 3D and top-down viewers available (see below)
- Skeleton of the learning pipeline: planner interface with a fixed
  docking path, path-relative observation, Gymnasium environment, PPO
  training and docking-test runner, and a control UI; interfaces are in
  place, reward and planner are placeholders, no trained result yet
- Living IEEE-format LaTeX report initialized from verified project state
- Kinematic state propagation and simulator implementation not yet started

## Vehicle-core verification

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/ruff check src tests
.venv/bin/mypy
.venv/bin/pytest --cov --cov-report=term-missing
.venv/bin/python -m trailer_docking.validation.vehicle_spec
```

The conformance command loads the versioned YAML specifications, validates all
fields and cross-field constraints, reconstructs the neutral assembly, checks
rigid geometry and dock fit, and prints configuration fingerprints suitable for
future experiment manifests.

For MuJoCo development, install the additional maintained Python bindings:

```bash
.venv/bin/pip install -e '.[dev,mujoco]'
```

The optional dependency currently admits MuJoCo 3.14 releases through the next
major version. See the MuJoCo contributor guide before creating or renaming any
MJCF element.

## Demo

This section is the walkthrough for seeing the project run. Each numbered step
is marked **Available** or **Planned**; planned steps are placeholders to be
filled in as the reinforcement-learning pipeline and the control UI are built.
Keep the step order and update the status and commands in place rather than
adding a separate guide.

### Setup (once)

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev,mujoco,viz]'
```

On Windows use `py -3.12 -m venv .venv` and replace `.venv/bin/` with
`.venv\Scripts\` in every command below. Run all commands from the repository
root.

### 1. See the generated dock and vehicle — Available

```bash
.venv/bin/python -m trailer_docking.visualization.viewer --mode both
```

Two windows open, driven by one simulation:

- **3D (MuJoCo):** the tractor, trailer, and loading bay generated from the
  YAML specifications.
- **2D (top-down):** the same scene drawn only from the shared planar state a
  controller receives, with speed, steering, articulation, and docking error.

Use `--mode 3d` or `--mode 2d` for one window only.

### 2. Drive it yourself — Available

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

### 3. Watch a scripted reverse — Available

```bash
.venv/bin/python -m trailer_docking.visualization.viewer --mode both --demo --speed 4
```

The vehicle reverses straight into the bay and stops when the trailer touches
the back wall. This is a fixed command, not a controller.

### 4. Change the scenario from the command line — Available

| Option | Effect |
|---|---|
| `--variant short\|nominal\|long` | Trailer hitch-to-axle length of 2, 4, or 5 m |
| `--no-dock` | Empty yard without the loading bay |
| `--speed N` | Play N times faster than real time |
| `--vehicle PATH`, `--dock PATH` | Load a different vehicle or dock YAML |

Any vehicle or dock YAML that passes validation can be viewed this way; copy
one of the files under `configs/`, change a value such as the bay width, and
pass the new path.

### 5. See the planned route — Available (skeleton planner)

Every docking episode starts with a path planner producing a reference route
for the trailer axle from the start pose to the dock; the agent's job is to
follow it. The viewer from steps 1–3 draws that route as a green line in both
windows. Press `P` to show or hide it.

The current planner is a fixed curve-then-straight path. It does not yet check
that the trailer can actually follow the curve or that it clears obstacles;
Reeds–Shepp or Hybrid A* planners can replace it behind the same interface.

### 6. Adjust parameters in the control UI — Available (skeleton)

```bash
.venv/bin/pip install -e '.[dev,mujoco,viz,rl,train]'
.venv/bin/python -m trailer_docking.ui.app
```

The **Scenario** tab edits a scenario: the frozen vehicle and dock
specifications plus a small set of overrides. **Apply** rebuilds the scene;
**Save scenario…** writes the scenario to its own YAML file. The frozen
specification files are never modified.

| Group | Parameters | Affects |
|---|---|---|
| Dock | Interior width and depth | 2D and 3D |
| Vehicle | Trailer variant | 2D and 3D |
| Start pose | Standoff, lateral offset, trailer yaw, articulation ranges | 2D and 3D |
| Physics (3D only) | Tractor and trailer mass, ground friction | MuJoCo only |

The live top-down view on the right shows the current scene, and **Show
planned route** toggles the route.

_To be expanded: more dock and vehicle dimensions, obstacle placement, and
contact and actuator settings._

### 7. Train a docking policy and watch it learn — Available (skeleton)

In the **Training** tab set a timestep budget and press **Start training**.
PPO trains in the background while the top-down view shows the training
episodes as they happen: early on the truck jackknifes, leaves the route, or
hits the dock, and the status panel reports the recent success rate. **Stop**
ends the run cleanly and keeps the latest checkpoint.

The same run without the UI:

```bash
.venv/bin/python -m trailer_docking.training.cli train --timesteps 200000
```

Each run gets an experiment ID and writes `results/<id>/manifest.yaml` and
`metrics.csv` (tracked) and `artifacts/<id>/checkpoints/` (ignored).

**Read this before interpreting anything:** the reward weights, preview
spacing, and observation scaling are placeholders, and the planner is
unverified. A short run does not learn to dock. Training currently runs
against the MuJoCo backend because the 2D simulator does not exist yet.

_To be expanded: curriculum (straight, then curved, then randomized),
domain randomization, parallel environments, resuming a run, and TensorBoard._

### 8. Run a docking test — Available (skeleton)

In the **Docking test** tab choose the latest checkpoint, a saved `.zip`
policy, or the untrained straight-reverse baseline, set the number of
episodes, and press **Run test**. The episodes play in the view and each one
is listed with its outcome (docked, jackknife, collision, off path, timeout)
and final errors.

From the command line:

```bash
.venv/bin/python -m trailer_docking.training.cli test --episodes 20
.venv/bin/python -m trailer_docking.training.cli test --policy PATH/policy_final.zip --episodes 20
```

_To be expanded: NMPC as a second controller on the same route, held-out
scenario sets, 3D playback of a test, and saved evaluation reports._

### What the demo currently does not show

The 2D kinematic simulator and NMPC are not implemented yet, so every step
uses the MuJoCo backend for motion. The RL pieces are a working skeleton, not
a result: no policy in this repository has been shown to dock reliably.

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
