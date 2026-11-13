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
- Living IEEE-format LaTeX report updated through the Gymnasium implementation
- Simple closed-loop tracker, NMPC, and PPO remain planned

## Vehicle-core verification

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev,mujoco,viz]'
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

### 6. Adjust parameters in a UI — Planned

A control panel for dock dimensions, vehicle size and mass, and ground and
contact physics, saving each edited scenario as a new named configuration.

_To be documented: how to open the panel, which parameters affect both
simulators and which affect only the 3D physics, and where saved
configurations are written._

### 7. Train a docking policy and watch it learn — Planned

Start PPO training from the UI and watch episodes play back in the viewers as
the policy improves.

_To be documented: the training command and UI control, where checkpoints and
metrics are stored, and how to resume or stop a run._

### 8. Run a docking test with a trained policy — Planned

Choose a scenario and a trained policy, run a simulated docking attempt in 2D
or 3D, and read the outcome (success, final error, collisions, jackknife).

_To be documented: selecting a checkpoint, running a single test or a batch,
and interpreting the reported metrics._

### What the demo currently does not show

The Gymnasium environment is functional and interactive, but NMPC and PPO are
not implemented, so no learned or optimizing controller is shown yet. The 2D
`--demo` is a constant reverse command used to validate the complete success
path; it is not a controller-performance result.

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
