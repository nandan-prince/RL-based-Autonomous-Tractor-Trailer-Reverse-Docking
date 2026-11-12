# Academic Research Index

This directory tracks the academic foundation for the autonomous tractor–trailer reverse-docking project. The initial reading set covers trajectory-aware reinforcement learning, nonlinear model predictive control, hierarchical motion planning, and imitation/offline reinforcement learning.

Project-level research records:

- [Engineering and research decisions](decisions.md)
- [Open research questions](open_questions.md)

## Core papers

| Paper | Main contribution | Project use |
|---|---|---|
| [Trajectory State Model-based Reinforcement Learning for Truck-trailer Reverse Driving](https://doi.org/10.1109/ICRCA60878.2024.10649133) | Combines a kinematic trajectory-state model, preview errors, and PPO for reverse path following. | Motivates the path-relative RL observation and PPO controller. |
| [A Non-linear MPC Local Planner for Tractor-Trailer Vehicles in Forward and Backward Maneuvering](https://arxiv.org/abs/2212.11427) | Uses constrained NMPC for tracking, static-obstacle avoidance, and jackknife prevention; demonstrates staged integration into Gazebo. | Guides the conventional baseline, constraints, and 2D-to-3D development process. |
| [A Hierarchical Approach for Tractor-trailer Motion Planning Using Graph Search and Reinforcement Learning](https://www.merl.com/publications/TR2025-093) | Combines reference-line graph search with constrained RL trajectory generation for long-horizon planning. | Supports separation of high-level routing from local trajectory execution. |
| [Edge-based Imitation Learning for Autonomous Tasks: A Case Study on Tractor-Trailer Positioning](https://doi.org/10.1515/auto-2025-0081) | Compares behavioral cloning, IQL, and TD3-BC using demonstrations from a PPO expert. | Provides a possible post-core extension using expert demonstrations. |

## Additional directly relevant work

- [Motion Planning and Model Predictive Control for Automated Tractor-Trailer Hitching Maneuver](https://www.merl.com/publications/TR2022-109) — integrated search-based planning and NMPC for a precision heavy-vehicle maneuver.
- [Learning-Based Motion Planning of a General 3-Trailer System](https://www.merl.com/publications/TR2022-133) — uses reinforcement learning to improve the planning heuristic for articulated vehicles.
- [Fast Adaptive Planning for Autonomous Tractor-Trailer Systems via Iterative LQR Steering](https://www.merl.com/publications/TR2026-118) — addresses positioning accuracy and uncertain system parameters with adaptive iLQR steering.
- [Lattice-based Motion Planning for a General 2-Trailer System](https://doi.org/10.1109/IVS.2017.7995820) — a key reference for dynamically feasible planning with articulated vehicles.
- [Minimum Swept-Path Control for Autonomous Reversing of a Tractor Semi-Trailer](https://doi.org/10.1109/TVT.2019.2895513) — focuses on reverse control and swept-path minimization.

## Reading questions

For every paper, record:

1. What vehicle geometry and low-speed assumptions are used?
2. Is the work solving global planning, local planning, trajectory tracking, or all three?
3. What information appears in the controller or policy state?
4. What are the control actions: steering, steering rate, velocity, or acceleration?
5. How are collision and jackknifing handled: reward, constraint, termination, or supervisor?
6. Which results are quantitative, and what baselines are used?
7. Is validation limited to kinematic simulation, or does it include a physics simulator or physical vehicle?
8. Which details are missing for reproduction?

## Planned literature outputs

- One-page summary for each core paper
- Shared notation and vehicle-geometry glossary
- Comparison table covering state, action, reward/cost, constraints, simulator, and metrics
- Final rationale for the project’s observation, action, baseline, and evaluation choices

## Current project decisions

The first comparison uses the same low-speed on-axle vehicle contract for both
controllers: steering rate is controller-selected, target speed is fixed at
`-0.20 m/s`, and steering angle remains part of state. Vehicle/dock parameters
and required MuJoCo names are versioned in YAML. See the
[decision log](decisions.md) and frozen
[vehicle/environment specification](../docs/design/vehicle-and-environment-specification.md)
before translating a paper's notation into code.

The first MuJoCo vehicle, dock, and shared-state adapter are implemented and
audited against that contract. This is engineering validation, not a controller
result or claim of real-vehicle fidelity. Direct 2D/MuJoCo parity remains
pending; see the [alignment audit](../docs/design/mujoco-alignment-audit.md).

## PDF policy

The `papers/` directory contains local study copies and its PDFs are currently
tracked in this working repository. Tracking is not evidence of redistribution
permission. Before publishing or sharing the repository, audit every file's
license and remove any copy that cannot legally be redistributed. Prefer the
stable links above in reports and shared documentation.
