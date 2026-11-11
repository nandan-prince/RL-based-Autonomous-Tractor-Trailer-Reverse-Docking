# Trajectory-Aware Reinforcement Learning for Autonomous Tractor–Trailer Reverse Docking

**MSML642 Project Proposal**
**Team Members:** Nandan Prince and Chris Pedretti

## 1. Project Goal

Reversing a tractor–trailer into a loading dock is difficult because the articulated vehicle is nonlinear, nonholonomic, and unstable while moving backward. Small steering errors can rapidly increase the articulation angle, causing poor alignment, collision, or jackknifing. This project will develop and evaluate a simulated autonomous system that places a semitrailer within a selected loading dock while respecting steering, collision, and articulation constraints.

The central research question is: **Can a Proximal Policy Optimization (PPO) controller using trajectory-relative preview information reliably reverse a tractor–trailer into a dock, and how does it compare with a model-based controller under randomized starting poses, vehicle parameters, and observation noise?**

The project will produce a verified low-speed vehicle model, a reference docking path, a conventional controller, a trained PPO policy, and a common evaluation suite. Performance will be measured using docking success rate, collision and jackknife rates, final position and heading error, maximum articulation angle, steering smoothness, maneuver time, and controller computation time. The target is at least 90% success on nominal held-out scenarios, no collisions in successful trials, and one reproducible three-dimensional docking demonstration.

## 2. Simulation Platform

Development will begin in a lightweight custom **two-dimensional Gymnasium environment** so vehicle dynamics, rewards, safety checks, and training can run quickly and reproducibly. The vehicle will use a low-speed kinematic bicycle model for the tractor coupled to a kinematic semitrailer model. The final demonstration will use **MuJoCo**, with separate tractor and trailer bodies, a hitch yaw joint, front-wheel steering, collision geometry, a loading-dock scene, and the same controller interface used in 2D.

MuJoCo is appropriate because the project emphasizes reinforcement learning and control rather than a complete ROS autonomy stack. Simulator ground-truth pose will be used for the core experiments; controlled noise will later test robustness. Camera-based dock recognition and LiDAR mapping are stretch goals and will not delay the controller comparison.

## 3. ML/RL Method

The principal learning method will be **PPO**, an on-policy actor–critic algorithm. PPO is suitable because the steering command is continuous, its clipped policy update is comparatively stable, and it has mature implementations. The policy will solve local reverse trajectory tracking, while path generation and safety validation remain explicit modules. This hybrid design reduces the learning burden and makes failures easier to diagnose than an end-to-end sensor-to-steering policy.

A nonlinear model predictive controller (**NMPC**) implemented with CasADi/IPOPT is the preferred conventional baseline. It will minimize path error, articulation, steering effort, and steering-rate variation while enforcing vehicle limits. If NMPC proves too time-consuming, a geometric feedback or LQR trajectory tracker will serve as the baseline. Both controllers will receive the same state estimates, reference paths, initial conditions, and evaluation scenarios.

## 4. Programming Language and Tools

- **Language:** Python 3
- **RL/environment:** Gymnasium, Stable-Baselines3, and PyTorch
- **Simulation/analysis:** MuJoCo, NumPy, SciPy, and Matplotlib
- **Planning/control:** CasADi/IPOPT; Reeds–Shepp or Hybrid A* if time permits
- **Engineering:** Git/GitHub, pytest, YAML configuration, and TensorBoard



## 5. Project Description and Technical Approach

### Robot, task, and simulation setup

The simulated robot consists of one tractor and one conventional semitrailer operating at low speed on level ground. Each episode begins from a randomized but feasible approach pose in a static logistics-yard map. The target is a known dock pose. The system must reverse the trailer into the bay within specified position and orientation tolerances while avoiding dock walls and obstacles, respecting steering limits, and preventing excessive articulation.

```
Vehicle state + target dock + static map
                    |
                    v
         Reference-path generator
                    |
                    v
          PPO policy / NMPC baseline
                    |
                    v
             Safety supervisor
                    |
                    v
        Tractor–trailer simulation
```

Development will start with predefined straight and curved docking paths. If time permits, Reeds–Shepp curves or Hybrid A* with tractor–trailer state propagation will generate start-to-dock references. Every generated path will be simulated and collision-checked because a feasible car-like path does not necessarily produce a safe trailer-articulation history.

### State, observation, action, and reward

The simulator state will contain tractor position and heading, trailer heading, articulation angle, steering angle, and longitudinal velocity. The PPO observation will use normalized path-relative quantities: trailer lateral error, tractor and trailer heading errors, articulation angle and rate, steering state, distance to the goal, and several local preview points or path-curvature samples. This representation should generalize across dock locations better than raw global coordinates.

The initial action will be a bounded steering-rate command at a fixed low reverse speed. If training is stable, longitudinal velocity may be added as a second action. The reward will combine progress along the reverse reference, small lateral and heading errors, low articulation, smooth steering, and a terminal docking bonus. Collision, jackknife, leaving the operating area, and timeout will produce penalties and episode termination. A separate safety supervisor will enforce hard steering and articulation limits so safety does not depend only on reward tuning.

### How the ML/RL method will be used

PPO will first learn straight reverse stabilization, then curved-path tracking, and finally randomized docking through curriculum learning. Domain randomization will vary the starting pose and selected vehicle parameters. The trained policy and baseline will be tested on identical held-out seeds. At least 100 trials per nominal condition will be used where computation permits. Planned comparisons are PPO versus the classical controller, global-coordinate versus trajectory-relative observations, and perfect versus noisy state measurements. Failed trials will be retained and categorized.

### Why this approach is appropriate

Tabular Q-learning is unsuitable because the state and action spaces are continuous and high-dimensional. DQN is principally designed for discrete actions, while direct end-to-end visual control would require substantially more data and obscure the source of failures. PPO offers a practical continuous-control method, while NMPC provides an interpretable model-based benchmark. Separating path generation from tracking keeps the project achievable and enables a fair controller comparison.



## 6. Project Plan

| Week | Planned outcome |
|---|---|
| 1 | Freeze geometry, interfaces, safety thresholds, and success criteria; validate straight and constant-steering motion. |
| 2 | Complete collision/jackknife checks, fixed paths, the Gymnasium environment, and a simple classical controller. |
| 3 | Train PPO on fixed paths; complete NMPC or its fallback; run the first held-out 2D comparison. |
| 4 | Build the MuJoCo vehicle and dock scene; connect both controllers through a shared adapter. |
| 5 | Freeze features and run nominal plus one robustness experiment with fixed seeds and recorded metrics. |
| 6 | Analyze failures, reproduce runs, finalize plots/video, write the report, and prepare the presentation. |

Nandan Prince will lead vehicle modeling, path generation, collision checking, and baseline control. Chris Pedretti will lead the Gymnasium environment, PPO training, MuJoCo integration, and experiment automation. Both members will define interfaces, review each other's work, integrate weekly, run final experiments, and write the report jointly.

## 7. Fallback Plan

The minimum viable fallback will preserve the research question while reducing integration risk. If automatic planning is not ready by the end of Week 2, the controllers will track a small library of predefined, collision-checked docking paths. If NMPC is unreliable, it will be replaced by a documented geometric feedback or LQR controller. If PPO does not learn the full randomized task by the end of Week 3, training will be restricted to fixed reverse speed, a narrower initial-state distribution, and a straight-to-curved curriculum. If 2D-to-MuJoCo transfer remains unreliable, the quantitative comparison will remain in the verified 2D simulator and MuJoCo will show one validated baseline or scripted docking maneuver.

Perception, OCR, LiDAR mapping, dynamic obstacles, imitation learning, detailed 3D assets, and multi-dock routing will be removed before any core item. The verified vehicle model, collision and jackknife checks, one conventional baseline, one PPO policy, held-out quantitative evaluation, and a complete reverse-docking demonstration will remain required.

## 8. Recent Research and Relationship to This Project

1. **Yan et al. (2024), “Trajectory State Model-based Reinforcement Learning for Truck-trailer Reverse Driving.”** This work combines trajectory-relative states and PPO for reverse path following, directly motivating our preview-based policy observation. Our project builds on it by defining a reproducible docking task with collision/jackknife checks, a held-out robustness protocol, and transfer from a fast 2D environment to MuJoCo.
2. **Moradi and Mehrandezh (2022), “A Non-linear MPC Local Planner for Tractor-Trailer Vehicles in Forward and Backward Maneuvering.”** This paper demonstrates constrained NMPC for trajectory tracking, obstacle avoidance, and jackknife prevention. It informs our baseline and staged 2D-to-3D workflow; our work differs by directly comparing NMPC or a classical substitute with trajectory-aware PPO under identical randomized docking conditions.
3. **Ma et al. (2025), “A Hierarchical Approach for Tractor-trailer Motion Planning Using Graph Search and Reinforcement Learning.”** This paper separates high-level route selection from learned local trajectory generation. It supports our modular architecture, but our project focuses on reverse loading-dock execution, controller comparison, and observation design rather than long-horizon yard routing.

## 9. References

[1] H. Yan et al., “Trajectory State Model-based Reinforcement Learning for Truck-trailer Reverse Driving,” *2024 7th International Conference on Robotics, Control and Automation (ICRCA)*, 2024. doi: 10.1109/ICRCA60878.2024.10649133.

[2] B. Moradi and M. Mehrandezh, “A Non-linear MPC Local Planner for Tractor-Trailer Vehicles in Forward and Backward Maneuvering,” *arXiv:2212.11427*, 2022.

[3] H. Ma et al., “A Hierarchical Approach for Tractor-trailer Motion Planning Using Graph Search and Reinforcement Learning,” *European Control Conference (ECC)*, 2025.
