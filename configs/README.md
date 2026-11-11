# Configuration

Configuration files will hold versioned parameters rather than embedding them
in simulator or controller code. Planned groups are:

- vehicle geometry and actuator limits
- 2D integration and controller timing
- dock dimensions, obstacles, and success tolerances
- Gymnasium reward and termination settings
- NMPC horizon, costs, constraints, and solver settings
- PPO training and curriculum settings
- held-out experiment matrices and random seeds

`experiments/` is reserved for named, reproducible experiment configurations.
Defaults will be added only after the mathematical specification is frozen.
