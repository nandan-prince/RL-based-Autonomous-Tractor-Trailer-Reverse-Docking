# Configuration

Configuration files are versioned experiment inputs, not convenient copies of
constants embedded in simulator or controller code.

## Implemented specifications

- `vehicles/compact_semitrailer_v1.yaml` is the machine-readable authority for
  vehicle geometry, variants, limits, timing, control fields, neutral semantic
  sites, MuJoCo physical defaults, and required MJCF names.
- `environments/nominal_dock_v1.yaml` is the machine-readable authority for the
  nominal loading bay, target pose, collision margin, and success tolerances.
- `environments/gymnasium_docking_v1.yaml` preserves the original absolute-error
  reward task.
- `environments/gymnasium_docking_v2.yaml` is the current PPO task and adds
  trailer-tail and heading improvement rewards with asymmetric deviation cost.
  Both task versions define workspace, timeout, preview distances, and
  decomposed reward weights.
- `vehicles/full_size_semitrailer_v1.yaml`,
  `environments/service_center_dock_v1.yaml`, and
  `environments/service_center_gymnasium_v1.yaml` define the optional
  full-size interactive 200 m by 100 m service-center yard with 20 doors and
  an open, color-highlighted target zone without side collision walls. They do
  not replace the compact controller-comparison baseline. The full-size
  vehicle uses a 35-degree soft warning and 60-degree terminal jackknife limit;
  the compact baseline retains its 35/45-degree limits.
- `controllers/baseline_controllers_v1.yaml` versions all implemented LQR and
  NMPC costs, horizons, tolerances, and iteration limits.
- `experiments/compact_docking_suite_v1.yaml` owns disjoint training,
  validation, and final-test seed ranges.
- `training/ppo_curriculum_v1.yaml` defines the initial three-stage PPO
  curriculum, its budgets, advancement gates, and resolved PPO settings.
- `training/ppo_curved_curriculum_v1.yaml` defines the shortened straight
  warm-up followed by gentle and mixed predefined-curve training stages.
- `training/ppo_curved_reward_study_v1.yaml` defines a controlled multi-seed
  PPO study in which only declared reward coefficients differ by condition.
- `scenarios/curved_docking_mixed_v1.yaml` is the standalone mixed-curve task
  using the original task reward.
- `scenarios/curved_docking_mixed_v2.yaml` selects the same curve/start
  distribution with the current dense tracking-improvement reward. Both are
  used for direct training, testing, and interactive inspection.

These use `schema_version: 1`. A schema version identifies the loader contract;
a `specification_id` identifies one parameter set. The strict loaders reject
unknown and missing keys, invalid units, non-finite values, and cross-field
inconsistencies. They also produce normalized SHA-256 fingerprints for future
experiment manifests.

The readable counterpart is the
[`vehicle-and-environment-specification.md`](../docs/design/vehicle-and-environment-specification.md)
document. When the YAML and prose disagree, stop and resolve both in one change.

## Remaining configuration work

Additional files will cover:

- MuJoCo-only gains, damping, contact, and solver tuning, or an equivalent
  resolved-tuning block in every experiment manifest
- additional experiment matrices beyond the compact version-1 suite
- policy-network architecture variants if they become a justified study

`experiments/` is reserved for immutable, named experiment configurations. See
its README for the evidence contract.

The current `MujocoTuning` values are versioned in Python and documented in
research decision D-007, but are not included in the vehicle YAML fingerprint.
Do not treat that fingerprint alone as complete MuJoCo experiment provenance.

## Change rules

- Load values through `trailer_docking.core.config`; do not create a second
  parser in a simulator.
- A frozen cross-simulator change requires coordinated YAML, design document,
  2D tests, MuJoCo parity tests, and report updates.
- Use a new specification ID for a new vehicle or dock parameter set. Increment
  the schema only when the software contract changes.
- Never silently edit a configuration already referenced by reported results;
  create a new version and retain the original fingerprint.

Validate the current pair with:

```bash
.venv/bin/python -m trailer_docking.validation.vehicle_spec
```
