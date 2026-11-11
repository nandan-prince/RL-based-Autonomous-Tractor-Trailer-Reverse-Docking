# Configuration

Configuration files are versioned experiment inputs, not convenient copies of
constants embedded in simulator or controller code.

## Implemented specifications

- `vehicles/compact_semitrailer_v1.yaml` is the machine-readable authority for
  vehicle geometry, variants, limits, timing, control fields, neutral semantic
  sites, MuJoCo physical defaults, and required MJCF names.
- `environments/nominal_dock_v1.yaml` is the machine-readable authority for the
  nominal loading bay, target pose, collision margin, and success tolerances.

Both use `schema_version: 1`. A schema version identifies the loader contract;
a `specification_id` identifies one parameter set. The strict loaders reject
unknown and missing keys, invalid units, non-finite values, and cross-field
inconsistencies. They also produce normalized SHA-256 fingerprints for future
experiment manifests.

The readable counterpart is the
[`vehicle-and-environment-specification.md`](../docs/design/vehicle-and-environment-specification.md)
document. When the YAML and prose disagree, stop and resolve both in one change.

## Planned configuration groups

Additional files will cover:

- vehicle geometry and actuator limits
- 2D integration and controller timing
- dock dimensions, obstacles, and success tolerances
- Gymnasium reward and termination settings
- NMPC horizon, costs, constraints, and solver settings
- PPO training and curriculum settings
- held-out experiment matrices and random seeds

`experiments/` is reserved for immutable, named experiment configurations. See
its README for the evidence contract.

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
