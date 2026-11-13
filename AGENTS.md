# AGENTS.md — Research-Aware Development Instructions

## 1. Purpose and working philosophy

This repository contains an evolving robotics research and engineering project on articulated tractor–trailer maneuvering in a simulated transportation/service-center yard. It is simultaneously:

1. A working robotics simulation and experimentation codebase.
2. A university project whose methods, implementation, and findings must be documented in a submission-ready technical report.
3. A potential foundation for a peer-reviewed research publication **if** the results warrant one.

**The research direction must emerge from implemented work and credible experiments. Do not invent a contribution, select a favorable conclusion, or force the code to fit a paper narrative.**

The core study is an empirical comparison of trajectory-aware Proximal Policy Optimization (PPO) and nonlinear model predictive control (NMPC) for low-speed tractor–semitrailer reverse trajectory tracking and loading-dock parking. NMPC is the preferred model-based baseline; a geometric or LQR tracker is only a documented fallback if NMPC cannot be made reliable within the project schedule. These methods are the agreed project scope, not a claim that either must win. Explore extensions only as the implementation, literature, or experimental evidence justifies them, and discuss major scope changes with the user.

The immediate priority is a **correct, functioning, testable simulation and experimental pipeline**. Maintain the paper in parallel, but do not let manuscript polishing delay core engineering work.

## 2. Current project context and authority

- **Core task:** reverse one tractor with one conventional semitrailer into a known loading bay in a static logistics-yard map. The trailer must reach a defined docking pose without collision or jackknifing and while respecting steering and articulation limits.
- **Vehicle:** the initial cross-simulator vehicle is `compact-semitrailer-v1`, an explicit, configurable research-scale geometry with a nominal on-axle hitch. The signed hitch offset and trailer length remain configurable for later experiments.
- **Phase 1:** implement and validate a fast 2D kinematic simulator, then expose the task through a Gymnasium-compatible environment for debugging, NMPC evaluation, PPO training, and quantitative experiments.
- **Phase 2:** build the corresponding 3D MuJoCo vehicle and loading-dock scene with the agreed bodies, semantic sites, joints, actuators, collision geometry, and physical parameters. The core study uses simulator ground-truth state. Cameras, LiDAR, mapping, OCR, and learned perception are stretch goals and must not delay controller evaluation.
- **Control interface:** the initial shared action is bounded steering rate at a fixed low reverse-speed target. The shared command retains a target-speed field for NMPC, scripted tests, and later extensions, but method-specific changes to action authority require an explicit experimental decision.
- **Path structure:** begin with predefined straight and curved docking references. Reeds–Shepp, Hybrid A*, reference-line graph search, and other automatic planners are optional extensions, not critical-path requirements.
- **Control methods:** trajectory-relative PPO and NMPC receive the same vehicle state contract, reference paths, plant limits, scenarios, safety definitions, and evaluation metrics as far as methodologically reasonable.
- **Experiments:** the planned study varies start pose, trailer hitch-to-axle length, parking-bay dimensions, clearance, and observation/model mismatch only after nominal controllers work. Geometry changes are robustness/generalization studies unless a component is deliberately removed as an ablation.
- **Fallback:** the complete quantitative PPO-versus-NMPC comparison must succeed in the verified 2D environment. If 3D transfer is delayed, MuJoCo may be limited to validated demonstrations or a smaller test set, as stated in the proposal.
- **Final output:** working code, repeatable experiments, saved results and figures, a submission-ready LaTeX course report, and a possible research manuscript if the evidence warrants one.
- **Parallel development:** the 2D vehicle core, Gymnasium environment, NMPC, and PPO work may proceed in parallel with the teammate's MuJoCo vehicle and scene, but both sides must conform to the frozen shared specification and parity checks.

Project authority, from highest to lowest, is:

1. The user's latest explicit instructions.
2. The submitted proposal and explicitly agreed scope changes.
3. Frozen, versioned design contracts under `docs/design/` and their machine-readable configurations under `configs/`.
4. This file's process, rigor, documentation, and traceability rules.

`AGENTS.md` does not independently broaden the project. Check the repository and the user's latest instructions before assuming an item is implemented or mandatory. A specification marked frozen remains authoritative for parallel work until it is deliberately revised with a new version or documented compatibility change.

The current cross-team contract is:

- `docs/design/vehicle-and-environment-specification.md`
- `configs/vehicles/compact_semitrailer_v1.yaml`
- `configs/environments/nominal_dock_v1.yaml`

The specification fixes coordinate/sign conventions, semantic reference frames, nominal geometry, state and command contracts, limits, timing, docking success, required MuJoCo names, and parity checks. Initial MuJoCo mass, inertia, contact, damping, and control gains are modeling choices that must be recorded and validated; a visually realistic model is not evidence of physical accuracy.

## 3. Standing rule: implementation and manuscript evolve together

For every substantive development task, assess whether the work changes the scientific description of the project. If it does, update the relevant paper source and research notes **in the same task**, after verifying the implementation as far as feasible. Until the LaTeX manuscript scaffold exists, update the applicable design document and research record instead, and backfill the verified material into the manuscript when it is created.

Examples of changes that usually require documentation:

- Adding or changing the vehicle geometry, kinematic/dynamic equations, hitch model, collision geometry, tires, contact parameters, or actuator mapping.
- Creating or changing 2D/3D environments, reward functions, observations, actions, resets, goals, termination conditions, or scenario distributions.
- Implementing or tuning PPO, NMPC, numerical solvers, trajectory generation, safety constraints, or control frequencies.
- Introducing new metrics, experiments, scenarios, baselines, ablations, robustness tests, or statistical analyses.
- Discovering reproducible failures, limitations, discrepancies between 2D and 3D, or unexpected behavior.
- Finding credible related literature that alters the project's novelty assessment or interpretation.

For routine refactoring, formatting, dependency updates, or bug fixes that do not materially change methods or findings, **do not pad the manuscript with noise**. Update a development/research log when the change has implications for reproducibility or interpretation.

At the end of a substantive task, briefly summarize:

1. What changed in the code.
2. What was tested, with outcomes and remaining uncertainty.
3. Which manuscript/research files were updated and why.
4. What evidence or experiments are still missing.

Never claim that the paper or Overleaf was updated unless the corresponding local files or actual remote project were changed.

## 4. Paper source and Overleaf workflow

Maintain the **Overleaf-compatible IEEE-format LaTeX manuscript** under `paper/`. It is the living course report and may later evolve into a research manuscript if the evidence warrants it. The current generic `IEEEtran` conference format does not imply selection of a particular IEEE venue. The directory contains, at minimum:

```text
paper/
  main.tex
  references.bib
  figures/
```

As the paper grows, optionally split it into `sections/` (introduction, related work, problem formulation, simulation setup, methods, experimental design, results, discussion, limitations, conclusions). Use `\input{}` where appropriate. Do not restructure an existing working manuscript without a good reason.

- **Treat the repository LaTeX files as the default source of truth** for Codex-authored manuscript changes. Preserve user edits already incorporated into the repository.
- Keep the LaTeX buildable in a normal Overleaf project. Avoid obscure packages, absolute file paths, local-only assets, or dependencies that Overleaf cannot resolve.
- Keep the main entrypoint obvious (`paper/main.tex` unless an existing convention takes precedence).
- Write precise, restrained, academic prose; the report should be readable even if no novel method is discovered.
- Use figures and tables generated from reproducible experimental outputs; keep publication figures under `paper/figures/` or an agreed synced path. Avoid copying large raw artifacts into the paper directory.
- Compile/check LaTeX when a local compiler exists. If unavailable, inspect the source for obvious syntax, missing references, missing figures, and broken paths; report that compilation was not verified.
- Do not change an existing class/template or target a specific publisher without user direction. A neutral technical-report format is acceptable initially.
- Do not hardcode a paper title asserting novelty, superiority, or state-of-the-art results.
- Treat the course report as the immediate deliverable and a publication manuscript as a possible evolution. Do not let speculative publication framing replace proposal commitments.

**Important:** Editing repository files does not itself synchronize Overleaf. The user will configure the Overleaf/GitHub or Overleaf Git workflow. Do not assume synchronization is automatic, do not overwrite remote edits, and do not push to external repositories/Overleaf without explicit user permission and configured access. If GitHub sync is used, remember that synchronization may require explicit push/pull in Overleaf.

## 5. Living manuscript: what to write at each stage

The paper is an evidence-based living document. Maintain sections progressively:

- **Introduction / motivation:** the operational problem, why articulated maneuvering matters, and the current questions under investigation. Do not pre-claim a unique contribution.
- **Related work:** independently verified literature on tractor–trailer modeling, reverse maneuvering/docking, MPC/NMPC, RL/PPO, and simulation/benchmarking. Identify overlap and gaps without unsupported novelty claims.
- **Problem formulation:** coordinate frames, state/action definitions, trailer goal pose, geometry, constraints, collision conditions, objective, assumptions, and notation—only when actually defined.
- **Simulation/modeling:** implemented 2D and 3D environments, vehicle specifications, dynamics, simulator settings, sensor/control interfaces, validation tests, and known approximations.
- **Methods:** controllers implemented, model information available to each controller, training/optimization settings, constraints, solver details, and tuning procedures.
- **Experimental protocol:** scenario generation, train/validation/test separation when applicable, seeds, baselines, metrics, hardware/software, evaluation conditions, and statistical methodology.
- **Results:** actual reproducible data, figures, tables, and clearly sourced interpretation. Leave placeholders if experiments are incomplete.
- **Discussion / limitations:** failure modes, modeling assumptions, threats to validity, computation/training cost, simulation-to-simulation gap, and remaining questions.
- **Conclusion:** write substantive conclusions only after results exist.

Use clearly marked `TODO` or `Planned experiment` placeholders for unimplemented items; distinguish **implemented**, **tested**, **observed**, **hypothesized**, and **planned** throughout.

Because this is also a class report, ensure that the manuscript documents what was actually built—even if comparisons are inconclusive or no publishable novelty emerges.

## 6. Research records and traceability

Maintain lightweight research records under the existing `research/` directory. Add files when their first real entry exists rather than creating empty bureaucracy. The intended records are:

```text
research/
  decisions.md
  experiment_log.md
  open_questions.md
```

The existing `research/README.md` is the literature index, so do not duplicate it with `literature.md`. Use these records to capture exploratory details before placing polished, verified claims in the manuscript.

### 6.1 Experiment and result storage

Use an immutable experiment ID for every significant training or evaluation run. A recommended ID is `YYYYMMDD-HHMM_<method>_<short-purpose>`, optionally followed by a repeat or seed identifier. Do not reuse an ID for a materially different run.

Store the reproducible definition under:

```text
configs/experiments/<experiment-id>.yaml
```

Store compact, publication-relevant outputs under:

```text
results/<experiment-id>/
  manifest.yaml          # provenance, versions, command, status, and paths
  metrics.csv            # per-episode or aggregated metrics when reasonably sized
  summary.json           # machine-readable headline results
  notes.md               # failures, validity, interpretation, and caveats
  figures/               # reproducibly generated experiment figures
```

Store large or regenerable artifacts under ignored paths such as:

```text
artifacts/<experiment-id>/
  checkpoints/
  trajectories/
  tensorboard/
  videos/
```

The current ignored `runs/`, `checkpoints/`, and `wandb/` paths may continue to be used during exploration, but a result is not report-ready until its configuration, provenance, status, and relevant metrics are copied or generated into the tracked experiment/result structure. Do not commit large checkpoints or raw rollouts merely to satisfy traceability; record their stable local or external path and checksum where practical.

Never overwrite a completed result directory. Mark invalid runs as invalid in their manifest and notes rather than deleting them when they provide useful diagnostic history. Exploratory smoke tests need not receive full result packages unless they reveal a meaningful failure or influence a design decision.

For every significant experiment, record as many of the following as are relevant and available:

- Experiment ID, date, purpose, hypothesis or research question.
- Git commit/hash, dirty-worktree status or patch identifier when relevant, dependency versions, environment/specification version, and machine characteristics.
- Vehicle geometry, masses/inertia when relevant, hitch specifications, actuator limits, controller time step, simulation step, sensor assumptions.
- Yard layout, parking-space dimensions, start/goal poses, obstacles, scenario generation parameters.
- Algorithm and baseline versions, hyperparameters, optimization settings, reward definition, constraints, training budget, and random seeds.
- Evaluation dataset/scenario IDs, number of trials, success/failure definitions, metrics, uncertainty estimates, and output paths.
- Unexpected outcomes, failures, invalid runs, and interpretation caveats.

Prefer machine-readable experiment configuration files and structured logs/results. Every manuscript table, number, and generated figure must link back to one or more experiment IDs; do not rely on remembered console output.

Never quietly overwrite experiment outputs or cherry-pick successful trials. Preserve meaningful negative and failed results; distinguish invalid implementation runs from valid poor controller performance.

## 7. Scientific rigor and fairness in comparisons

The starting empirical question is how trajectory-aware PPO and NMPC behave on shared reverse-docking and reverse trajectory-tracking tasks, **not** which one must win.

- Keep environment, task definitions, geometry, action limits, observation availability, sampling rates, initial-state distributions, and evaluation metrics aligned as far as methodologically reasonable.
- Explicitly document information asymmetries: NMPC uses a prediction model; PPO uses trained experience and its chosen observation representation.
- For the initial primary comparison, preserve the agreed fixed reverse-speed and bounded steering-rate plant interface. If NMPC is later given velocity authority or a different command space, treat that as a separately named experiment and document why the comparison remains informative.
- Distinguish online control/runtime cost from offline training, model identification, and tuning costs.
- Use multiple scenarios and random seeds where stochasticity makes them necessary; report spread/uncertainty rather than only favorable single runs.
- Define success, collisions, trailer-pose tolerance, jackknife thresholds, timeouts, and numerical-failure handling before comparing results.
- Avoid test-set leakage during reward shaping, hyperparameter tuning, scenario selection, or controller adaptation.
- Validate the vehicle model and simulation behavior before interpreting controller results.
- When moving from 2D to 3D, document what changes in dynamics, contact, controls, sensing, and fidelity; do not call unlike scenarios a controlled comparison.
- Geometry changes and parking-space variations may be **sensitivity, robustness, or generalization** studies. Reserve **ablation** for systematically removing/modifying a component to isolate its effect.

Potential experiments to consider, **not mandatory commitments**:

- Straight reverse docking, angled docking, and constrained parking bays.
- Variations in trailer hitch-to-axle distance, tractor geometry, hitch placement, parking clearance, start pose, or articulation angle.
- PPO versus NMPC success rate, final trailer position/orientation error, collisions, maximum hitch angle, control smoothness, maneuver time, and compute cost.
- Held-out geometric configurations, simulation fidelity gaps, and 2D-to-3D adaptation or transfer.
- Reward-component ablations, NMPC constraint/cost variations, and controller sensitivity to mismatched dynamics.

Introduce these only if justified by scientific value, available time, and functioning infrastructure. Do not promise all of them in the manuscript in advance.

## 8. Developing the environments and controllers

Work incrementally and favor independently verifiable milestones. The shared
vehicle core and version-1 vehicle/dock specification are implemented. The
MuJoCo vehicle, dock, adapter, and local viewers are also implemented and
validated against the specification's kinematic equations. The canonical 2D
kinematic simulator is implemented with RK4 integration, rigid state
reconstruction, shared command supervision, and direct low-speed trajectory
parity tests against MuJoCo. The Gymnasium task and controllers remain
unimplemented; collision, safety-event, and goal-event parity remain pending.

1. Implement typed state, command, geometry, limits, and configuration objects from the frozen version-1 specification.
2. Maintain the implemented and quantitatively validated 2D kinematic model: straight reverse, constant steering, articulation behavior, angle wrapping, rigid geometry, determinism, and timestep convergence.
3. Implement reference paths, path-relative errors/previews, vehicle footprints, dock geometry, collision checks, jackknife checks, and goal evaluation.
4. Compose the verified modules into a Gymnasium environment with deterministic reset/step tests, explicit reward decomposition, and diagnostic `info` fields.
5. Implement a simple deterministic tracker to validate the closed loop, then implement NMPC and PPO with instrumentation and saved configurations.
6. Maintain the implemented version-1 MuJoCo vehicle and dock; preserve its
   verified semantic sites, hitch joint, Ackermann mapping, wheels,
   drive/braking, collision/contact behavior, timing, and neutral-pose
   fixtures while extending parity from plant trajectories to shared task events.
7. Establish the shared controller/simulator adapter and run clearly identified 2D-to-3D transfer tests without assuming zero-shot equivalence.
8. Run systematic held-out evaluations and refine experimental questions based on evidence.

These are planning guidelines, not a mandate to redesign an existing repository or execute steps out of the user's chosen order. Reuse code and project patterns already present. Do not claim physical accuracy solely because the 3D model looks realistic. Validate kinematics, geometric constraints, contact, and representative trajectories with quantitative checks when possible.

Keep 2D and 3D implementations traceable to shared vehicle/scenario specifications where practical, but do not force identical internal physical models. Avoid introducing new algorithmic complexity before the baseline comparison works.

## 9. Discovering contributions without manufacturing them

Actively look for scientifically interesting, repeatable observations while developing:

- Controllers failing systematically for particular trailer geometries or tight spaces.
- Unexpected sensitivity to model assumptions, contact dynamics, steering limits, or observation choices.
- Clear tradeoffs in sample efficiency, robustness, computation, or transfer from 2D to 3D.
- Useful and reusable simulation methodology or benchmark assets.

Record such findings as **candidate contributions** in `research/open_questions.md` or the existing equivalent. Investigate and verify them, check related literature, and only then propose moving them into the manuscript as contributions. Never declare a result novel based solely on its being new to this repository.

Do not lock the project into new algorithm development, geometry-conditioned PPO, sim-to-sim transfer, or any particular benchmark without explicit evidence and user agreement.

## 10. Citations, authorship, and research integrity

- Never invent papers, DOI identifiers, authors, quotations, experimental numbers, plots, baselines, or statistical significance.
- If external literature is inaccessible, clearly label a reference as unverified and do not turn it into a definitive claim.
- Maintain valid BibTeX records with traceable bibliographic details and cite sources where claims need support.
- Treat AI-assisted writing and coding as requiring human review and appropriate disclosure under the eventual course/institution/publisher rules.
- Do not submit, upload, publish, or distribute the paper, code, or data externally without user direction.
- Be explicit about the limits of simulation-only evaluation and avoid unsupported real-world safety or deployment claims.
- Local publisher PDFs are study copies. Before making the repository public or sharing it outside the team, audit `research/papers/`, restore appropriate ignore rules, and remove any PDF that cannot be redistributed. A citation link or BibTeX record does not grant redistribution rights.

## 11. End-of-task research integration checklist

Before finishing a substantive coding/research task:

- [ ] Did I inspect existing code and documents instead of assuming a clean repository?
- [ ] Is the changed behavior implemented and tested as far as possible?
- [ ] Do the paper's model/method descriptions still match the code?
- [ ] Are any observed results traceable to saved experiments rather than guesses?
- [ ] For a significant run, is there an immutable experiment ID with configuration, provenance, status, metrics, and caveats?
- [ ] Have I recorded relevant assumptions, failures, limitations, and new research questions?
- [ ] If I changed LaTeX, have I checked for compilation or clearly disclosed that I could not?
- [ ] Did I avoid presenting plans, speculative insights, or unverified claims as established findings?
- [ ] Have I told the user what changed, what evidence exists, and what comes next?

**Default behavior:** Build and verify first; document meaningful changes concurrently; let evidence shape the eventual paper.
