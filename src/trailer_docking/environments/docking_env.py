"""Gymnasium reverse-docking environment over any shared-contract simulator.

The environment implements the proposal's pipeline::

    start state + dock -> path planner -> path-relative errors + preview points
        -> policy -> bounded steering-rate command -> simulator

On every reset a planner produces a reference path for the trailer axle inside
the generated scenario; the policy observes only path-relative quantities and
is rewarded for following the path to the dock.  The path is available as
:attr:`DockingEnv.reference_path` so viewers can draw it.

This is a *skeleton*.  The interfaces are intended to be stable; the reward
weights, preview spacing, and normalization are placeholders that the source
material leaves open, and must be settled and recorded before any result is
interpreted.

The environment owns no vehicle model.  It drives a
:class:`~trailer_docking.simulators.base.Simulator` and scores it with the
shared docking-outcome functions, so the same policy runs unchanged against
the 2D backend or MuJoCo.

Action (``Box(-1, 1, (1,))``): normalized steering rate.  Target speed is fixed
at the configured nominal reverse speed (decision D-002).

Observation: see :mod:`trailer_docking.observations.builder`.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

import gymnasium
import numpy as np

from trailer_docking.core.config import DockConfiguration, VehicleConfiguration
from trailer_docking.core.state import VehicleCommand, VehicleState
from trailer_docking.evaluation.metrics import DockingOutcome, evaluate_docking
from trailer_docking.evaluation.scenarios import Scenario, sample_start
from trailer_docking.observations.builder import ObservationLayout, build_observation
from trailer_docking.paths.generators import DEFAULT_PLANNER, Planner
from trailer_docking.paths.reference import ReferencePath
from trailer_docking.paths.tracking import TrackingErrors, tracking_errors
from trailer_docking.simulators.base import Simulator

SimulatorFactory = Callable[[VehicleConfiguration, DockConfiguration], Simulator]

PREVIEW_DISTANCES = (1.0, 2.0, 4.0)
OBSERVATION_BOUND = 10.0
# Leaving the operating area: too far from the reference path.
OFF_PATH_DISTANCE = 3.0

# Placeholder reward weights; see the module docstring.
PROGRESS_WEIGHT = 1.0
LATERAL_WEIGHT = 0.05
HEADING_WEIGHT = 0.05
ARTICULATION_WEIGHT = 0.05
STEERING_RATE_WEIGHT = 0.005
STEP_PENALTY = 0.005
SUCCESS_REWARD = 10.0
FAILURE_PENALTY = 10.0

OUTCOME_DOCKED = "docked"
OUTCOME_JACKKNIFE = "jackknife"
OUTCOME_COLLISION = "collision"
OUTCOME_OFF_PATH = "off_path"
OUTCOME_TIMEOUT = "timeout"


def mujoco_simulator_factory(
    vehicle: VehicleConfiguration, dock: DockConfiguration
) -> Simulator:
    """Default backend until the 2D kinematic simulator exists."""

    from trailer_docking.simulators.mujoco.adapter import MujocoSimulator

    return MujocoSimulator(vehicle, dock)


class DockingEnv(gymnasium.Env[np.ndarray, np.ndarray]):
    """Follow a planned reference path to reverse a semitrailer into a bay."""

    metadata = {"render_modes": []}  # noqa: RUF012 - Gymnasium API

    def __init__(
        self,
        scenario: Scenario | None = None,
        *,
        simulator_factory: SimulatorFactory | None = None,
        planner: Planner | None = None,
        max_episode_steps: int = 900,
        on_state: Callable[[VehicleState], None] | None = None,
    ) -> None:
        if max_episode_steps < 1:
            raise ValueError("max_episode_steps must be at least 1")
        self.scenario = scenario or Scenario()
        self.built = self.scenario.build()
        self.vehicle = self.built.vehicle
        self.dock = self.built.dock
        factory = simulator_factory or mujoco_simulator_factory
        self.simulator = factory(self.vehicle, self.dock)
        self.planner: Planner = planner or DEFAULT_PLANNER
        self.max_episode_steps = max_episode_steps
        # Called with every new state so a UI can watch episodes live.
        self.on_state = on_state
        self.layout = ObservationLayout(preview_count=len(PREVIEW_DISTANCES))
        self.action_space = gymnasium.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        self.observation_space = gymnasium.spaces.Box(
            -OBSERVATION_BOUND, OBSERVATION_BOUND, (self.layout.size,), dtype=np.float32
        )
        self.reference_path: ReferencePath | None = None
        self._start_rng = random.Random(0)
        self._steps = 0
        self._previous_arc_length = 0.0

    # ------------------------------------------------------------- helpers

    def _evaluate(self, state: VehicleState) -> DockingOutcome:
        return evaluate_docking(
            state, self.vehicle.geometry, self.dock.geometry, self.vehicle.limits
        )

    def _track(self, state: VehicleState) -> TrackingErrors:
        if self.reference_path is None:
            raise RuntimeError("reset() must be called before step()")
        return tracking_errors(self.reference_path, state, PREVIEW_DISTANCES)

    def _observation(self, state: VehicleState, errors: TrackingErrors) -> np.ndarray:
        values = np.array(
            build_observation(state, errors, self.vehicle.limits), dtype=np.float32
        )
        return np.clip(values, -OBSERVATION_BOUND, OBSERVATION_BOUND)

    def _info(
        self, state: VehicleState, outcome: DockingOutcome, errors: TrackingErrors
    ) -> dict[str, Any]:
        return {
            "longitudinal_error": outcome.error.longitudinal,
            "lateral_error": outcome.error.lateral,
            "yaw_error": outcome.error.yaw,
            "articulation_angle": state.articulation_angle,
            "path_lateral_error": errors.trailer_lateral_error,
            "path_heading_error": errors.trailer_heading_error,
            "path_progress": errors.arc_length,
            "path_remaining": errors.remaining_distance,
            "collisions": outcome.collisions,
            "is_success": outcome.success,
            "simulation_time": state.simulation_time,
        }

    # ------------------------------------------------------- Gymnasium API

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        # Only an explicit seed restarts the start-pose sequence; unseeded
        # resets continue it, so a training run is reproducible from the seed
        # of its first reset.
        if seed is not None:
            self._start_rng = random.Random(seed)
        tractor_pose, trailer_yaw = sample_start(
            self.scenario, self.built, self._start_rng
        )
        state = self.simulator.reset(tractor_pose, trailer_yaw)
        self.reference_path = self.planner.plan(
            self.vehicle.geometry, self.dock.geometry, state
        )
        self._steps = 0
        errors = self._track(state)
        self._previous_arc_length = errors.arc_length
        outcome = self._evaluate(state)
        if self.on_state is not None:
            self.on_state(state)
        return self._observation(state, errors), self._info(state, outcome, errors)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        limits = self.vehicle.limits
        normalized = float(
            np.clip(np.asarray(action, dtype=np.float64).ravel()[0], -1.0, 1.0)
        )
        command = VehicleCommand(
            steering_rate=normalized * limits.steering_rate_abs_max,
            target_speed=self.vehicle.command.nominal_target_speed,
        )
        state = self.simulator.step(command)
        self._steps += 1
        errors = self._track(state)
        outcome = self._evaluate(state)

        # Progress along the path, tracking and articulation penalties, and a
        # small steering-rate penalty, as listed in the proposal.
        progress = errors.arc_length - self._previous_arc_length
        self._previous_arc_length = errors.arc_length
        articulation = state.articulation_angle / limits.articulation_hard_abs_max
        reward = (
            PROGRESS_WEIGHT * progress
            - LATERAL_WEIGHT * abs(errors.trailer_lateral_error)
            - HEADING_WEIGHT * abs(errors.trailer_heading_error)
            - ARTICULATION_WEIGHT * articulation**2
            - STEERING_RATE_WEIGHT * normalized**2
            - STEP_PENALTY
        )

        terminated = True
        truncated = False
        if outcome.success:
            reason = OUTCOME_DOCKED
            reward += SUCCESS_REWARD
        elif outcome.jackknifed:
            reason = OUTCOME_JACKKNIFE
            reward -= FAILURE_PENALTY
        elif outcome.collisions:
            reason = OUTCOME_COLLISION
            reward -= FAILURE_PENALTY
        elif abs(errors.trailer_lateral_error) > OFF_PATH_DISTANCE:
            reason = OUTCOME_OFF_PATH
            reward -= FAILURE_PENALTY
        else:
            terminated = False
            reason = ""
            if self._steps >= self.max_episode_steps:
                truncated = True
                reason = OUTCOME_TIMEOUT

        info = self._info(state, outcome, errors)
        if reason:
            info["outcome"] = reason
        if self.on_state is not None:
            self.on_state(state)
        return (
            self._observation(state, errors),
            float(reward),
            terminated,
            truncated,
            info,
        )
