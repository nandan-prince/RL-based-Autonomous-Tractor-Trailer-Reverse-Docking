"""Scenario-driven environment over the verified Gymnasium docking task.

Training and docking tests describe a problem as a
:class:`~trailer_docking.evaluation.scenarios.Scenario`.  This environment
builds the project's :class:`~trailer_docking.environments.TrailerDockingEnv`
from that scenario, draws each episode's start pose from the scenario's start
distribution, and applies the scenario's optional domain randomization
(trailer length per episode, observation noise).  It adds no dynamics, reward,
or event logic: every step is the task environment's step.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import replace
from typing import Any, cast

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from trailer_docking.core.state import VehicleState
from trailer_docking.environments.docking_env import (
    Backend,
    TrailerDockingEnv,
)
from trailer_docking.evaluation.scenarios import (
    BuiltScenario,
    Scenario,
    sample_start,
)
from trailer_docking.paths.reference import ReferencePath
from trailer_docking.training.config import validate_backend

OUTCOME_SUCCESS = "success"
OUTCOME_TIME_LIMIT = "time_limit"
UNKNOWN_OUTCOME = "unknown"
FRAME_INTERVAL = 3
"""Control steps between 3D frames: 10 Hz control gives about 3 frames/s of
simulated time, which keeps rendering a small share of a training step."""

StateCallback = Callable[[VehicleState], None]
RouteCallback = Callable[[ReferencePath], None]
FrameCallback = Callable[[NDArray[np.uint8]], None]
Observation = NDArray[np.float32]


def episode_outcome(info: dict[str, Any]) -> str:
    """Name how an episode ended, using the task's own termination reasons."""

    return str(info.get("termination_reason") or UNKNOWN_OUTCOME)


class ScenarioDockingEnv(gym.Env[Observation, Observation]):
    """``TrailerDockingEnv`` in ``fixed_reverse`` mode driven by a scenario.

    ``max_episode_steps`` replaces the task configuration's step limit when
    given; callers that override it must record the value with their results,
    because the task fingerprint describes the unmodified configuration.
    ``backend`` is ``"2d"`` (kinematic) or ``"3d"`` (MuJoCo).
    ``on_state`` receives every new vehicle state and ``on_route`` each
    episode's reference path, so a UI can watch episodes live.  With the 3D
    backend, ``on_frame`` receives a rendered RGB image of the MuJoCo scene
    every few steps; it is ignored in 2D, which has no 3D scene.

    With trailer-length randomization the vehicle, and with it the simulator,
    is rebuilt at every reset; :attr:`built` and :attr:`task_env` then describe
    the episode in progress.
    """

    metadata: dict[str, Any] = {"render_modes": []}  # noqa: RUF012

    def __init__(
        self,
        scenario: Scenario | None = None,
        *,
        max_episode_steps: int | None = None,
        backend: str = "2d",
        on_state: StateCallback | None = None,
        on_route: RouteCallback | None = None,
        on_frame: FrameCallback | None = None,
    ) -> None:
        super().__init__()
        if max_episode_steps is not None and max_episode_steps < 1:
            raise ValueError("max_episode_steps must be at least 1")
        self.scenario = scenario if scenario is not None else Scenario()
        self.backend = cast(Backend, validate_backend(backend))
        self.max_episode_steps = max_episode_steps
        self.on_state = on_state
        self.on_route = on_route
        self.on_frame = on_frame if self.backend == "3d" else None
        self._start_rng = random.Random(0)
        self._noise_rng = np.random.default_rng(0)
        self._frame_renderer: Any = None
        self._steps = 0
        self.built: BuiltScenario = self.scenario.build()
        # The task's own fingerprints describe the base specifications only;
        # this one also covers every override, start range, and randomization.
        self.scenario_fingerprint = self.scenario.fingerprint
        self.task_env = self._make_task_env(self.built)
        self.action_space = self.task_env.action_space
        self.observation_space = self.task_env.observation_space

    def _make_task_env(self, built: BuiltScenario) -> TrailerDockingEnv:
        settings = built.task
        if self.max_episode_steps is not None:
            settings = replace(settings, max_episode_steps=self.max_episode_steps)
        return TrailerDockingEnv(
            vehicle=built.vehicle,
            dock=built.dock,
            task_settings=settings,
            action_mode="fixed_reverse",
            backend=self.backend,
            planner=self.scenario.planner,
            obstacles=built.obstacles,
        )

    @property
    def reference_path(self) -> ReferencePath:
        """Reference path of the episode in progress."""

        return self.task_env.path

    @property
    def state(self) -> VehicleState:
        return self.task_env.state

    def _close_renderer(self) -> None:
        if self._frame_renderer is not None:
            self._frame_renderer.close()
            self._frame_renderer = None

    def _randomize_vehicle(self) -> None:
        """Draw this episode's trailer length and rebuild the task for it."""

        bounds = self.scenario.randomization.trailer_hitch_to_axle
        if bounds is None:
            return
        length = self._start_rng.uniform(*bounds)
        # This episode's scenario has the drawn length fixed, not a range.
        episode = replace(
            self.scenario,
            overrides=replace(self.scenario.overrides, trailer_hitch_to_axle=length),
            randomization=replace(
                self.scenario.randomization, trailer_hitch_to_axle=None
            ),
        )
        self._close_renderer()
        self.task_env.close()
        self.built = episode.build()
        self.task_env = self._make_task_env(self.built)

    def _annotate(self, info: dict[str, Any]) -> None:
        info["scenario_fingerprint"] = self.scenario_fingerprint
        info["trailer_hitch_to_axle"] = self.built.vehicle.trailer.hitch_to_axle

    def _noisy(self, observation: Observation) -> Observation:
        deviation = self.scenario.randomization.observation_noise_std
        if deviation == 0.0:
            return observation
        noise = self._noise_rng.normal(0.0, deviation, size=observation.shape)
        return np.clip(observation + noise, -1.0, 1.0).astype(np.float32)

    def _publish(self, *, force_frame: bool = False) -> None:
        """Send the new state, and a 3D frame when one is due, to the watchers."""

        if self.on_state is not None:
            self.on_state(self.task_env.state)
        if self.on_frame is None:
            return
        if not force_frame and self._steps % FRAME_INTERVAL != 0:
            return
        if self._frame_renderer is None:
            from trailer_docking.visualization.render3d import (
                MujocoFrameRenderer,
                scene_length,
            )

            self._frame_renderer = MujocoFrameRenderer(
                cast(Any, self.task_env.simulator).model,
                scene_length=scene_length(
                    self.built.vehicle,
                    self.built.dock,
                    max(self.scenario.start.standoff),
                ),
            )
        from trailer_docking.visualization.render3d import display

        simulator = cast(Any, self.task_env.simulator)
        route = self.task_env.path.points if display.show_route else None
        self.on_frame(self._frame_renderer.render(simulator.data, route))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)
        # Only an explicit seed restarts the random sequences; unseeded resets
        # continue them, so a run is reproducible from its first seed.
        if seed is not None:
            self._start_rng = random.Random(seed)
            self._noise_rng = np.random.default_rng(seed)
        self._randomize_vehicle()
        chosen = dict(options or {})
        if "tractor_pose" not in chosen:
            tractor_pose, trailer_yaw = sample_start(
                self.scenario, self.built, self._start_rng
            )
            chosen["tractor_pose"] = tractor_pose
            chosen["trailer_yaw"] = trailer_yaw
        observation, info = self.task_env.reset(seed=seed, options=chosen)
        self._annotate(info)
        self._steps = 0
        if self.on_route is not None:
            self.on_route(self.task_env.path)
        self._publish(force_frame=True)
        return self._noisy(observation), info

    def step(
        self, action: Observation
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.task_env.step(action)
        self._annotate(info)
        self._steps += 1
        self._publish(force_frame=terminated or truncated)
        return self._noisy(observation), float(reward), terminated, truncated, info

    def close(self) -> None:
        self._close_renderer()
        self.task_env.close()


EnvironmentFactory = Callable[..., ScenarioDockingEnv]
"""Called as ``factory(scenario, max_episode_steps=..., backend=...,
on_state=..., on_route=..., on_frame=...)``; the default is
:class:`ScenarioDockingEnv`."""
