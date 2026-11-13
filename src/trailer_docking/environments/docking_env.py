"""Gymnasium environment composed from the verified planar plant and task."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from trailer_docking.core.config import (
    DockConfiguration,
    DockingTaskConfiguration,
    VehicleConfiguration,
    load_dock_configuration,
    load_docking_task_configuration,
    load_vehicle_configuration,
)
from trailer_docking.core.geometry import Bounds1D
from trailer_docking.core.state import Point2D, Pose2D, VehicleCommand, VehicleState
from trailer_docking.environments.rendering import DockingRenderer
from trailer_docking.environments.task import (
    TaskStatus,
    default_start_pose,
    evaluate_task,
    workspace_bounds,
)
from trailer_docking.observations import FloatArray, ObservationBuilder
from trailer_docking.paths import ReferencePath, manual_reference, tracking_state
from trailer_docking.paths.tracking import TrackingState
from trailer_docking.simulators.kinematic import KinematicSimulator

ActionArray = NDArray[np.float32]
ActionMode = Literal["fixed_reverse", "teleop"]

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VEHICLE_PATH = (
    REPOSITORY_ROOT / "configs" / "vehicles" / "compact_semitrailer_v1.yaml"
)
DEFAULT_DOCK_PATH = (
    REPOSITORY_ROOT / "configs" / "environments" / "nominal_dock_v1.yaml"
)
DEFAULT_TASK_PATH = (
    REPOSITORY_ROOT / "configs" / "environments" / "gymnasium_docking_v1.yaml"
)


class TrailerDockingEnv(gym.Env[FloatArray, ActionArray]):
    """Low-speed reverse-docking task with auditable event and reward output.

    ``fixed_reverse`` is the research contract: one normalized steering-rate
    action and the configured fixed reverse speed. ``teleop`` adds normalized
    speed as a second action for manual debugging and demonstrations.
    """

    metadata: dict[str, Any] = {  # noqa: RUF012
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(
        self,
        *,
        vehicle: VehicleConfiguration | None = None,
        dock: DockConfiguration | None = None,
        vehicle_path: str | Path = DEFAULT_VEHICLE_PATH,
        dock_path: str | Path = DEFAULT_DOCK_PATH,
        task_path: str | Path = DEFAULT_TASK_PATH,
        trailer_variant: str | None = None,
        action_mode: ActionMode = "fixed_reverse",
        render_mode: Literal["human", "rgb_array"] | None = None,
        task_settings: DockingTaskConfiguration | None = None,
    ) -> None:
        super().__init__()
        if action_mode not in ("fixed_reverse", "teleop"):
            raise ValueError("action_mode must be 'fixed_reverse' or 'teleop'")
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError(f"unsupported render_mode={render_mode!r}")
        self.vehicle = vehicle or load_vehicle_configuration(vehicle_path)
        if trailer_variant is not None:
            self.vehicle = self.vehicle.with_trailer_variant(trailer_variant)
        self.dock = dock or load_dock_configuration(dock_path)
        self.action_mode = action_mode
        self.render_mode = render_mode
        self.settings = task_settings or load_docking_task_configuration(task_path)
        action_size = 1 if action_mode == "fixed_reverse" else 2
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_size,), dtype=np.float32
        )
        observation_size = 15 + 2 * len(self.settings.preview_distances)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(observation_size,),
            dtype=np.float32,
        )
        self.simulator = KinematicSimulator(self.vehicle)
        self.path: ReferencePath
        self.observation_builder: ObservationBuilder
        self.tracking: TrackingState
        self.status: TaskStatus
        self._x_bounds: Bounds1D
        self._y_bounds: Bounds1D
        self._previous_distance = 0.0
        self._steps = 0
        self._episode_ended = False
        self._last_reward = 0.0
        self._reward_terms: dict[str, float] = {}
        self._renderer: DockingRenderer | None = None
        self.reset()

    @property
    def state(self) -> VehicleState:
        return self.simulator.state

    @property
    def episode_steps(self) -> int:
        return self._steps

    def _start_from_options(
        self, options: Mapping[str, Any]
    ) -> tuple[Pose2D, float, float]:
        pose = options.get("tractor_pose")
        trailer_yaw = options.get("trailer_yaw")
        steering_angle = options.get("steering_angle", 0.0)
        if pose is not None:
            if not isinstance(pose, Pose2D):
                raise ValueError("options['tractor_pose'] must be Pose2D")
            yaw = pose.yaw if trailer_yaw is None else trailer_yaw
            return pose, float(yaw), float(steering_angle)

        pose = default_start_pose(self.vehicle, self.dock.geometry, self.settings)
        yaw = pose.yaw
        if bool(options.get("randomize", False)):
            lateral = float(self.np_random.uniform(-1.0, 1.0))
            heading = float(self.np_random.uniform(-0.12, 0.12))
            articulation = float(self.np_random.uniform(-0.08, 0.08))
            pose = Pose2D(pose.x, lateral, heading)
            yaw = heading - articulation
        return pose, float(yaw), float(steering_angle)

    def _build_path(self, state: VehicleState) -> ReferencePath:
        goal = self.dock.geometry.target_position
        current = self.vehicle.trailer.rear_face(state.trailer_pose)
        entry = Point2D(self.dock.geometry.interior_depth + 0.75, 0.0)
        if abs(current.x - goal.x) < 1e-12 and abs(current.y - goal.y) < 1e-12:
            return manual_reference((goal, entry))
        if abs(current.y) < 1e-12:
            return manual_reference((goal, current))
        return manual_reference((goal, entry, current))

    def _tracking(self, state: VehicleState) -> TrackingState:
        docking = self.vehicle.trailer.rear_face(state.trailer_pose)
        return tracking_state(
            state, self.path, docking, self.settings.preview_distances
        )

    def _observation(
        self, state: VehicleState, tracking: TrackingState, status: TaskStatus
    ) -> FloatArray:
        errors = status.errors
        return self.observation_builder.build(
            state,
            tracking,
            errors.longitudinal,
            errors.lateral,
            errors.trailer_yaw,
        )

    def _info(self) -> dict[str, Any]:
        return {
            "vehicle_specification": self.vehicle.specification_id,
            "vehicle_fingerprint": self.vehicle.effective_fingerprint,
            "dock_specification": self.dock.geometry.specification_id,
            "dock_fingerprint": self.dock.source_fingerprint,
            "task_specification": self.settings.specification_id,
            "task_fingerprint": self.settings.source_fingerprint,
            "is_success": self.status.success,
            "termination_reason": self.status.reason,
            "collision": self.status.collision.collided,
            "collision_contacts": self.status.collision.contacts,
            "jackknifed": self.status.articulation.jackknifed,
            "soft_articulation_limit": self.status.articulation.soft_limit_reached,
            "out_of_bounds": self.status.out_of_bounds,
            "trailer_completely_inside": self.status.trailer_completely_inside,
            "docking_error_longitudinal": self.status.errors.longitudinal,
            "docking_error_lateral": self.status.errors.lateral,
            "docking_error_yaw": self.status.errors.trailer_yaw,
            "path_progress": self.tracking.progress,
            "distance_to_goal": self.tracking.distance_to_goal,
            "reward_terms": dict(self._reward_terms),
            "applied_command": self.simulator.applied_command.to_dict(),
            "episode_steps": self._steps,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[FloatArray, dict[str, Any]]:
        """Reset deterministically; optional randomized starts use Gym's RNG."""

        super().reset(seed=seed)
        pose, trailer_yaw, steering = self._start_from_options(options or {})
        state = self.simulator.reset(pose, trailer_yaw, steering_angle=steering)
        self.path = self._build_path(state)
        self.observation_builder = ObservationBuilder(
            self.vehicle,
            self.dock.geometry,
            self.path,
            self.settings.preview_distances,
        )
        self._x_bounds, self._y_bounds = workspace_bounds(
            pose, self.dock.geometry, self.settings
        )
        self.tracking = self._tracking(state)
        self.status = evaluate_task(
            state,
            self.vehicle,
            self.dock.geometry,
            self._x_bounds,
            self._y_bounds,
        )
        self._previous_distance = self.tracking.distance_to_goal
        self._steps = 0
        self._episode_ended = False
        self._last_reward = 0.0
        self._reward_terms = {}
        if self._renderer is not None:
            self._renderer.reset_path(self.path)
        observation = self._observation(state, self.tracking, self.status)
        if self.render_mode == "human":
            self.render()
        return observation, self._info()

    def _command(self, action: ActionArray) -> VehicleCommand:
        rate = float(action[0]) * self.vehicle.limits.steering_rate_abs_max
        if self.action_mode == "fixed_reverse":
            speed = self.vehicle.command.nominal_target_speed
        else:
            normalized_speed = float(action[1])
            speed_limit = (
                self.vehicle.limits.forward_speed_max
                if normalized_speed >= 0.0
                else self.vehicle.limits.reverse_speed_abs_max
            )
            speed = normalized_speed * speed_limit
        return VehicleCommand(rate, speed)

    def _reward(self, state: VehicleState) -> float:
        progress = self._previous_distance - self.tracking.distance_to_goal
        lateral_scale = max(self.dock.geometry.interior_width, 1.0)
        heading = self.tracking.trailer_heading_error
        articulation = (
            state.articulation_angle / self.vehicle.limits.articulation_hard_abs_max
        )
        steering_rate = (
            state.steering_rate / self.vehicle.limits.steering_rate_abs_max
        )
        terms = {
            "progress": self.settings.reward_progress * progress,
            "time": -self.settings.reward_time,
            "lateral": -self.settings.reward_lateral_error
            * (self.tracking.lateral_error / lateral_scale) ** 2,
            "heading": -self.settings.reward_heading_error * heading**2,
            "articulation": -self.settings.reward_articulation * articulation**2,
            "steering_rate": -self.settings.reward_steering_rate * steering_rate**2,
            "success": self.settings.reward_success if self.status.success else 0.0,
            "collision": -self.settings.reward_collision
            if self.status.collision.collided
            else 0.0,
            "jackknife": -self.settings.reward_jackknife
            if self.status.articulation.jackknifed
            else 0.0,
            "out_of_bounds": -self.settings.reward_out_of_bounds
            if self.status.out_of_bounds
            else 0.0,
        }
        self._reward_terms = terms
        return float(sum(terms.values()))

    def step(
        self, action: ActionArray
    ) -> tuple[FloatArray, float, bool, bool, dict[str, Any]]:
        """Advance one controller period and evaluate all task conditions."""

        if self._episode_ended:
            raise RuntimeError("step() called after episode end; call reset()")
        candidate = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(candidate):
            raise ValueError(
                f"action must be finite, shape {self.action_space.shape}, in [-1, 1]"
            )
        state = self.simulator.step(self._command(candidate))
        self._steps += 1
        self.tracking = self._tracking(state)
        self.status = evaluate_task(
            state,
            self.vehicle,
            self.dock.geometry,
            self._x_bounds,
            self._y_bounds,
        )
        reward = self._reward(state)
        self._last_reward = reward
        self._previous_distance = self.tracking.distance_to_goal
        terminated = self.status.terminated
        truncated = not terminated and self._steps >= self.settings.max_episode_steps
        self._episode_ended = terminated or truncated
        observation = self._observation(state, self.tracking, self.status)
        info = self._info()
        if truncated:
            info["termination_reason"] = "time_limit"
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info

    def render(self) -> NDArray[np.uint8] | None:
        """Render through the backend-independent Matplotlib top-down view."""

        if self.render_mode is None:
            return None
        if self._renderer is None:
            self._renderer = DockingRenderer(
                self.vehicle,
                self.dock,
                self.path,
                cast(Literal["human", "rgb_array"], self.render_mode),
                self._x_bounds,
                self._y_bounds,
            )
        return self._renderer.render(
            self.state,
            self.tracking,
            self.status,
            self._last_reward,
        )

    @property
    def figure(self) -> Any:
        """Matplotlib figure used by the interactive teleoperation entry point."""

        if self._renderer is None:
            self.render()
        return None if self._renderer is None else self._renderer.view.figure

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
