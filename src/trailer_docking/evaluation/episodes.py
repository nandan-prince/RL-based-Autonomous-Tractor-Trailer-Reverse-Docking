"""Resolved, replayable episode definitions for paired experiments.

A :class:`Scenario` describes distributions.  A :class:`ResolvedEpisode`
captures one draw from those distributions, including the vehicle geometry,
initial articulated pose, and planned reference path.  The same object can
therefore be replayed by every controller and simulator backend without
silently resampling any part of the task.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Any

from trailer_docking.core.state import Point2D, Pose2D
from trailer_docking.evaluation.scenarios import (
    Scenario,
    sample_reference_planner,
    sample_start,
)
from trailer_docking.paths.planners import plan_reference
from trailer_docking.paths.reference import ReferencePath
from trailer_docking.simulators.kinematic import KinematicSimulator


@dataclass(frozen=True, slots=True)
class ResolvedEpisode:
    """One immutable task draw shared by all evaluated control methods."""

    seed: int
    scenario_fingerprint: str
    tractor_pose: Pose2D
    trailer_yaw: float
    steering_angle: float
    trailer_hitch_to_axle: float
    reference_path: ReferencePath
    episode_fingerprint: str
    reference_planner: str = "explicit"

    def reset_options(self) -> dict[str, Any]:
        """Return exact options accepted by the underlying task environment."""

        return {
            "tractor_pose": self.tractor_pose,
            "trailer_yaw": self.trailer_yaw,
            "steering_angle": self.steering_angle,
            "reference_path": self.reference_path,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-compatible, provenance-preserving mapping."""

        return {
            "seed": self.seed,
            "scenario_fingerprint": self.scenario_fingerprint,
            "episode_fingerprint": self.episode_fingerprint,
            "tractor_pose": self.tractor_pose.to_dict(),
            "trailer_yaw": self.trailer_yaw,
            "steering_angle": self.steering_angle,
            "trailer_hitch_to_axle": self.trailer_hitch_to_axle,
            "reference_path": [point.to_dict() for point in self.reference_path.points],
            "reference_planner": self.reference_planner,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ResolvedEpisode:
        """Restore a saved episode and verify its content fingerprint."""

        expected = {
            "seed",
            "scenario_fingerprint",
            "episode_fingerprint",
            "tractor_pose",
            "trailer_yaw",
            "steering_angle",
            "trailer_hitch_to_axle",
            "reference_path",
            "reference_planner",
        }
        legacy = expected - {"reference_planner"}
        keys = set(data)
        if keys not in (expected, legacy):
            raise ValueError("resolved episode fields do not match the schema")
        try:
            fingerprint_fields = keys - {"episode_fingerprint"}
            content = {key: data[key] for key in fingerprint_fields}
            actual = _fingerprint(content)
            recorded = str(data["episode_fingerprint"])
            if actual != recorded:
                raise ValueError("resolved episode fingerprint does not match content")
            return cls(
                seed=int(data["seed"]),
                scenario_fingerprint=str(data["scenario_fingerprint"]),
                tractor_pose=Pose2D.from_dict(data["tractor_pose"]),
                trailer_yaw=float(data["trailer_yaw"]),
                steering_angle=float(data["steering_angle"]),
                trailer_hitch_to_axle=float(data["trailer_hitch_to_axle"]),
                reference_path=ReferencePath(
                    tuple(Point2D.from_dict(item) for item in data["reference_path"])
                ),
                episode_fingerprint=recorded,
                reference_planner=str(data.get("reference_planner", "legacy-explicit")),
            )
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, ValueError) and str(error).startswith(
                "resolved episode"
            ):
                raise
            raise ValueError(f"invalid resolved episode: {error}") from error


def _fingerprint(content: dict[str, Any]) -> str:
    serialized = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return sha256(serialized.encode("utf-8")).hexdigest()


def resolve_episode(scenario: Scenario, seed: int) -> ResolvedEpisode:
    """Draw and freeze one episode using the environment's canonical RNG order.

    Trailer length is drawn before the initial pose, matching
    :class:`~trailer_docking.training.environment.ScenarioDockingEnv`.
    Planning uses the shared initial geometry/state, so both backends receive
    byte-for-byte identical path points.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("episode seed must be a non-negative integer")
    rng = random.Random(seed)
    episode_scenario = scenario
    bounds = scenario.randomization.trailer_hitch_to_axle
    if bounds is not None:
        length = rng.uniform(*bounds)
        episode_scenario = replace(
            scenario,
            overrides=replace(
                scenario.overrides, trailer_hitch_to_axle=length
            ),
            randomization=replace(
                scenario.randomization, trailer_hitch_to_axle=None
            ),
        )
    built = episode_scenario.build()
    tractor_pose, trailer_yaw = sample_start(scenario, built, rng)
    steering_angle = 0.0
    state = KinematicSimulator(built.vehicle).reset(
        tractor_pose, trailer_yaw, steering_angle=steering_angle
    )
    reference_planner = sample_reference_planner(scenario, rng)
    reference_path = plan_reference(
        reference_planner,
        built.vehicle.geometry,
        built.dock.geometry,
        state,
        built.obstacles,
    )
    content: dict[str, Any] = {
        "seed": seed,
        "scenario_fingerprint": scenario.fingerprint,
        "tractor_pose": tractor_pose.to_dict(),
        "trailer_yaw": trailer_yaw,
        "steering_angle": steering_angle,
        "trailer_hitch_to_axle": built.vehicle.trailer.hitch_to_axle,
        "reference_path": [point.to_dict() for point in reference_path.points],
        "reference_planner": reference_planner,
    }
    return ResolvedEpisode(
        seed=seed,
        scenario_fingerprint=scenario.fingerprint,
        tractor_pose=tractor_pose,
        trailer_yaw=trailer_yaw,
        steering_angle=steering_angle,
        trailer_hitch_to_axle=built.vehicle.trailer.hitch_to_axle,
        reference_path=reference_path,
        episode_fingerprint=_fingerprint(content),
        reference_planner=reference_planner,
    )


def resolve_episodes(
    scenario: Scenario, *, episodes: int, seed: int
) -> tuple[ResolvedEpisode, ...]:
    """Resolve consecutive seeds into an immutable paired evaluation set."""

    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    return tuple(resolve_episode(scenario, seed + index) for index in range(episodes))
