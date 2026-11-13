"""Gymnasium-compatible task environments."""

from gymnasium.envs.registration import register, registry

from trailer_docking.core.config import (
    DockingTaskConfiguration as DockingTaskSettings,
)
from trailer_docking.environments.docking_env import TrailerDockingEnv

if "TrailerDocking-v0" not in registry:
    register(
        id="TrailerDocking-v0",
        entry_point="trailer_docking.environments:TrailerDockingEnv",
        max_episode_steps=None,
    )

__all__ = ["DockingTaskSettings", "TrailerDockingEnv"]
