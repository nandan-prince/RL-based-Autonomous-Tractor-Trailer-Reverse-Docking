"""Observation-only PPO inference adapter for the shared controller layer."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from trailer_docking.controllers.base import (
    ControllerAction,
    ControllerContext,
    validate_controller_action,
)

Policy = Callable[[Any], Any]


class PPOController:
    """Expose a deterministic saved policy through the shared interface.

    Only ``context.observation`` is passed to the policy. The physical state,
    path, tracking object, and vehicle model remain unavailable to PPO.
    """

    def __init__(self, policy: Policy, *, name: str = "ppo-policy") -> None:
        if not callable(policy):
            raise TypeError("policy must be callable")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        self.policy = policy
        self._name = name.strip()

    @property
    def name(self) -> str:
        return self._name

    @property
    def display_name(self) -> str:
        return "PROXIMAL POLICY OPTIMIZATION (PPO)"

    def reset(self) -> None:
        """The current feed-forward PPO policy has no episode memory."""

    def act(self, context: ControllerContext) -> ControllerAction:
        return validate_controller_action(self.policy(context.observation))
