"""Control methods the campaign runner, CLI, and UI can select.

``ppo`` is trained here.  ``nmpc`` is the model-based baseline; it is not
trained, so selecting it skips training and only tests the controller on the
same scenario, seeds, and simulators as a PPO policy.

The NMPC controller itself lives in :mod:`trailer_docking.controllers.nmpc`.
This module only defines how it is found, so that work can proceed
independently.  The expected hook is one function::

    def make_docking_controller(env) -> Callable[[observation], action]

called once at the start of every episode, after the environment was reset.
``env`` is the :class:`~trailer_docking.training.environment.ScenarioDockingEnv`;
it exposes the shared vehicle state (``env.state``), the reference path
(``env.reference_path``), and the built vehicle, dock, and task
(``env.built``).  The returned callable receives the policy observation and
may ignore it; it must return one normalized steering-rate action in
``[-1, 1]`` with shape ``(1,)``, the same bounded command PPO produces.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any

METHOD_PPO = "ppo"
METHOD_NMPC = "nmpc"
METHODS = (METHOD_PPO, METHOD_NMPC)
NMPC_MODULE = "trailer_docking.controllers.nmpc"
NMPC_HOOK = "make_docking_controller"

ControllerFactory = Callable[[Any], Callable[[Any], Any]]
"""``factory(env)`` returns the per-episode ``observation -> action`` callable."""


class MethodUnavailableError(RuntimeError):
    """Raised when a selected control method has no implementation yet."""


def validate_method(value: object) -> str:
    """Return *value* if it names a control method, else raise."""

    if value not in METHODS:
        raise ValueError(f"method must be one of {', '.join(METHODS)}")
    return str(value)


def nmpc_controller_factory() -> ControllerFactory:
    """Return the NMPC controller hook, or explain that it does not exist yet."""

    try:
        module = import_module(NMPC_MODULE)
    except ImportError as error:
        raise MethodUnavailableError(f"NMPC cannot be loaded: {error}") from error
    factory = getattr(module, NMPC_HOOK, None)
    if not callable(factory):
        raise MethodUnavailableError(
            f"NMPC is not implemented yet: {NMPC_MODULE} does not provide"
            f" {NMPC_HOOK}(env)"
        )
    hook: ControllerFactory = factory
    return hook
