"""Control methods the campaign runner, CLI, and UI can select.

``ppo`` is trained here.  ``nmpc`` is the model-based baseline; it is not
trained, so selecting it skips training and only tests the controller on the
same scenario, seeds, and simulators as a PPO policy.

The model-based controllers live under :mod:`trailer_docking.controllers`.
This module exposes their per-episode factories to the shared evaluator::

    def make_docking_controller(env) -> Controller

called once at the start of every episode, after the environment was reset.
``env`` is the :class:`~trailer_docking.training.environment.ScenarioDockingEnv`;
it exposes the shared vehicle state (``env.state``), the reference path
(``env.reference_path``), and the built vehicle, dock, and task
(``env.built``). The evaluator constructs a ``ControllerContext`` for every
step and validates the same normalized action contract for every method.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trailer_docking.controllers.base import Controller

METHOD_PPO = "ppo"
METHOD_NMPC = "nmpc"
METHODS = (METHOD_PPO, METHOD_NMPC)
NMPC_MODULE = "trailer_docking.controllers.nmpc"
NMPC_HOOK = "make_docking_controller"
LQR_MODULE = "trailer_docking.controllers.lqr"
LQR_HOOK = "make_docking_controller"

ControllerFactory = Callable[[Any], "Controller"]
"""``factory(env)`` returns one controller for the resolved episode."""


class MethodUnavailableError(RuntimeError):
    """Raised when a selected control method has no implementation yet."""


def validate_method(value: object) -> str:
    """Return *value* if it names a control method, else raise."""

    if value not in METHODS:
        raise ValueError(f"method must be one of {', '.join(METHODS)}")
    return str(value)


def nmpc_controller_factory(configuration: Any = None) -> ControllerFactory:
    """Return the implemented NMPC hook, with a clear optional-dependency error."""

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
    if configuration is None:
        hook: ControllerFactory = factory
        return hook
    return lambda env: factory(env, configuration)


def lqr_controller_factory(configuration: Any = None) -> ControllerFactory:
    """Return the implemented trajectory-error LQR episode factory."""

    module = import_module(LQR_MODULE)
    factory = getattr(module, LQR_HOOK, None)
    if not callable(factory):
        raise MethodUnavailableError(
            f"LQR is not implemented: {LQR_MODULE} does not provide"
            f" {LQR_HOOK}(env)"
        )
    if configuration is None:
        hook: ControllerFactory = factory
        return hook
    return lambda env: factory(env, configuration)
