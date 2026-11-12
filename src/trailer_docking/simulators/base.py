"""Common reset, step, state, and timing protocol for simulator backends."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from trailer_docking.core.state import VehicleCommand, VehicleState


@runtime_checkable
class SimulatorBackend(Protocol):
    """Minimum interface used by controllers, tasks, and evaluation code.

    Backends may expose richer reset arguments and diagnostics, but every
    backend must support a no-argument reset and one-command control step.
    """

    @property
    def controller_period(self) -> float:
        """Duration represented by one :meth:`step` call, in seconds."""

        ...

    @property
    def state(self) -> VehicleState:
        """Current shared planar state."""

        ...

    @property
    def applied_command(self) -> VehicleCommand:
        """Command that the plant accepted during the previous step."""

        ...

    def reset(self) -> VehicleState:
        """Reset to the backend's default initial state."""

        ...

    def step(self, command: VehicleCommand) -> VehicleState:
        """Advance exactly one controller period."""

        ...
