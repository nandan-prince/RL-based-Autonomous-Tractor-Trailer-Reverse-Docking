"""Common reset, step, state, and timing protocol for simulator backends.

Every backend, 2D or MuJoCo, exposes this interface.  Environments,
controllers, evaluation, and the UI depend only on it, which is what lets a
policy trained against one backend be run against another unchanged.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from trailer_docking.core.state import Pose2D, VehicleCommand, VehicleState


@runtime_checkable
class Simulator(Protocol):
    """A plant that advances the shared planar state one controller period."""

    @property
    def controller_period(self) -> float:
        """Seconds of simulated time advanced by one :meth:`step`."""

    @property
    def state(self) -> VehicleState:
        """The current shared planar state."""

    @property
    def applied_command(self) -> VehicleCommand:
        """What the plant effectively received during the last step."""

    def reset(
        self,
        tractor_pose: Pose2D | None = None,
        trailer_yaw: float | None = None,
        *,
        steering_angle: float = 0.0,
    ) -> VehicleState:
        """Place the vehicle at rest; arguments mirror ``assembled_state``."""

    def step(self, command: VehicleCommand) -> VehicleState:
        """Advance exactly one controller period under a held command."""
