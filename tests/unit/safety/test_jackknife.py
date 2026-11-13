"""Shared articulation-threshold tests."""

from trailer_docking.core.state import Pose2D
from trailer_docking.safety.jackknife import articulation_status


def test_articulation_regions(vehicle_config) -> None:
    geometry = vehicle_config.geometry
    limits = vehicle_config.limits
    safe = articulation_status(
        geometry.assembled_state(Pose2D(0, 0, 0.2), 0), limits
    )
    warning = articulation_status(
        geometry.assembled_state(
            Pose2D(0, 0, limits.articulation_soft_abs_max), 0
        ),
        limits,
    )
    terminal = articulation_status(
        geometry.assembled_state(
            Pose2D(0, 0, -limits.articulation_hard_abs_max), 0
        ),
        limits,
    )
    assert not safe.soft_limit_reached and not safe.jackknifed
    assert warning.soft_limit_reached and not warning.jackknifed
    assert terminal.soft_limit_reached and terminal.jackknifed
