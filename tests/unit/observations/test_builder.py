"""Trajectory-relative observation schema tests."""

import numpy as np

from trailer_docking.core.state import Point2D, Pose2D
from trailer_docking.environments.task import docking_errors
from trailer_docking.observations import ObservationBuilder
from trailer_docking.paths import ReferencePath, tracking_state


def test_observation_is_named_bounded_float32(vehicle_config, dock_config) -> None:
    state = vehicle_config.geometry.assembled_state(
        Pose2D(15, 0.3, 0.1), 0.05, steering_angle=0.2, steering_rate=-0.1
    )
    docking = vehicle_config.trailer.rear_face(state.trailer_pose)
    path = ReferencePath((dock_config.geometry.target_position, Point2D(10, 0)))
    previews = (0.5, 1.0, 2.0)
    tracking = tracking_state(state, path, docking, previews)
    errors = docking_errors(state, vehicle_config, dock_config.geometry)
    builder = ObservationBuilder(
        vehicle_config, dock_config.geometry, path, previews
    )
    observation = builder.build(
        state,
        tracking,
        errors.longitudinal,
        errors.lateral,
        errors.trailer_yaw,
    )
    assert observation.shape == builder.shape == (21,)
    assert observation.dtype == np.float32
    assert np.isfinite(observation).all()
    assert np.max(np.abs(observation)) <= 1.0
    assert builder.names[0] == "lateral_error"
    assert builder.names[-1] == "preview_2_y"
