"""Reference path interpolation, projection, and tracking tests."""

from math import pi

import pytest

from trailer_docking.core.state import Point2D, Pose2D
from trailer_docking.paths import ReferencePath, ReferencePathError, tracking_state


def test_sampling_projection_and_signed_error() -> None:
    path = ReferencePath((Point2D(0, 0), Point2D(4, 0), Point2D(4, 3)))
    assert path.length == pytest.approx(7.0)
    assert path.sample(-2).point == Point2D(0, 0)
    assert path.sample(2).point == Point2D(2, 0)
    assert path.sample(5).point == Point2D(4, 1)
    assert path.sample(20).point == Point2D(4, 3)

    upper = path.nearest(Point2D(2, 1))
    lower = path.nearest(Point2D(2, -1))
    assert upper.sample.progress == pytest.approx(2.0)
    assert upper.signed_lateral_error == pytest.approx(1.0)
    assert lower.signed_lateral_error == pytest.approx(-1.0)
    right = path.nearest(Point2D(5, 2))
    assert right.sample.progress == pytest.approx(6.0)
    assert right.signed_lateral_error == pytest.approx(-1.0)


def test_preview_sampling_moves_toward_goal() -> None:
    path = ReferencePath((Point2D(0, 0), Point2D(10, 0)))
    previews = path.previews(6.0, (0.0, 1.0, 3.0, 20.0))
    assert tuple(sample.progress for sample in previews) == (6.0, 5.0, 3.0, 0.0)


def test_tracking_state_uses_docking_point_and_vehicle_heading(
    vehicle_config,
) -> None:
    state = vehicle_config.geometry.assembled_state(
        Pose2D(6, 1, 0.2), 0.1, steering_angle=0.2
    )
    docking = vehicle_config.trailer.rear_face(state.trailer_pose)
    path = ReferencePath((Point2D(0, 0), Point2D(10, 0)))
    tracking = tracking_state(state, path, docking, (0.5, 1.0))
    assert tracking.lateral_error == pytest.approx(docking.y)
    assert tracking.trailer_heading_error == pytest.approx(0.1)
    assert tracking.tractor_heading_error == pytest.approx(0.2)
    assert len(tracking.preview_points_trailer_frame) == 2


@pytest.mark.parametrize(
    "operation",
    [
        lambda: ReferencePath((Point2D(0, 0),)),
        lambda: ReferencePath((Point2D(0, 0), Point2D(0, 0))),
        lambda: ReferencePath((Point2D(0, 0), Point2D(1, 0))).sample(float("nan")),
        lambda: ReferencePath((Point2D(0, 0), Point2D(1, 0))).previews(1, (-1,)),
    ],
)
def test_invalid_paths_and_queries_fail(operation) -> None:
    with pytest.raises(ReferencePathError):
        operation()


def test_heading_is_segment_tangent() -> None:
    path = ReferencePath((Point2D(0, 0), Point2D(0, 2)))
    assert path.sample(1).heading == pytest.approx(pi / 2)
