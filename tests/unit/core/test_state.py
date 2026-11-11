import math

import pytest

from trailer_docking.core.state import (
    Point2D,
    Pose2D,
    StateValidationError,
    VehicleCommand,
    VehicleState,
    wrap_angle,
)


@pytest.mark.parametrize(
    ("angle", "expected"),
    [
        (0.0, 0.0),
        (math.pi, -math.pi),
        (-math.pi, -math.pi),
        (3.0 * math.pi, -math.pi),
        (-1.5 * math.pi, 0.5 * math.pi),
        (2.0 * math.pi, 0.0),
    ],
)
def test_wrap_angle_half_open_interval(angle: float, expected: float) -> None:
    assert wrap_angle(angle) == pytest.approx(expected)


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan, True, "1.0"])
def test_primitives_reject_non_finite_or_non_numeric_values(value: object) -> None:
    with pytest.raises(StateValidationError):
        Point2D(value, 0.0)  # type: ignore[arg-type]
    with pytest.raises(StateValidationError):
        Pose2D(0.0, 0.0, value)  # type: ignore[arg-type]


def test_pose_normalizes_yaw_and_serializes_exactly() -> None:
    pose = Pose2D(1, 2, 2.5 * math.pi)
    assert pose == Pose2D(1.0, 2.0, 0.5 * math.pi)
    assert Pose2D.from_dict(pose.to_dict()) == pose


def test_point_and_pose_reject_unknown_serialized_fields() -> None:
    with pytest.raises(StateValidationError, match="only x and y"):
        Point2D.from_dict({"x": 0, "y": 0, "z": 0})
    with pytest.raises(StateValidationError, match="only x, y, and yaw"):
        Pose2D.from_dict({"x": 0, "y": 0})


def test_command_preserves_raw_out_of_limit_values() -> None:
    command = VehicleCommand(steering_rate=99.0, target_speed=-42.0)
    assert command.steering_rate == 99.0
    assert command.target_speed == -42.0
    assert VehicleCommand.from_dict(command.to_dict()) == command


def test_command_rejects_non_finite_and_extra_fields() -> None:
    with pytest.raises(StateValidationError):
        VehicleCommand(math.nan, 0.0)
    with pytest.raises(StateValidationError, match="only steering_rate"):
        VehicleCommand.from_dict(
            {"steering_rate": 0.0, "target_speed": 0.0, "hidden": 1.0}
        )


def _state() -> VehicleState:
    return VehicleState(
        tractor_pose=Pose2D(1.0, 2.0, 0.3),
        trailer_pose=Pose2D(-2.0, 1.0, -0.2),
        hitch_position=Point2D(0.5, 1.5),
        steering_angle=0.1,
        steering_rate=-0.02,
        longitudinal_speed=-0.2,
        tractor_yaw_rate=0.04,
        trailer_yaw_rate=-0.01,
        simulation_time=1.5,
    )


def test_state_derives_articulation_and_round_trips() -> None:
    state = _state()
    assert state.articulation_angle == pytest.approx(0.5)
    assert state.articulation_rate == pytest.approx(0.05)
    assert VehicleState.from_dict(state.to_dict()) == state
    assert "articulation_angle" not in state.to_dict(include_derived=False)


def test_state_rejects_inconsistent_serialized_articulation() -> None:
    data = _state().to_dict()
    data["articulation_angle"] = 1.0
    with pytest.raises(StateValidationError, match="inconsistent"):
        VehicleState.from_dict(data)

    data = _state().to_dict()
    data["articulation_rate"] = -1.0
    with pytest.raises(StateValidationError, match="inconsistent"):
        VehicleState.from_dict(data)


def test_state_rejects_negative_time_and_bad_mapping_shape() -> None:
    values = _state().to_dict(include_derived=False)
    values["simulation_time"] = -0.1
    with pytest.raises(StateValidationError, match="cannot be negative"):
        VehicleState.from_dict(values)

    values = _state().to_dict(include_derived=False)
    del values["trailer_pose"]
    with pytest.raises(StateValidationError, match="missing"):
        VehicleState.from_dict(values)


def test_state_rejects_wrong_domain_object_types() -> None:
    state = _state()
    with pytest.raises(StateValidationError, match="tractor_pose"):
        VehicleState(
            tractor_pose={"x": 0, "y": 0, "yaw": 0},  # type: ignore[arg-type]
            trailer_pose=state.trailer_pose,
            hitch_position=state.hitch_position,
            steering_angle=0.0,
            steering_rate=0.0,
            longitudinal_speed=0.0,
            tractor_yaw_rate=0.0,
            trailer_yaw_rate=0.0,
            simulation_time=0.0,
        )
