"""Equation and integration tests for the canonical kinematic model."""

from dataclasses import replace
from math import cos, pi, sin, tan

import pytest

from trailer_docking.core.config import VehicleConfiguration
from trailer_docking.core.geometry import VehicleGeometry
from trailer_docking.simulators.kinematic.model import (
    KinematicInput,
    KinematicModel,
    KinematicModelError,
    PlanarKinematicState,
)


def test_on_axle_derivative_matches_frozen_equations(
    vehicle_config: VehicleConfiguration,
) -> None:
    geometry = vehicle_config.geometry
    model = KinematicModel(geometry)
    state = PlanarKinematicState(1.0, -2.0, 0.4, -0.2, 0.3, -0.2, 1.5)
    control = KinematicInput(-0.1, 0.25)
    derivative = model.derivative(state, control)
    articulation = state.tractor_yaw - state.trailer_yaw
    assert derivative.tractor_x == pytest.approx(-0.2 * cos(0.4))
    assert derivative.tractor_y == pytest.approx(-0.2 * sin(0.4))
    assert derivative.tractor_yaw == pytest.approx(
        -0.2 * tan(0.3) / geometry.tractor.wheelbase
    )
    assert derivative.trailer_yaw == pytest.approx(
        -0.2 * sin(articulation) / geometry.trailer.hitch_to_axle
    )
    assert derivative.steering_angle == -0.1
    assert derivative.longitudinal_speed == 0.25


def test_signed_hitch_offsets_are_present_in_trailer_rate(
    vehicle_config: VehicleConfiguration,
) -> None:
    base = vehicle_config.geometry
    tractor = replace(base.tractor, hitch_longitudinal=0.35, hitch_lateral=-0.12)
    geometry = VehicleGeometry(tractor, base.trailer, base.wheels)
    model = KinematicModel(geometry)
    state = PlanarKinematicState(0.0, 0.0, 0.5, -0.1, 0.25, 0.18)
    derivative = model.derivative(state, KinematicInput(0.0, 0.0))
    tractor_rate = 0.18 * tan(0.25) / tractor.wheelbase
    articulation = 0.6
    expected = (
        (0.18 - tractor.hitch_lateral * tractor_rate) * sin(articulation)
        + tractor.hitch_longitudinal * tractor_rate * cos(articulation)
    ) / geometry.trailer.hitch_to_axle
    assert derivative.trailer_yaw == pytest.approx(expected)
    propagated = model.rk4_step(state, KinematicInput(0.1, -0.05), 0.02)
    shared = model.shared_state(propagated, steering_rate=0.1)
    assert max(geometry.constraint_residuals(shared).values()) < 1e-12


def test_stationary_derivative_and_rk4_step_do_not_move(
    vehicle_config: VehicleConfiguration,
) -> None:
    model = KinematicModel(vehicle_config.geometry)
    state = PlanarKinematicState(2.0, -3.0, 1.2, -0.7, 0.0, 0.0)
    derivative = model.derivative(state, KinematicInput(0.0, 0.0))
    assert derivative.tractor_x == 0.0
    assert derivative.tractor_y == 0.0
    assert derivative.tractor_yaw == 0.0
    assert derivative.trailer_yaw == 0.0
    stepped = model.rk4_step(state, KinematicInput(0.0, 0.0), 0.02)
    assert stepped == replace(state, simulation_time=0.02)


def test_rk4_matches_analytic_tractor_circle(
    vehicle_config: VehicleConfiguration,
) -> None:
    geometry = vehicle_config.geometry
    model = KinematicModel(geometry)
    steering, speed, duration, dt = 0.3, 0.2, 4.0, 0.02
    state = PlanarKinematicState(0.0, 0.0, 0.0, 0.0, steering, speed)
    for _ in range(round(duration / dt)):
        state = model.rk4_step(state, KinematicInput(0.0, 0.0), dt)
    yaw_rate = speed * tan(steering) / geometry.tractor.wheelbase
    radius = speed / yaw_rate
    expected_yaw = yaw_rate * duration
    assert state.tractor_x == pytest.approx(radius * sin(expected_yaw), abs=1e-10)
    assert state.tractor_y == pytest.approx(
        radius * (1.0 - cos(expected_yaw)), abs=1e-10
    )
    assert state.tractor_yaw == pytest.approx(expected_yaw, abs=1e-12)


def test_rk4_wraps_yaws_without_changing_motion(
    vehicle_config: VehicleConfiguration,
) -> None:
    model = KinematicModel(vehicle_config.geometry)
    state = PlanarKinematicState(0.0, 0.0, pi - 0.001, pi - 0.001, 0.4, 0.2)
    for _ in range(10):
        state = model.rk4_step(state, KinematicInput(0.0, 0.0), 0.02)
    assert -pi <= state.tractor_yaw < pi
    assert -pi <= state.trailer_yaw < pi
    assert state.tractor_x < 0.0


@pytest.mark.parametrize(
    "build",
    [
        lambda model: KinematicModel(object()),  # type: ignore[arg-type]
        lambda model: PlanarKinematicState(0, 0, 0, 0, float("nan"), 0),
        lambda model: PlanarKinematicState(0, 0, 0, 0, 0, 0, -1),
        lambda model: PlanarKinematicState(10**400, 0, 0, 0, 0, 0),
        lambda model: KinematicInput(True, 0),
        lambda model: model.rk4_step(
            PlanarKinematicState(0, 0, 0, 0, 0, 0),
            KinematicInput(0, 0),
            0,
        ),
        lambda model: model.derivative(  # type: ignore[arg-type]
            object(), KinematicInput(0, 0)
        ),
        lambda model: model.derivative(  # type: ignore[arg-type]
            PlanarKinematicState(0, 0, 0, 0, 0, 0), object()
        ),
    ],
)
def test_invalid_model_inputs_fail_explicitly(
    vehicle_config: VehicleConfiguration, build: object
) -> None:
    model = KinematicModel(vehicle_config.geometry)
    with pytest.raises(KinematicModelError):
        build(model)  # type: ignore[operator]


def test_shared_state_reconstructs_exact_rigid_geometry(
    vehicle_config: VehicleConfiguration,
) -> None:
    model = KinematicModel(vehicle_config.geometry)
    internal = PlanarKinematicState(2.0, -1.0, 0.7, -0.2, 0.25, -0.15, 3.0)
    state = model.shared_state(internal, steering_rate=0.1)
    residuals = vehicle_config.geometry.constraint_residuals(state)
    assert max(residuals.values()) < 1e-12
    assert state.tractor_pose.yaw == pytest.approx(0.7)
    assert state.trailer_pose.yaw == pytest.approx(-0.2)
    assert state.steering_rate == 0.1
    assert state.longitudinal_speed == -0.15
    assert state.simulation_time == 3.0
