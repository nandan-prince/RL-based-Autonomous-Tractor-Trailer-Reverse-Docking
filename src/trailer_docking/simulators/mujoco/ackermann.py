"""Ackermann mapping between the virtual bicycle angle and physical wheels.

The shared steering angle is the virtual front-wheel angle of the kinematic
bicycle model.  MuJoCo steers two physical wheels, so the adapter converts in
both directions with the equations frozen in the vehicle specification:

    R           = wheelbase / tan(delta)
    delta_left  = atan(wheelbase / (R - front_track / 2))
    delta_right = atan(wheelbase / (R + front_track / 2))

The functions below use the algebraically equivalent tangent form, which has
no singularity at zero steering.  This module deliberately has no MuJoCo
dependency so it can be unit-tested everywhere.
"""

from __future__ import annotations

from math import atan, cos, tan


def ackermann_wheel_angles(
    virtual_angle: float, wheelbase: float, front_track: float
) -> tuple[float, float]:
    """Return ``(left, right)`` physical steering angles in radians."""

    half_track = front_track / 2.0
    tangent = tan(virtual_angle)
    left = atan(wheelbase * tangent / (wheelbase - half_track * tangent))
    right = atan(wheelbase * tangent / (wheelbase + half_track * tangent))
    return left, right


def _virtual_tangents(
    left: float, right: float, wheelbase: float, front_track: float
) -> tuple[float, float]:
    half_track = front_track / 2.0
    tan_left = tan(left)
    tan_right = tan(right)
    from_left = wheelbase * tan_left / (wheelbase + half_track * tan_left)
    from_right = wheelbase * tan_right / (wheelbase - half_track * tan_right)
    return from_left, from_right


def virtual_steering_angle(
    left: float, right: float, wheelbase: float, front_track: float
) -> float:
    """Reconstruct the virtual bicycle angle from both physical wheel angles.

    Each wheel yields one estimate of ``tan(delta)``; the two are averaged so a
    small tracking error on one wheel does not bias the reported state.
    """

    from_left, from_right = _virtual_tangents(left, right, wheelbase, front_track)
    return atan((from_left + from_right) / 2.0)


def virtual_steering_rate(
    left: float,
    right: float,
    left_rate: float,
    right_rate: float,
    wheelbase: float,
    front_track: float,
) -> float:
    """Return the time derivative of :func:`virtual_steering_angle`."""

    half_track = front_track / 2.0
    from_left, from_right = _virtual_tangents(left, right, wheelbase, front_track)
    mean_tangent = (from_left + from_right) / 2.0
    left_gain = (wheelbase / (cos(left) * (wheelbase + half_track * tan(left)))) ** 2
    right_gain = (wheelbase / (cos(right) * (wheelbase - half_track * tan(right)))) ** 2
    tangent_rate = (left_gain * left_rate + right_gain * right_rate) / 2.0
    return tangent_rate / (1.0 + mean_tangent**2)
