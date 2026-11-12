"""Backend-independent 2D top-down view of the dock and vehicle.

The drawing uses only the shared planar contract: a :class:`VehicleState`, the
vehicle geometry, and the dock geometry.  It therefore shows exactly what a
controller sees and works for any simulator backend that produces the shared
state, including the future 2D kinematic simulator.

Matplotlib is an optional dependency (``pip install -e '.[viz]'``).
"""

from __future__ import annotations

from typing import Any

from trailer_docking.core.config import DockConfiguration, VehicleConfiguration
from trailer_docking.core.geometry import Bounds1D, local_to_world, rectangle_corners
from trailer_docking.core.state import Point2D, Pose2D, VehicleState
from trailer_docking.simulators.mujoco.ackermann import ackermann_wheel_angles

TRACTOR_COLOR = "#2a5cb8"
TRAILER_COLOR = "#c9c9cf"
WALL_COLOR = "#8c8c80"
WHEEL_COLOR = "#1f1f1f"


def _xy(points: tuple[Point2D, ...]) -> list[tuple[float, float]]:
    return [(point.x, point.y) for point in points]


class TopDownView:
    """A matplotlib figure that redraws the vehicle from a shared state."""

    def __init__(
        self,
        vehicle: VehicleConfiguration,
        dock: DockConfiguration | None = None,
        *,
        title: str | None = None,
    ) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon, Rectangle

        self.vehicle = vehicle
        self.dock = dock
        self._polygon = Polygon
        self.figure, self.axes = plt.subplots(figsize=(11, 5.5))
        self.axes.set_aspect("equal", adjustable="box")
        self.axes.set_xlabel("x [m]  (dock frame: +x outward from the rear wall)")
        self.axes.set_ylabel("y [m]")
        self.axes.grid(True, linewidth=0.4, alpha=0.5)
        self.axes.set_title(
            title
            or f"Top-down view: {vehicle.specification_id}"
            f" ({vehicle.trailer.variant} trailer)"
        )

        if dock is not None:
            bay = dock.geometry
            half_width = bay.interior_width / 2.0
            thickness = bay.wall_thickness
            self.axes.add_patch(
                Rectangle(
                    (0.0, -half_width),
                    bay.interior_depth,
                    bay.interior_width,
                    facecolor="#f2d94a",
                    alpha=0.25,
                    edgecolor="none",
                    label="bay interior",
                )
            )
            walls = (
                (
                    -thickness,
                    -half_width - thickness,
                    thickness,
                    2 * half_width + 2 * thickness,
                ),
                (0.0, half_width, bay.interior_depth, thickness),
                (0.0, -half_width - thickness, bay.interior_depth, thickness),
            )
            for index, (x, y, width, height) in enumerate(walls):
                self.axes.add_patch(
                    Rectangle(
                        (x, y),
                        width,
                        height,
                        facecolor=WALL_COLOR,
                        edgecolor="black",
                        linewidth=0.6,
                        label="dock wall" if index == 0 else None,
                    )
                )
            self.axes.plot(
                [bay.target_position.x],
                [bay.target_position.y],
                marker="x",
                color="green",
                markersize=10,
                markeredgewidth=2.5,
                linestyle="none",
                label="docking target",
            )

        self._trailer = Polygon(
            [(0, 0)] * 4,
            closed=True,
            facecolor=TRAILER_COLOR,
            edgecolor="black",
            alpha=0.85,
            label="trailer",
        )
        self._tractor = Polygon(
            [(0, 0)] * 4,
            closed=True,
            facecolor=TRACTOR_COLOR,
            edgecolor="black",
            alpha=0.9,
            label="tractor",
        )
        self.axes.add_patch(self._trailer)
        self.axes.add_patch(self._tractor)
        self._wheels = [
            Polygon([(0, 0)] * 4, closed=True, facecolor=WHEEL_COLOR, zorder=4)
            for _ in range(6)
        ]
        for wheel in self._wheels:
            self.axes.add_patch(wheel)
        (self._hitch,) = self.axes.plot(
            [],
            [],
            marker="o",
            color="orange",
            linestyle="none",
            label="hitch",
            zorder=5,
        )
        (self._docking_point,) = self.axes.plot(
            [],
            [],
            marker="o",
            color="red",
            linestyle="none",
            label="docking point",
            zorder=5,
        )
        (self._trail,) = self.axes.plot(
            [], [], color="red", linewidth=0.8, alpha=0.6, label="docking-point path"
        )
        self._trail_points: list[tuple[float, float]] = []
        self._readout = self.axes.text(
            0.01,
            0.98,
            "",
            transform=self.axes.transAxes,
            verticalalignment="top",
            family="monospace",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        self.axes.legend(loc="lower right", fontsize=7, ncol=2)
        self._limits_set = False

    def _wheel_corners(
        self, frame: Pose2D, x: float, y: float, steer: float
    ) -> list[tuple[float, float]]:
        wheels = self.vehicle.wheels
        center = local_to_world(Point2D(x, y), frame)
        return _xy(
            rectangle_corners(
                Bounds1D(-wheels.radius, wheels.radius),
                Bounds1D(-wheels.width / 2.0, wheels.width / 2.0),
                Pose2D(center.x, center.y, frame.yaw + steer),
            )
        )

    def clear_trail(self) -> None:
        self._trail_points.clear()

    def update(self, state: VehicleState, *, note: str = "") -> None:
        """Redraw the vehicle for one shared state."""

        tractor = self.vehicle.tractor
        trailer = self.vehicle.trailer
        self._tractor.set_xy(_xy(tractor.footprint(state.tractor_pose)))
        self._trailer.set_xy(_xy(trailer.footprint(state.trailer_pose)))
        left, right = ackermann_wheel_angles(
            state.steering_angle, tractor.wheelbase, tractor.front_track
        )
        wheel_specs = (
            (state.tractor_pose, tractor.wheelbase, tractor.front_track / 2.0, left),
            (state.tractor_pose, tractor.wheelbase, -tractor.front_track / 2.0, right),
            (state.tractor_pose, 0.0, tractor.rear_track / 2.0, 0.0),
            (state.tractor_pose, 0.0, -tractor.rear_track / 2.0, 0.0),
            (state.trailer_pose, 0.0, trailer.axle_track / 2.0, 0.0),
            (state.trailer_pose, 0.0, -trailer.axle_track / 2.0, 0.0),
        )
        for patch, (frame, x, y, steer) in zip(self._wheels, wheel_specs, strict=True):
            patch.set_xy(self._wheel_corners(frame, x, y, steer))
        self._hitch.set_data([state.hitch_position.x], [state.hitch_position.y])
        docking = trailer.rear_face(state.trailer_pose)
        self._docking_point.set_data([docking.x], [docking.y])
        self._trail_points.append((docking.x, docking.y))
        self._trail.set_data(*zip(*self._trail_points, strict=True))

        lines = [
            f"t = {state.simulation_time:7.2f} s",
            f"speed        = {state.longitudinal_speed:+.3f} m/s",
            f"steering     = {state.steering_angle:+.3f} rad",
            f"articulation = {state.articulation_angle:+.3f} rad",
            f"docking pt   = ({docking.x:+.3f}, {docking.y:+.3f}) m",
        ]
        if self.dock is not None:
            target = self.dock.geometry.target_position
            lines.append(
                f"dock error   = ({docking.x - target.x:+.3f},"
                f" {docking.y - target.y:+.3f}) m"
            )
        if note:
            lines.append(note)
        self._readout.set_text("\n".join(lines))

        if not self._limits_set:
            self._set_limits(state)
            self._limits_set = True

    def _set_limits(self, state: VehicleState) -> None:
        points = _xy(self.vehicle.tractor.footprint(state.tractor_pose)) + _xy(
            self.vehicle.trailer.footprint(state.trailer_pose)
        )
        if self.dock is not None:
            bay = self.dock.geometry
            reach = bay.interior_width / 2.0 + bay.wall_thickness
            points += [(-bay.wall_thickness, -reach), (bay.interior_depth, reach)]
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        margin = 2.5
        self.axes.set_xlim(min(xs) - margin, max(xs) + margin)
        self.axes.set_ylim(min(ys) - margin - 2.0, max(ys) + margin + 2.0)

    def draw(self) -> Any:
        """Flush pending drawing; returns the figure for saving in tests."""

        self.figure.canvas.draw_idle()
        return self.figure
