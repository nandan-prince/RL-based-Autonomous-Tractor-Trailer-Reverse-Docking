"""Offscreen MuJoCo rendering for views that embed the 3D scene.

The control UI shows the 3D simulator inside its own window, so it needs
images rather than MuJoCo's separate interactive viewer.  A renderer owns an
OpenGL context that belongs to the thread that created it; create, use, and
close one from the same thread.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from typing import Any

import numpy as np
from numpy.typing import NDArray

from trailer_docking.core.config import DockConfiguration, VehicleConfiguration
from trailer_docking.core.state import Point2D, Pose2D

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_AZIMUTH = 70.0
CAMERA_ELEVATION = -28.0
# Camera distance per metre of scene, for the 4:3 frame and default lens.
DISTANCE_PER_METRE = 1.25


ROUTE_COLOR = (0.1, 0.75, 0.25, 1.0)
ROUTE_RADIUS = 0.04


class DisplayOptions:
    """What the embedded 3D picture shows; set by the view, read by renderers.

    The picture is drawn on the thread that runs the simulation, while the
    choice is made in the window, so the option lives here rather than being
    passed through every training and test call.  Reading or writing one
    boolean is safe across threads.
    """

    show_route: bool = True


display = DisplayOptions()


def scene_length(
    vehicle: VehicleConfiguration, dock: DockConfiguration, standoff: float
) -> float:
    """Distance from the bay's rear wall to the tractor nose at a start.

    *standoff* is how far outside the opening the trailer's docking point
    starts, so this is the length a camera must cover to show a whole episode.
    """

    trailer = vehicle.trailer
    tractor = vehicle.tractor
    return float(
        dock.geometry.interior_depth
        + standoff
        + trailer.rear_overhang
        + trailer.hitch_to_axle
        - tractor.hitch_longitudinal
        + tractor.footprint_x.maximum
    )


class MujocoFrameRenderer:
    """Render RGB frames of one compiled MuJoCo model.

    The camera looks at the middle of the stretch from the bay's rear wall to
    ``scene_length`` metres out along the bay axis, from far enough away to
    keep all of it in view, so the vehicle is in frame from the first step to
    the last whatever the truck, bay, and start distance are.
    """

    def __init__(
        self,
        model: Any,
        *,
        scene_length: float = 18.0,
        width: int = FRAME_WIDTH,
        height: int = FRAME_HEIGHT,
    ) -> None:
        import mujoco

        self._camera = mujoco.MjvCamera()
        self._camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._camera.lookat[:] = (scene_length / 2.0, 0.0, 0.5)
        self._camera.distance = max(scene_length, 4.0) * DISTANCE_PER_METRE
        self._camera.azimuth = CAMERA_AZIMUTH
        self._camera.elevation = CAMERA_ELEVATION
        self._renderer = mujoco.Renderer(model, height=height, width=width)

    def render(
        self, data: Any, route: Sequence[Point2D] | None = None
    ) -> NDArray[np.uint8]:
        """Return a ``(height, width, 3)`` image of the current simulation.

        *route* is drawn on the ground as a green line.  It exists only in
        the picture and never enters the physics model.
        """

        import mujoco

        self._renderer.update_scene(data, camera=self._camera)
        scene = self._renderer.scene
        color = np.array(ROUTE_COLOR, dtype=np.float32)
        for start, end in pairwise(route or ()):
            if scene.ngeom >= scene.maxgeom:
                break
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.zeros(3),
                np.zeros(3),
                np.zeros(9),
                color,
            )
            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                ROUTE_RADIUS,
                np.array([start.x, start.y, 0.02]),
                np.array([end.x, end.y, 0.02]),
            )
            scene.ngeom += 1
        frame: NDArray[np.uint8] = self._renderer.render().copy()
        return frame

    def close(self) -> None:
        self._renderer.close()


def render_scene(
    vehicle: VehicleConfiguration,
    dock: DockConfiguration,
    tractor_pose: Pose2D | None = None,
    trailer_yaw: float | None = None,
    *,
    standoff: float = 6.0,
    route: Sequence[Point2D] | None = None,
) -> NDArray[np.uint8]:
    """Render one still image of a vehicle and dock at a start pose."""

    from trailer_docking.simulators.mujoco.adapter import MujocoSimulator

    simulator = MujocoSimulator(vehicle, dock)
    simulator.reset(tractor_pose, trailer_yaw)
    renderer = MujocoFrameRenderer(
        simulator.model, scene_length=scene_length(vehicle, dock, standoff)
    )
    try:
        return renderer.render(simulator.data, route)
    finally:
        renderer.close()
