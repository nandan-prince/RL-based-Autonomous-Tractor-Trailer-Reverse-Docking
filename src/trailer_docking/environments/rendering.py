"""Top-down Gymnasium rendering without simulator-specific dependencies."""

from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from trailer_docking.core.config import DockConfiguration, VehicleConfiguration
from trailer_docking.core.geometry import Bounds1D
from trailer_docking.core.state import VehicleState
from trailer_docking.environments.task import TaskStatus
from trailer_docking.paths.reference import ReferencePath
from trailer_docking.paths.tracking import TrackingState
from trailer_docking.visualization.topdown import TopDownView


class DockingRenderer:
    """Add reference-path and task diagnostics to the shared top-down view."""

    def __init__(
        self,
        vehicle: VehicleConfiguration,
        dock: DockConfiguration,
        path: ReferencePath,
        mode: Literal["human", "rgb_array"],
        x_bounds: Bounds1D,
        y_bounds: Bounds1D,
    ) -> None:
        self.mode = mode
        title = (
            "Full-size service-center yard — Gymnasium 2D"
            if dock.geometry.specification_id == "service-center-dock-v1"
            else "Gymnasium 2D docking environment"
        )
        self.view = TopDownView(vehicle, dock, title=title)
        (self._path_line,) = self.view.axes.plot(
            [], [], "--", color="#69359c", linewidth=1.5, label="reference path"
        )
        (self._preview_points,) = self.view.axes.plot(
            [], [], "o", color="#00a6a6", markersize=4, label="preview points"
        )
        self.view.axes.legend(loc="lower right", fontsize=7, ncol=2)
        self.view.set_view_bounds(x_bounds, y_bounds)
        self.reset_path(path)
        if mode == "human":
            import matplotlib.pyplot as plt

            if "agg" not in plt.get_backend().lower():
                plt.show(block=False)

    def reset_path(self, path: ReferencePath) -> None:
        points = [self.view.display_xy(point) for point in path.points]
        self._path_line.set_data(
            [point[0] for point in points], [point[1] for point in points]
        )
        self.view.clear_trail()

    def render(
        self,
        state: VehicleState,
        tracking: TrackingState,
        status: TaskStatus,
        reward: float,
    ) -> NDArray[np.uint8] | None:
        samples = tracking.preview_samples
        preview_points = [self.view.display_xy(sample.point) for sample in samples]
        self._preview_points.set_data(
            [point[0] for point in preview_points],
            [point[1] for point in preview_points],
        )
        note = (
            f"reward={reward:+.3f}  status={status.reason or 'running'}\n"
            f"path lateral={tracking.lateral_error:+.3f} m  "
            f"remaining={tracking.distance_to_goal:.3f} m"
        )
        self.view.update(state, note=note)
        self.view.figure.canvas.draw()
        if self.mode == "human":
            self.view.figure.canvas.flush_events()
            return None
        canvas = cast(Any, self.view.figure.canvas)
        rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
        return rgba[:, :, :3].copy()

    def close(self) -> None:
        import matplotlib.pyplot as plt

        plt.close(self.view.figure)
