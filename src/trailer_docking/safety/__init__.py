"""Safety checks and command supervision independent of the simulator."""

from trailer_docking.safety.collision import (
    CollisionReport,
    dock_wall_polygons,
    polygons_intersect,
    trailer_inside_bay,
    vehicle_dock_collision,
    vehicle_inside_workspace,
)
from trailer_docking.safety.jackknife import ArticulationStatus, articulation_status

__all__ = [
    "ArticulationStatus",
    "CollisionReport",
    "articulation_status",
    "dock_wall_polygons",
    "polygons_intersect",
    "trailer_inside_bay",
    "vehicle_dock_collision",
    "vehicle_inside_workspace",
]
