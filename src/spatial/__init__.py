"""Spatial package for ARC reasoning."""

from src.spatial.dsl import (
    connect_matching_objects_op,
    connect_points_op,
    connect_same_color_op,
    draw_line_op,
    extend_until_collision_op,
    fill_between_op,
    fill_enclosed_op,
    flood_fill_op,
    project_until_boundary_op,
    project_until_object_op,
    raycast_op,
    trace_boundary_op,
)
from src.spatial.raycast import RaycastResult, raycast
from src.spatial.relations import (
    between,
    collinear,
    diagonal_alignment,
    enclosed_by,
    line_pixels_between,
    obstacle_between,
    ray_intersects,
    visible_from,
)

__all__ = [
    # Raycast
    "raycast",
    "RaycastResult",
    # Relations
    "collinear",
    "diagonal_alignment",
    "between",
    "enclosed_by",
    "line_pixels_between",
    "visible_from",
    "obstacle_between",
    "ray_intersects",
    # DSL Operators
    "flood_fill_op",
    "fill_enclosed_op",
    "trace_boundary_op",
    "draw_line_op",
    "raycast_op",
    "connect_points_op",
    "connect_same_color_op",
    "connect_matching_objects_op",
    "extend_until_collision_op",
    "fill_between_op",
    "project_until_boundary_op",
    "project_until_object_op",
]
