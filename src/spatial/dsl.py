"""Deterministic Spatial Domain-Specific Language (DSL) for ARC reasoning.

Implements interpretable spatial operators:
  - flood_fill_op
  - fill_enclosed_op
  - trace_boundary_op
  - draw_line_op
  - raycast_op
  - connect_points_op
  - connect_same_color_op
  - connect_matching_objects_op
  - extend_until_collision_op
  - fill_between_op
  - project_until_boundary_op
  - project_until_object_op
"""

from __future__ import annotations

from typing import Sequence

from src.data.models import Grid
from src.objects.segmentation import segment_grid
from src.spatial.raycast import raycast
from src.spatial.relations import line_pixels_between
from src.topology.engine import boundary_trace, enclosure_detection, flood_fill


def flood_fill_op(
    grid: Grid,
    start: tuple[int, int],
    fill_color: int,
    target_color: int | None = None,
    connectivity: int = 4,
) -> Grid:
    """Flood-fill connected region starting at start coordinate."""
    return flood_fill(grid, start, fill_color=fill_color, target_color=target_color, connectivity=connectivity)


def fill_enclosed_op(
    grid: Grid,
    fill_color: int,
    background: int = 0,
) -> Grid:
    """Fill all enclosed background cavities with fill_color."""
    enclosures = enclosure_detection(grid, background=background)
    if not enclosures:
        return grid

    cells = grid.to_list()
    for enc in enclosures:
        for r, c in enc["enclosed_coords"]:
            cells[r][c] = fill_color

    return Grid.from_list(cells)


def trace_boundary_op(
    grid: Grid,
    target_color: int | None = None,
    boundary_color: int = 1,
    background: int = 0,
) -> Grid:
    """Trace and recolor outer boundary pixels of non-background objects to boundary_color."""
    hyp = segment_grid(grid, background=background)
    cells = grid.to_list()

    for obj in hyp.objects:
        if target_color is not None and obj.color != target_color:
            continue
        boundary_pixels = boundary_trace(grid, obj.pixel_coords)
        for r, c in boundary_pixels:
            cells[r][c] = boundary_color

    return Grid.from_list(cells)


def draw_line_op(
    grid: Grid,
    p1: tuple[int, int],
    p2: tuple[int, int],
    color: int,
) -> Grid:
    """Draw a straight line between p1 and p2 using Bresenham's line algorithm."""
    line_pixels = line_pixels_between(p1, p2)
    cells = grid.to_list()
    h, w = grid.height, grid.width

    for r, c in line_pixels:
        if 0 <= r < h and 0 <= c < w:
            cells[r][c] = color

    return Grid.from_list(cells)


def raycast_op(
    grid: Grid,
    start: tuple[int, int],
    direction: str,
    fill_color: int,
    stop_condition: str = "boundary",
    background: int = 0,
) -> Grid:
    """Cast a ray from start position and paint traversed path with fill_color."""
    res = raycast(grid, start=start, direction=direction, stop_condition=stop_condition, background=background)
    cells = grid.to_list()
    for r, c in res.path:
        cells[r][c] = fill_color
    return Grid.from_list(cells)


def connect_points_op(
    grid: Grid,
    points: Sequence[tuple[int, int]],
    color: int,
    close_loop: bool = False,
) -> Grid:
    """Connect a sequence of (r, c) points with straight lines."""
    if len(points) < 2:
        return grid

    curr_grid = grid
    for i in range(len(points) - 1):
        curr_grid = draw_line_op(curr_grid, points[i], points[i + 1], color)

    if close_loop and len(points) >= 3:
        curr_grid = draw_line_op(curr_grid, points[-1], points[0], color)

    return curr_grid


def connect_same_color_op(
    grid: Grid,
    target_color: int,
    line_color: int | None = None,
    background: int = 0,
) -> Grid:
    """Connect all landmark pixels of target_color in sequence or pairwise."""
    paint_color = target_color if line_color is None else line_color
    coords = []
    for r in range(grid.height):
        for c in range(grid.width):
            if grid.get(r, c) == target_color:
                coords.append((r, c))

    if len(coords) < 2:
        return grid

    return connect_points_op(grid, coords, paint_color)


def connect_matching_objects_op(
    grid: Grid,
    line_color: int = 1,
    background: int = 0,
) -> Grid:
    """Connect centroids of objects sharing identical shape signatures."""
    hyp = segment_grid(grid, background=background)
    objs = hyp.objects
    if len(objs) < 2:
        return grid

    # Group by shape signature
    groups: dict[tuple[tuple[int, ...], ...], list[tuple[int, int]]] = {}
    for obj in objs:
        centroid_int = (int(round(obj.centroid[0])), int(round(obj.centroid[1])))
        groups.setdefault(obj.shape_signature, []).append(centroid_int)

    curr_grid = grid
    for sig, centroids in groups.items():
        if len(centroids) >= 2:
            curr_grid = connect_points_op(curr_grid, centroids, line_color)

    return curr_grid


def extend_until_collision_op(
    grid: Grid,
    target_color: int,
    direction: str,
    fill_color: int | None = None,
    background: int = 0,
) -> Grid:
    """Extend all objects of target_color along direction until collision with another object or grid edge."""
    paint_color = target_color if fill_color is None else fill_color
    hyp = segment_grid(grid, background=background)
    objs = [o for o in hyp.objects if o.color == target_color]
    if not objs:
        return grid

    curr_grid = grid
    for obj in objs:
        for r, c in obj.pixel_coords:
            res = raycast_op(curr_grid, start=(r, c), direction=direction, fill_color=paint_color, stop_condition="non_background", background=background)
            curr_grid = res

    return curr_grid


def fill_between_op(
    grid: Grid,
    p1: tuple[int, int],
    p2: tuple[int, int],
    fill_color: int,
) -> Grid:
    """Fill line segment between p1 and p2 with fill_color."""
    return draw_line_op(grid, p1, p2, color=fill_color)


def project_until_boundary_op(
    grid: Grid,
    start: tuple[int, int],
    direction: str,
    fill_color: int,
) -> Grid:
    """Project ray from start to grid boundary with fill_color."""
    return raycast_op(grid, start=start, direction=direction, fill_color=fill_color, stop_condition="boundary")


def project_until_object_op(
    grid: Grid,
    start: tuple[int, int],
    direction: str,
    fill_color: int,
    background: int = 0,
) -> Grid:
    """Project ray from start until non-background object collision."""
    return raycast_op(grid, start=start, direction=direction, fill_color=fill_color, stop_condition="non_background", background=background)
