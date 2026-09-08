"""Extended geometric and spatial predicates for ARC objects and coordinates.

All predicates are computed deterministically from grid coordinates and object geometry.
"""

from __future__ import annotations

import math
from typing import Any

from src.data.models import Grid
from src.objects.segmentation import ObjectInstance
from src.spatial.raycast import RaycastResult, raycast


def collinear(p1: tuple[int, int], p2: tuple[int, int], p3: tuple[int, int]) -> bool:
    """True if three points (r, c) lie on the same line."""
    r1, c1 = p1
    r2, c2 = p2
    r3, c3 = p3
    # Cross product (r2-r1)*(c3-c1) - (c2-c1)*(r3-r1) == 0
    return (r2 - r1) * (c3 - c1) == (c2 - c1) * (r3 - r1)


def diagonal_alignment(p1: tuple[int, int], p2: tuple[int, int]) -> bool:
    """True if two points are aligned diagonally (|dr| == |dc| > 0)."""
    dr = abs(p1[0] - p2[0])
    dc = abs(p1[1] - p2[1])
    return dr == dc and dr > 0


def between(target: tuple[int, int], p1: tuple[int, int], p2: tuple[int, int]) -> bool:
    """True if target point lies strictly on the bounding segment connecting p1 and p2."""
    if not collinear(p1, p2, target):
        return False
    tr, tc = target
    r1, c1 = p1
    r2, c2 = p2
    return (min(r1, r2) <= tr <= max(r1, r2)) and (min(c1, c2) <= tc <= max(c1, c2)) and (target not in (p1, p2))


def inside_object(a: ObjectInstance, b: ObjectInstance) -> bool:
    """True if all pixels of object a lie within bounding box of object b."""
    return (b.min_row <= a.min_row and b.max_row >= a.max_row and
            b.min_col <= a.min_col and b.max_col >= a.max_col)


def outside_object(a: ObjectInstance, b: ObjectInstance) -> bool:
    """True if object a does not overlap bounding box of object b."""
    return (a.max_row < b.min_row or a.min_row > b.max_row or
            a.max_col < b.min_col or a.min_col > b.max_col)


def enclosed_by(a: ObjectInstance, b: ObjectInstance, grid: Grid, background: int = 0) -> bool:
    """True if object a is positioned inside an internal cavity enclosed by object b."""
    if not inside_object(a, b):
        return False
    # Check if a's centroid falls inside b's cavity
    from src.topology.engine import enclosure_detection
    enclosures = enclosure_detection(grid, background=background)
    b_coords = b.pixel_coords
    for enc in enclosures:
        if enc["boundary_coords"].issubset(b_coords):
            if a.pixel_coords.issubset(enc["enclosed_coords"]):
                return True
    return False


def line_pixels_between(p1: tuple[int, int], p2: tuple[int, int]) -> list[tuple[int, int]]:
    """Bresenham's line algorithm returning list of grid coordinates between p1 and p2 inclusive."""
    r1, c1 = p1
    r2, c2 = p2
    pixels = []

    dr = abs(r2 - r1)
    dc = abs(c2 - c1)
    sr = 1 if r1 < r2 else -1
    sc = 1 if c1 < c2 else -1

    err = dr - dc
    curr_r, curr_c = r1, c1

    while True:
        pixels.append((curr_r, curr_c))
        if curr_r == r2 and curr_c == c2:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            curr_r += sr
        if e2 < dr:
            err += dr
            curr_c += sc

    return pixels


def visible_from(p1: tuple[int, int], p2: tuple[int, int], grid: Grid, background: int = 0) -> bool:
    """True if straight line path between p1 and p2 (excluding endpoints) contains only background."""
    line = line_pixels_between(p1, p2)
    if len(line) <= 2:
        return True
    # Check interior segment
    for r, c in line[1:-1]:
        if grid.get(r, c) != background:
            return False
    return True


def obstacle_between(p1: tuple[int, int], p2: tuple[int, int], grid: Grid, background: int = 0) -> bool:
    """True if any non-background cell obstructs straight line between p1 and p2."""
    return not visible_from(p1, p2, grid, background=background)


def ray_intersects(ray_res: RaycastResult, region: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> bool:
    """True if ray path intersects specified region coordinates."""
    path_set = set(ray_res.path)
    return bool(path_set & set(region))
