"""Topology engine for ARC grids.

Implements deterministic topological primitives:
  - flood_fill
  - connected_regions (enclosed vs. exterior background)
  - enclosure_detection
  - boundary_trace
  - hole_detection
  - region_adjacency
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from src.data.models import Grid


def flood_fill(
    grid: Grid,
    start: tuple[int, int],
    fill_color: int,
    target_color: int | None = None,
    connectivity: int = 4,
) -> Grid:
    """Flood-fill a connected region of cells with fill_color.
    
    Args:
        grid: Input Grid.
        start: (row, col) seed point.
        fill_color: Color to paint (0-9).
        target_color: Color to replace (if None, reads color at start).
        connectivity: 4 for cardinal neighbors, 8 for cardinal + diagonal.
        
    Returns:
        New Grid instance with flood fill applied.
    """
    sr, sc = start
    if not (0 <= sr < grid.height and 0 <= sc < grid.width):
        return grid

    src_color = grid.get(sr, sc) if target_color is None else target_color
    if src_color == fill_color:
        return grid

    cells = grid.to_list()
    h, w = grid.height, grid.width

    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        offsets.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])

    queue = deque([(sr, sc)])
    visited = {(sr, sc)}
    cells[sr][sc] = fill_color

    while queue:
        r, c = queue.popleft()
        for dr, dc in offsets:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                if (nr, nc) not in visited and cells[nr][nc] == src_color:
                    visited.add((nr, nc))
                    cells[nr][nc] = fill_color
                    queue.append((nr, nc))

    return Grid.from_list(cells)


def boundary_trace(
    grid: Grid,
    region_coords: set[tuple[int, int]] | frozenset[tuple[int, int]],
    connectivity: int = 4,
) -> set[tuple[int, int]]:
    """Extract boundary pixels for a set of region coordinates.
    
    A pixel is on the boundary if it is in region_coords and has at least
    one neighbor (under given connectivity) outside region_coords or outside grid bounds.
    """
    boundary = set()
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        offsets.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])

    h, w = grid.height, grid.width

    for r, c in region_coords:
        is_boundary = False
        for dr, dc in offsets:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or (nr, nc) not in region_coords:
                is_boundary = True
                break
        if is_boundary:
            boundary.add((r, c))

    return boundary


def connected_regions(
    grid: Grid,
    background: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    """Identify background regions (exterior vs enclosed) and foreground regions."""
    h, w = grid.height, grid.width
    visited: set[tuple[int, int]] = set()

    exterior_bg_coords: set[tuple[int, int]] = set()
    enclosed_bg_regions: list[set[tuple[int, int]]] = []
    foreground_regions: list[set[tuple[int, int]]] = []

    # 1. First pass: find exterior background touching outer border via BFS
    queue: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if grid.get(r, c) == background and (r, c) not in exterior_bg_coords:
                queue.append((r, c))
                exterior_bg_coords.add((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if grid.get(r, c) == background and (r, c) not in exterior_bg_coords:
                queue.append((r, c))
                exterior_bg_coords.add((r, c))

    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                if grid.get(nr, nc) == background and (nr, nc) not in exterior_bg_coords:
                    exterior_bg_coords.add((nr, nc))
                    queue.append((nr, nc))

    visited.update(exterior_bg_coords)

    # 2. Second pass: remaining background components are enclosed cavities
    for r in range(h):
        for c in range(w):
            if (r, c) in visited:
                continue
            if grid.get(r, c) == background:
                comp: set[tuple[int, int]] = set()
                q = deque([(r, c)])
                visited.add((r, c))
                while q:
                    cr, cc = q.popleft()
                    comp.add((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            if (nr, nc) not in visited and grid.get(nr, nc) == background:
                                visited.add((nr, nc))
                                q.append((nr, nc))
                enclosed_bg_regions.append(comp)
            else:
                # Foreground component
                comp = set()
                q = deque([(r, c)])
                visited.add((r, c))
                while q:
                    cr, cc = q.popleft()
                    comp.add((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            if (nr, nc) not in visited and grid.get(nr, nc) != background:
                                visited.add((nr, nc))
                                q.append((nr, nc))
                foreground_regions.append(comp)

    return {
        "exterior_background": [{"coords": list(exterior_bg_coords), "size": len(exterior_bg_coords)}],
        "enclosed_background": [{"coords": list(c), "size": len(c)} for c in enclosed_bg_regions],
        "foreground_regions": [{"coords": list(c), "size": len(c)} for c in foreground_regions],
    }


def enclosure_detection(
    grid: Grid,
    background: int = 0,
) -> list[dict[str, Any]]:
    """Detect enclosed background cavities and their enclosing boundaries."""
    regions_info = connected_regions(grid, background=background)
    enclosed_bg = regions_info["enclosed_background"]

    enclosures: list[dict[str, Any]] = []
    h, w = grid.height, grid.width

    for enc_info in enclosed_bg:
        enc_coords = set(tuple(p) for p in enc_info["coords"])
        # Find 4-connected neighboring foreground pixels that enclose this cavity
        boundary_coords: set[tuple[int, int]] = set()
        boundary_colors: set[int] = set()

        for r, c in enc_coords:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    if (nr, nc) not in enc_coords and grid.get(nr, nc) != background:
                        boundary_coords.add((nr, nc))
                        boundary_colors.add(grid.get(nr, nc))

        enclosures.append({
            "enclosed_coords": enc_coords,
            "region_size": len(enc_coords),
            "boundary_coords": boundary_coords,
            "boundary_colors": boundary_colors,
            "primary_boundary_color": sorted(list(boundary_colors))[0] if boundary_colors else None,
        })

    return enclosures


def hole_detection(
    grid: Grid,
    target_coords: set[tuple[int, int]] | None = None,
    background: int = 0,
) -> list[set[tuple[int, int]]]:
    """Detect internal cavities/holes inside a target object or across all objects."""
    enclosures = enclosure_detection(grid, background=background)
    if target_coords is None:
        return [enc["enclosed_coords"] for enc in enclosures]

    # Filter enclosures that are completely bounded by target_coords
    object_holes = []
    for enc in enclosures:
        b_coords = enc["boundary_coords"]
        if b_coords.issubset(target_coords):
            object_holes.append(enc["enclosed_coords"])

    return object_holes


def region_adjacency(
    grid: Grid,
    regions: list[set[tuple[int, int]]],
    connectivity: int = 4,
) -> dict[int, list[int]]:
    """Build an adjacency dictionary indicating which region indices touch each other."""
    adj: dict[int, list[int]] = {i: [] for i in range(len(regions))}
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        offsets.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])

    # Map coord to region_idx
    coord_map: dict[tuple[int, int], int] = {}
    for idx, reg in enumerate(regions):
        for r, c in reg:
            coord_map[(r, c)] = idx

    for idx, reg in enumerate(regions):
        neighbors = set()
        for r, c in reg:
            for dr, dc in offsets:
                nr, nc = r + dr, c + dc
                if (nr, nc) in coord_map and coord_map[(nr, nc)] != idx:
                    neighbors.add(coord_map[(nr, nc)])
        adj[idx] = sorted(list(neighbors))

    return adj
