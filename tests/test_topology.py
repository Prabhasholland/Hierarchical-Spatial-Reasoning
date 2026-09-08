"""Unit tests for topology engine."""

from __future__ import annotations

from src.data.models import Grid
from src.topology.engine import (
    boundary_trace,
    connected_regions,
    enclosure_detection,
    flood_fill,
    hole_detection,
    region_adjacency,
)


def test_flood_fill():
    grid = Grid.from_list([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ])
    filled = flood_fill(grid, start=(0, 0), fill_color=3, connectivity=4)
    assert filled.get(0, 0) == 3
    assert filled.get(0, 1) == 3
    assert filled.get(1, 1) == 1  # 1 is unchanged


def test_enclosure_detection_and_connected_regions():
    # Frame 1 encloses cavity of 0
    grid = Grid.from_list([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ])
    enclosures = enclosure_detection(grid, background=0)
    assert len(enclosures) == 1
    assert enclosures[0]["enclosed_coords"] == {(1, 1)}
    assert enclosures[0]["boundary_colors"] == {1}

    regions = connected_regions(grid, background=0)
    assert len(regions["enclosed_background"]) == 1
    assert len(regions["foreground_regions"]) == 1


def test_boundary_trace():
    grid = Grid.from_list([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ])
    region = {(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)}
    b_pixels = boundary_trace(grid, region)
    assert (1, 1) not in b_pixels  # Center pixel is not boundary
    assert (0, 0) in b_pixels


def test_hole_detection_and_region_adjacency():
    grid = Grid.from_list([
        [1, 1, 1, 0, 2],
        [1, 0, 1, 0, 2],
        [1, 1, 1, 0, 2],
    ])
    holes = hole_detection(grid, background=0)
    assert len(holes) == 1
    assert holes[0] == {(1, 1)}

    regions = [{(0, 0)}, {(0, 1)}, {(0, 2)}]
    adj = region_adjacency(grid, regions)
    assert 1 in adj[0]
    assert 0 in adj[1] and 2 in adj[1]
