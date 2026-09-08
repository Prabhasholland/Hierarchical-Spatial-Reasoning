"""Unit tests for raycasting engine."""

from __future__ import annotations

from src.data.models import Grid
from src.spatial.raycast import raycast


def test_raycast_boundary_stop():
    grid = Grid.from_list([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ])
    res = raycast(grid, start=(1, 1), direction="right", stop_condition="boundary")
    assert res.stop_reason == "boundary"
    assert res.path == [(1, 1), (1, 2)]
    assert res.steps == 1


def test_raycast_non_background_hit():
    grid = Grid.from_list([
        [0, 0, 0, 0],
        [0, 0, 5, 0],
        [0, 0, 0, 0],
    ])
    res = raycast(grid, start=(1, 0), direction="right", stop_condition="non_background", background=0)
    assert res.stop_reason == "non_background"
    assert res.hit_coord == (1, 2)
    assert res.hit_color == 5
    assert res.path == [(1, 0), (1, 1), (1, 2)]


def test_raycast_diagonal_direction():
    grid = Grid.from_list([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 3],
    ])
    res = raycast(grid, start=(0, 0), direction="down_right", stop_condition="non_background", background=0)
    assert res.hit_coord == (2, 2)
    assert res.hit_color == 3
