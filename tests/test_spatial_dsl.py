"""Unit tests for Spatial DSL operators."""

from __future__ import annotations

from src.data.models import Grid
from src.spatial.dsl import (
    connect_same_color_op,
    draw_line_op,
    fill_enclosed_op,
    flood_fill_op,
    raycast_op,
    trace_boundary_op,
)
from src.spatial.relations import collinear, diagonal_alignment, visible_from


def test_spatial_dsl_operators():
    grid = Grid.from_list([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ])
    filled = fill_enclosed_op(grid, fill_color=3, background=0)
    assert filled.get(1, 1) == 3

    line_grid = draw_line_op(Grid.from_list([[0]*3]*3), (0, 0), (2, 2), color=4)
    assert line_grid.get(0, 0) == 4
    assert line_grid.get(1, 1) == 4
    assert line_grid.get(2, 2) == 4

    traced = trace_boundary_op(grid, boundary_color=5, background=0)
    assert traced.get(0, 0) == 5


def test_spatial_predicates():
    assert collinear((0, 0), (1, 1), (2, 2)) is True
    assert collinear((0, 0), (0, 1), (1, 1)) is False
    assert diagonal_alignment((0, 0), (2, 2)) is True

    grid = Grid.from_list([
        [1, 0, 1],
        [0, 2, 0],
        [0, 0, 0],
    ])
    assert visible_from((0, 0), (0, 2), grid, background=0) is True
    assert visible_from((0, 0), (2, 2), grid, background=0) is False  # 2 obstructs (1, 1)
