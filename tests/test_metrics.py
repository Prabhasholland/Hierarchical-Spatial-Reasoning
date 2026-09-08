"""Tests for evaluation metrics."""

from src.evaluation.metrics import grid_match


def test_grid_match_identical():
    grid = [[1, 2], [3, 4]]
    assert grid_match(grid, grid) is True


def test_grid_match_different_values():
    grid_a = [[1, 2], [3, 4]]
    grid_b = [[1, 2], [3, 5]]
    assert grid_match(grid_a, grid_b) is False


def test_grid_match_different_dimensions():
    grid_a = [[1, 2], [3, 4]]
    grid_b = [[1, 2, 3], [4, 5, 6]]
    assert grid_match(grid_a, grid_b) is False


def test_grid_match_different_row_count():
    grid_a = [[1, 2]]
    grid_b = [[1, 2], [3, 4]]
    assert grid_match(grid_a, grid_b) is False
