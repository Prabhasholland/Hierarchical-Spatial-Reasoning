"""Unit tests for StateDifference analyzer."""

from __future__ import annotations

from src.data.models import Grid
from src.reasoning.state import GridState
from src.reasoning.state_difference import StateDifference


def test_state_difference_analysis():
    g1 = Grid.from_list([
        [1, 1, 0],
        [1, 1, 0],
    ])
    g2 = Grid.from_list([
        [1, 1],
        [1, 1],
    ])

    st1 = GridState.from_grid(g1)
    st2 = GridState.from_grid(g2)

    diff = StateDifference.compute(st1, st2)
    assert diff.dim_change == (0, -1)
    assert diff.is_size_preserving is False
    assert diff.is_object_extraction is True
