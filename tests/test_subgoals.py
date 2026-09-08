"""Unit tests for SubGoal generation."""

from __future__ import annotations

from src.data.models import Grid
from src.reasoning.state import GridState
from src.reasoning.subgoals import propose_subgoals


def test_propose_subgoals():
    g1 = Grid.from_list([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ])
    g2 = Grid.from_list([
        [1, 1, 1],
        [1, 3, 1],
        [1, 1, 1],
    ])

    st1 = GridState.from_grid(g1)
    st2 = GridState.from_grid(g2)

    subgoals = propose_subgoals(st1, st2)
    assert len(subgoals) >= 2
    types = [sg.goal_type for sg in subgoals]
    assert "fill_enclosed_region" in types or "change_color" in types
