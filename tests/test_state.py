"""Unit tests for GridState state representation."""

from __future__ import annotations

from src.data.models import Grid
from src.reasoning.state import GridState


def test_grid_state_computation():
    grid = Grid.from_list([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ])
    st = GridState.from_grid(grid)
    assert st.height == 3
    assert st.width == 3
    assert st.num_colors == 2
    assert st.foreground_cell_count == 8
    assert st.num_enclosed_regions == 1
    assert st.horizontal_symmetry == 1.0
    assert st.vertical_symmetry == 1.0

    d = st.to_dict()
    assert "color_entropy" in d
    assert "num_holes" in d
