"""Object-level transformations: symmetry completion, gravity, and object mirroring."""

from __future__ import annotations

from typing import Any

from src.data.models import Grid
from src.transformations.base import Transformation


class SymmetryCompletionTransformation(Transformation):
    """Completes partial symmetric objects by overlaying reflected pixels onto background.
    
    If axis is 'horizontal': mirrors across vertical center (left-right).
    If axis is 'vertical': mirrors across horizontal center (up-down).
    If axis is 'both': completes across both axes.
    """

    def __init__(self, axis: str = "horizontal", background: int = 0) -> None:
        if axis not in ("horizontal", "vertical", "both"):
            raise ValueError(f"Unknown symmetry axis: {axis}")
        self.axis = axis
        self.background = background

    def apply(self, grid: Grid) -> Grid:
        h, w = grid.height, grid.width
        bg = self.background
        cells = [list(row) for row in grid.cells]

        if self.axis in ("horizontal", "both"):
            for r in range(h):
                for c in range(w):
                    c_mirror = w - 1 - c
                    if cells[r][c] == bg and cells[r][c_mirror] != bg:
                        cells[r][c] = cells[r][c_mirror]

        if self.axis in ("vertical", "both"):
            for r in range(h):
                for c in range(w):
                    r_mirror = h - 1 - r
                    if cells[r][c] == bg and cells[r_mirror][c] != bg:
                        cells[r][c] = cells[r_mirror][c]

        return Grid.from_list(cells)

    @property
    def name(self) -> str:
        return f"SymmetryCompletion_{self.axis}"

    @property
    def params(self) -> dict[str, Any]:
        return {"axis": self.axis, "background": self.background}

    @property
    def complexity(self) -> float:
        return 1.6


class GravityTransformation(Transformation):
    """Simulates gravity on non-background pixels in a given direction."""

    def __init__(self, direction: str = "down", background: int = 0) -> None:
        if direction not in ("down", "up", "left", "right"):
            raise ValueError(f"Unknown gravity direction: {direction}")
        self.direction = direction
        self.background = background

    def apply(self, grid: Grid) -> Grid:
        h, w = grid.height, grid.width
        bg = self.background
        cells = [list(row) for row in grid.cells]

        if self.direction == "down":
            for c in range(w):
                # Collect non-background items from top to bottom
                items = [cells[r][c] for r in range(h) if cells[r][c] != bg]
                # Pad top with background
                col_data = [bg] * (h - len(items)) + items
                for r in range(h):
                    cells[r][c] = col_data[r]
        elif self.direction == "up":
            for c in range(w):
                items = [cells[r][c] for r in range(h) if cells[r][c] != bg]
                col_data = items + [bg] * (h - len(items))
                for r in range(h):
                    cells[r][c] = col_data[r]
        elif self.direction == "right":
            for r in range(h):
                items = [cells[r][c] for c in range(w) if cells[r][c] != bg]
                row_data = [bg] * (w - len(items)) + items
                for c in range(w):
                    cells[r][c] = row_data[c]
        elif self.direction == "left":
            for r in range(h):
                items = [cells[r][c] for c in range(w) if cells[r][c] != bg]
                row_data = items + [bg] * (w - len(items))
                for c in range(w):
                    cells[r][c] = row_data[c]

        return Grid.from_list(cells)

    @property
    def name(self) -> str:
        return f"Gravity_{self.direction}"

    @property
    def params(self) -> dict[str, Any]:
        return {"direction": self.direction, "background": self.background}

    @property
    def complexity(self) -> float:
        return 1.7
