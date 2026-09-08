"""Color transformation operations for ARC grids.

Supports arbitrary color substitution maps, pairwise color swaps, background
replacement, color filtering, and foreground uniformization.
"""

from __future__ import annotations

from typing import Any, Mapping

from src.data.models import Grid
from src.transformations.base import Transformation


class ColorSubstitutionTransformation(Transformation):
    """Maps colors according to a discrete mapping dictionary.
    
    Mathematical definition: G'(r, c) = M.get(G(r, c), G(r, c))
    """

    def __init__(self, mapping: Mapping[int, int]) -> None:
        # Filter out identity mappings (c -> c)
        self._mapping = {k: v for k, v in mapping.items() if k != v}

    @property
    def mapping(self) -> dict[int, int]:
        return dict(self._mapping)

    def apply(self, grid: Grid) -> Grid:
        if not self._mapping:
            return grid
        new_cells = tuple(
            tuple(self._mapping.get(val, val) for val in row)
            for row in grid.cells
        )
        return Grid(cells=new_cells)

    @property
    def name(self) -> str:
        return "ColorSubstitution"

    @property
    def params(self) -> dict[str, Any]:
        return {"mapping": self._mapping}

    @property
    def complexity(self) -> float:
        # Cost scales with number of substituted colors
        return 1.0 + 0.2 * len(self._mapping)


class ColorSwapTransformation(Transformation):
    """Swaps two specific colors across the grid.
    
    Mathematical definition:
        G'(r, c) = c2 if G(r, c) == c1 else (c1 if G(r, c) == c2 else G(r, c))
    """

    def __init__(self, color1: int, color2: int) -> None:
        self.color1 = color1
        self.color2 = color2

    def apply(self, grid: Grid) -> Grid:
        if self.color1 == self.color2:
            return grid
        c1, c2 = self.color1, self.color2
        new_cells = tuple(
            tuple(c2 if val == c1 else (c1 if val == c2 else val) for val in row)
            for row in grid.cells
        )
        return Grid(cells=new_cells)

    @property
    def name(self) -> str:
        return "ColorSwap"

    @property
    def params(self) -> dict[str, Any]:
        return {"color1": self.color1, "color2": self.color2}

    @property
    def complexity(self) -> float:
        return 1.2


class ColorReplaceTransformation(Transformation):
    """Replaces all occurrences of a single source color with a target color."""

    def __init__(self, old_color: int, new_color: int) -> None:
        self.old_color = old_color
        self.new_color = new_color

    def apply(self, grid: Grid) -> Grid:
        if self.old_color == self.new_color:
            return grid
        new_cells = tuple(
            tuple(self.new_color if val == self.old_color else val for val in row)
            for row in grid.cells
        )
        return Grid(cells=new_cells)

    @property
    def name(self) -> str:
        return "ColorReplace"

    @property
    def params(self) -> dict[str, Any]:
        return {"old_color": self.old_color, "new_color": self.new_color}

    @property
    def complexity(self) -> float:
        return 1.1


class InvertForegroundTransformation(Transformation):
    """Replaces all non-background pixels with a single target color.
    
    Mathematical definition:
        G'(r, c) = target_color if G(r, c) != background else background
    """

    def __init__(self, target_color: int, background: int = 0) -> None:
        self.target_color = target_color
        self.background = background

    def apply(self, grid: Grid) -> Grid:
        bg = self.background
        target = self.target_color
        new_cells = tuple(
            tuple(bg if val == bg else target for val in row)
            for row in grid.cells
        )
        return Grid(cells=new_cells)

    @property
    def name(self) -> str:
        return "InvertForeground"

    @property
    def params(self) -> dict[str, Any]:
        return {"target_color": self.target_color, "background": self.background}

    @property
    def complexity(self) -> float:
        return 1.3
