"""Geometric transformations for ARC grids.

Includes rigid transformations, reflections, rotations, and transpositions.
All transformations are deterministic and preserve rectangularity.
"""

from __future__ import annotations

from typing import Any

from src.data.models import Grid
from src.transformations.base import Transformation


class IdentityTransformation(Transformation):
    """Identity transformation: returns the input grid unchanged.
    
    Mathematical definition: f(G) = G
    """

    def apply(self, grid: Grid) -> Grid:
        return grid

    @property
    def name(self) -> str:
        return "Identity"

    @property
    def complexity(self) -> float:
        return 0.1


class Rotate90Transformation(Transformation):
    """90 degrees clockwise rotation.
    
    Mathematical definition: G'(c, H - 1 - r) = G(r, c)
    Output dimensions: (W, H)
    """

    def apply(self, grid: Grid) -> Grid:
        h, w = grid.height, grid.width
        new_cells = tuple(
            tuple(grid.cells[h - 1 - r][c] for r in range(h))
            for c in range(w)
        )
        return Grid(cells=new_cells)

    @property
    def name(self) -> str:
        return "Rotate90"

    @property
    def complexity(self) -> float:
        return 1.0


class Rotate180Transformation(Transformation):
    """180 degrees clockwise rotation.
    
    Mathematical definition: G'(H - 1 - r, W - 1 - c) = G(r, c)
    Output dimensions: (H, W)
    """

    def apply(self, grid: Grid) -> Grid:
        h, w = grid.height, grid.width
        new_cells = tuple(
            tuple(grid.cells[h - 1 - r][w - 1 - c] for c in range(w))
            for r in range(h)
        )
        return Grid(cells=new_cells)

    @property
    def name(self) -> str:
        return "Rotate180"

    @property
    def complexity(self) -> float:
        return 1.0


class Rotate270Transformation(Transformation):
    """270 degrees clockwise (90 degrees counter-clockwise) rotation.
    
    Mathematical definition: G'(W - 1 - c, r) = G(r, c)
    Output dimensions: (W, H)
    """

    def apply(self, grid: Grid) -> Grid:
        h, w = grid.height, grid.width
        new_cells = tuple(
            tuple(grid.cells[r][w - 1 - c] for r in range(h))
            for c in range(w)
        )
        return Grid(cells=new_cells)

    @property
    def name(self) -> str:
        return "Rotate270"

    @property
    def complexity(self) -> float:
        return 1.0


class HorizontalFlipTransformation(Transformation):
    """Horizontal reflection (mirror across vertical axis, left-right flip).
    
    Mathematical definition: G'(r, W - 1 - c) = G(r, c)
    Output dimensions: (H, W)
    """

    def apply(self, grid: Grid) -> Grid:
        h, w = grid.height, grid.width
        new_cells = tuple(
            tuple(grid.cells[r][w - 1 - c] for c in range(w))
            for r in range(h)
        )
        return Grid(cells=new_cells)

    @property
    def name(self) -> str:
        return "HorizontalFlip"

    @property
    def complexity(self) -> float:
        return 1.0


class VerticalFlipTransformation(Transformation):
    """Vertical reflection (mirror across horizontal axis, up-down flip).
    
    Mathematical definition: G'(H - 1 - r, c) = G(r, c)
    Output dimensions: (H, W)
    """

    def apply(self, grid: Grid) -> Grid:
        h, w = grid.height, grid.width
        new_cells = tuple(
            tuple(grid.cells[h - 1 - r][c] for c in range(w))
            for r in range(h)
        )
        return Grid(cells=new_cells)

    @property
    def name(self) -> str:
        return "VerticalFlip"

    @property
    def complexity(self) -> float:
        return 1.0


class TransposeTransformation(Transformation):
    """Matrix transposition (reflection across main diagonal).
    
    Mathematical definition: G'(c, r) = G(r, c)
    Output dimensions: (W, H)
    """

    def apply(self, grid: Grid) -> Grid:
        h, w = grid.height, grid.width
        new_cells = tuple(
            tuple(grid.cells[r][c] for r in range(h))
            for c in range(w)
        )
        return Grid(cells=new_cells)

    @property
    def name(self) -> str:
        return "Transpose"

    @property
    def complexity(self) -> float:
        return 1.1


class AntiTransposeTransformation(Transformation):
    """Reflection across anti-diagonal.
    
    Mathematical definition: G'(W - 1 - c, H - 1 - r) = G(r, c)
    Output dimensions: (W, H)
    """

    def apply(self, grid: Grid) -> Grid:
        h, w = grid.height, grid.width
        new_cells = tuple(
            tuple(grid.cells[h - 1 - r][w - 1 - c] for r in range(h))
            for c in range(w)
        )
        return Grid(cells=new_cells)

    @property
    def name(self) -> str:
        return "AntiTranspose"

    @property
    def complexity(self) -> float:
        return 1.1


ALL_GEOMETRIC_TRANSFORMATIONS: list[Transformation] = [
    IdentityTransformation(),
    Rotate90Transformation(),
    Rotate180Transformation(),
    Rotate270Transformation(),
    HorizontalFlipTransformation(),
    VerticalFlipTransformation(),
    TransposeTransformation(),
    AntiTransposeTransformation(),
]
