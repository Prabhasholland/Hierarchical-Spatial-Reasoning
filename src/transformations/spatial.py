"""Spatial transformations: translations, cropping, padding, and resizing."""

from __future__ import annotations

from typing import Any

from src.data.models import Grid, InvalidGridError
from src.transformations.base import Transformation


class TranslateTransformation(Transformation):
    """Translate (shift) grid content by (dr, dc).
    
    Mathematical definition:
        If wrap is False: G'(r, c) = G(r - dr, c - dc) if within bounds else fill_color
        If wrap is True:  G'(r, c) = G((r - dr) % H, (c - dc) % W)
    """

    def __init__(self, dr: int, dc: int, fill_color: int = 0, wrap: bool = False) -> None:
        self.dr = dr
        self.dc = dc
        self.fill_color = fill_color
        self.wrap = wrap

    def apply(self, grid: Grid) -> Grid:
        h, w = grid.height, grid.width
        dr, dc = self.dr, self.dc
        fill = self.fill_color

        new_rows = []
        for r in range(h):
            new_row = []
            for c in range(w):
                src_r = r - dr
                src_c = c - dc
                if self.wrap:
                    src_r %= h
                    src_c %= w
                    new_row.append(grid.cells[src_r][src_c])
                else:
                    if 0 <= src_r < h and 0 <= src_c < w:
                        new_row.append(grid.cells[src_r][src_c])
                    else:
                        new_row.append(fill)
            new_rows.append(tuple(new_row))

        return Grid(cells=tuple(new_rows))

    @property
    def name(self) -> str:
        return "Translate"

    @property
    def params(self) -> dict[str, Any]:
        return {"dr": self.dr, "dc": self.dc, "fill_color": self.fill_color, "wrap": self.wrap}

    @property
    def complexity(self) -> float:
        return 1.4


class CropBoundingBoxTransformation(Transformation):
    """Crops the grid to the minimal bounding box enclosing all non-background pixels."""

    def __init__(self, background: int = 0, pad: int = 0) -> None:
        self.background = background
        self.pad = pad

    def apply(self, grid: Grid) -> Grid:
        bg = self.background
        coords = [
            (r, c)
            for r in range(grid.height)
            for c in range(grid.width)
            if grid.cells[r][c] != bg
        ]

        if not coords:
            # If entirely background, return a 1x1 background grid
            return Grid(cells=((bg,),))

        min_r = max(0, min(r for r, _ in coords) - self.pad)
        max_r = min(grid.height - 1, max(r for r, _ in coords) + self.pad)
        min_c = max(0, min(c for _, c in coords) - self.pad)
        max_c = min(grid.width - 1, max(c for _, c in coords) + self.pad)

        cropped_cells = tuple(
            tuple(grid.cells[r][c] for c in range(min_c, max_c + 1))
            for r in range(min_r, max_r + 1)
        )
        return Grid(cells=cropped_cells)

    @property
    def name(self) -> str:
        return "CropBoundingBox"

    @property
    def params(self) -> dict[str, Any]:
        return {"background": self.background, "pad": self.pad}

    @property
    def complexity(self) -> float:
        return 1.5


class CropColorObjectTransformation(Transformation):
    """Crops to the bounding box of cells containing a specific color."""

    def __init__(self, target_color: int) -> None:
        self.target_color = target_color

    def apply(self, grid: Grid) -> Grid:
        target = self.target_color
        coords = [
            (r, c)
            for r in range(grid.height)
            for c in range(grid.width)
            if grid.cells[r][c] == target
        ]

        if not coords:
            return Grid(cells=((0,),))

        min_r = min(r for r, _ in coords)
        max_r = max(r for r, _ in coords)
        min_c = min(c for _, c in coords)
        max_c = max(c for _, c in coords)

        cropped_cells = tuple(
            tuple(grid.cells[r][c] for c in range(min_c, max_c + 1))
            for r in range(min_r, max_r + 1)
        )
        return Grid(cells=cropped_cells)

    @property
    def name(self) -> str:
        return "CropColorObject"

    @property
    def params(self) -> dict[str, Any]:
        return {"target_color": self.target_color}

    @property
    def complexity(self) -> float:
        return 1.6


class CropFixedTransformation(Transformation):
    """Crops to fixed row/column bounds (r_start:r_end, c_start:c_end)."""

    def __init__(self, r_start: int, r_end: int, c_start: int, c_end: int) -> None:
        self.r_start = r_start
        self.r_end = r_end
        self.c_start = c_start
        self.c_end = c_end

    def apply(self, grid: Grid) -> Grid:
        h, w = grid.height, grid.width
        r0 = max(0, min(h - 1, self.r_start))
        r1 = max(r0 + 1, min(h, self.r_end))
        c0 = max(0, min(w - 1, self.c_start))
        c1 = max(c0 + 1, min(w, self.c_end))

        cropped = tuple(
            tuple(grid.cells[r][c] for c in range(c0, c1))
            for r in range(r0, r1)
        )
        return Grid(cells=cropped)

    @property
    def name(self) -> str:
        return "CropFixed"

    @property
    def params(self) -> dict[str, Any]:
        return {
            "r_start": self.r_start,
            "r_end": self.r_end,
            "c_start": self.c_start,
            "c_end": self.c_end,
        }

    @property
    def complexity(self) -> float:
        return 1.8


class PadTransformation(Transformation):
    """Pads grid borders with fill_color."""

    def __init__(
        self, top: int = 0, bottom: int = 0, left: int = 0, right: int = 0, fill_color: int = 0
    ) -> None:
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        self.fill_color = fill_color

    def apply(self, grid: Grid) -> Grid:
        new_h = grid.height + self.top + self.bottom
        new_w = grid.width + self.left + self.right

        if new_h > 30 or new_w > 30:
            raise InvalidGridError(f"Padded grid dimensions ({new_h}x{new_w}) exceed max 30x30.")

        fill = self.fill_color
        rows = []
        for _ in range(self.top):
            rows.append(tuple(fill for _ in range(new_w)))
        for r in range(grid.height):
            middle = tuple(grid.cells[r])
            left_pad = tuple(fill for _ in range(self.left))
            right_pad = tuple(fill for _ in range(self.right))
            rows.append(left_pad + middle + right_pad)
        for _ in range(self.bottom):
            rows.append(tuple(fill for _ in range(new_w)))

        return Grid(cells=tuple(rows))

    @property
    def name(self) -> str:
        return "Pad"

    @property
    def params(self) -> dict[str, Any]:
        return {
            "top": self.top,
            "bottom": self.bottom,
            "left": self.left,
            "right": self.right,
            "fill_color": self.fill_color,
        }

    @property
    def complexity(self) -> float:
        return 1.7
