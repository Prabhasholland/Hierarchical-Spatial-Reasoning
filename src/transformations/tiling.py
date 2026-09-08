"""Tiling, scaling, and fractal expansion transformations for ARC grids."""

from __future__ import annotations

from typing import Any

from src.data.models import Grid, InvalidGridError
from src.transformations.base import Transformation


class TileTransformation(Transformation):
    """Tiles (repeats) the input grid n_rows by n_cols times.
    
    Mathematical definition:
        G'(r, c) = G(r % H, c % W)
        Output dimensions: (H * n_rows, W * n_cols)
    """

    def __init__(self, n_rows: int = 1, n_cols: int = 1) -> None:
        self.n_rows = max(1, n_rows)
        self.n_cols = max(1, n_cols)

    def apply(self, grid: Grid) -> Grid:
        h, w = grid.height, grid.width
        out_h = h * self.n_rows
        out_w = w * self.n_cols

        if out_h > 30 or out_w > 30:
            raise InvalidGridError(f"Tiled dimensions ({out_h}x{out_w}) exceed max 30x30.")

        tiled_rows = []
        for r in range(out_h):
            src_r = r % h
            row = []
            for _ in range(self.n_cols):
                row.extend(grid.cells[src_r])
            tiled_rows.append(tuple(row))

        return Grid(cells=tuple(tiled_rows))

    @property
    def name(self) -> str:
        return "Tile"

    @property
    def params(self) -> dict[str, Any]:
        return {"n_rows": self.n_rows, "n_cols": self.n_cols}

    @property
    def complexity(self) -> float:
        return 1.4 + 0.1 * (self.n_rows + self.n_cols)


class KroneckerScaleTransformation(Transformation):
    """Scales each cell into a block of size (scale_r, scale_c).
    
    Mathematical definition:
        G'(r, c) = G(r // scale_r, c // scale_c)
        Output dimensions: (H * scale_r, W * scale_c)
    """

    def __init__(self, scale_r: int = 2, scale_c: int = 2) -> None:
        self.scale_r = max(1, scale_r)
        self.scale_c = max(1, scale_c)

    def apply(self, grid: Grid) -> Grid:
        h, w = grid.height, grid.width
        out_h = h * self.scale_r
        out_w = w * self.scale_c

        if out_h > 30 or out_w > 30:
            raise InvalidGridError(f"Scaled dimensions ({out_h}x{out_w}) exceed max 30x30.")

        new_rows = []
        for r in range(h):
            expanded_row = []
            for val in grid.cells[r]:
                expanded_row.extend([val] * self.scale_c)
            row_tuple = tuple(expanded_row)
            for _ in range(self.scale_r):
                new_rows.append(row_tuple)

        return Grid(cells=tuple(new_rows))

    @property
    def name(self) -> str:
        return "KroneckerScale"

    @property
    def params(self) -> dict[str, Any]:
        return {"scale_r": self.scale_r, "scale_c": self.scale_c}

    @property
    def complexity(self) -> float:
        return 1.5 + 0.1 * (self.scale_r + self.scale_c)


class FractalTilingTransformation(Transformation):
    """Fractal self-expansion (Kronecker product of grid with its binary mask).
    
    Replaces each non-background pixel (G(r, c) != bg) with a copy of G (colored with G(r,c)),
    and background pixels with an empty (bg) subgrid of size (H, W).
    Output dimensions: (H * H, W * W)
    """

    def __init__(self, background: int = 0) -> None:
        self.background = background

    def apply(self, grid: Grid) -> Grid:
        h, w = grid.height, grid.width
        out_h = h * h
        out_w = w * w

        if out_h > 30 or out_w > 30:
            raise InvalidGridError(f"Fractal dimensions ({out_h}x{out_w}) exceed max 30x30.")

        bg = self.background
        empty_subgrid = [[bg] * w for _ in range(h)]

        # Precompute subgrids for each non-zero pixel
        full_grid_data = [[bg] * out_w for _ in range(out_h)]

        for r in range(h):
            for c in range(w):
                val = grid.cells[r][c]
                start_r = r * h
                start_c = c * w
                if val != bg:
                    # Place a copy of the grid recolored with val
                    for sub_r in range(h):
                        for sub_c in range(w):
                            if grid.cells[sub_r][sub_c] != bg:
                                full_grid_data[start_r + sub_r][start_c + sub_c] = val
                            else:
                                full_grid_data[start_r + sub_r][start_c + sub_c] = bg

        return Grid.from_list(full_grid_data)

    @property
    def name(self) -> str:
        return "FractalTiling"

    @property
    def params(self) -> dict[str, Any]:
        return {"background": self.background}

    @property
    def complexity(self) -> float:
        return 2.0
