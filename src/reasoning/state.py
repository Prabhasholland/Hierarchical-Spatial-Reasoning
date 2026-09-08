"""Measurable GridState representation for ARC grids.

Computes a deterministic feature vector characterizing structural, topological,
colorimetric, and spatial properties of an ARC Grid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.data.models import Grid
from src.objects.segmentation import detect_background, segment_grid
from src.topology.engine import enclosure_detection, hole_detection


@dataclass(frozen=True)
class GridState:
    """Immutable, serializable state representation of a 2D ARC grid."""
    height: int
    width: int
    num_colors: int
    color_histogram: dict[int, int]
    color_entropy: float
    foreground_cell_count: int
    connected_component_count: int
    object_count: int
    min_object_size: int
    max_object_size: int
    mean_object_size: float
    min_bbox_area: int
    max_bbox_area: int
    horizontal_symmetry: float
    vertical_symmetry: float
    diagonal_symmetry_main: float
    diagonal_symmetry_anti: float
    spatial_occupancy: float
    num_enclosed_regions: int
    num_holes: int
    num_isolated_pixels: int
    background_color: int

    @classmethod
    def from_grid(cls, grid: Grid, background: int | None = None) -> GridState:
        """Compute GridState from an ARC Grid instance."""
        h, w = grid.height, grid.width
        if background is None:
            background = detect_background(grid)

        color_counts = grid.color_counts()
        num_colors = len(color_counts)

        # Color entropy
        total_pixels = h * w
        entropy = 0.0
        for cnt in color_counts.values():
            p = cnt / total_pixels
            if p > 0:
                entropy -= p * math.log2(p)

        fg_count = sum(cnt for c, cnt in color_counts.items() if c != background)
        occupancy = fg_count / total_pixels if total_pixels > 0 else 0.0

        # Segment objects
        hyp = segment_grid(grid, background=background)
        objs = hyp.objects
        obj_count = len(objs)
        comp_count = obj_count

        if objs:
            sizes = [o.area for o in objs]
            min_obj_sz = min(sizes)
            max_obj_sz = max(sizes)
            mean_obj_sz = float(sum(sizes) / len(sizes))

            bb_areas = [o.height * o.width for o in objs]
            min_bb = min(bb_areas)
            max_bb = max(bb_areas)
        else:
            min_obj_sz = max_obj_sz = 0
            mean_obj_sz = 0.0
            min_bb = max_bb = 0

        # Symmetry scores
        arr = grid.to_numpy()
        h_sym = float(np.mean(arr == np.fliplr(arr)))
        v_sym = float(np.mean(arr == np.flipud(arr)))

        if h == w:
            d_main_sym = float(np.mean(arr == arr.T))
            d_anti_sym = float(np.mean(arr == np.rot90(arr.T)))
        else:
            d_main_sym = 0.0
            d_anti_sym = 0.0

        # Topology
        enclosures = enclosure_detection(grid, background=background)
        num_enclosed = len(enclosures)

        holes = hole_detection(grid, background=background)
        num_holes = len(holes)

        # Isolated pixels
        isolated_count = 0
        for obj in objs:
            if obj.area == 1:
                isolated_count += 1

        return cls(
            height=h,
            width=w,
            num_colors=num_colors,
            color_histogram=dict(sorted(color_counts.items())),
            color_entropy=round(entropy, 4),
            foreground_cell_count=fg_count,
            connected_component_count=comp_count,
            object_count=obj_count,
            min_object_size=min_obj_sz,
            max_object_size=max_obj_sz,
            mean_object_size=round(mean_obj_sz, 2),
            min_bbox_area=min_bb,
            max_bbox_area=max_bb,
            horizontal_symmetry=round(h_sym, 4),
            vertical_symmetry=round(v_sym, 4),
            diagonal_symmetry_main=round(d_main_sym, 4),
            diagonal_symmetry_anti=round(d_anti_sym, 4),
            spatial_occupancy=round(occupancy, 4),
            num_enclosed_regions=num_enclosed,
            num_holes=num_holes,
            num_isolated_pixels=isolated_count,
            background_color=background,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise state properties to a plain dictionary."""
        return {
            "height": self.height,
            "width": self.width,
            "num_colors": self.num_colors,
            "color_histogram": self.color_histogram,
            "color_entropy": self.color_entropy,
            "foreground_cell_count": self.foreground_cell_count,
            "connected_component_count": self.connected_component_count,
            "object_count": self.object_count,
            "min_object_size": self.min_object_size,
            "max_object_size": self.max_object_size,
            "mean_object_size": self.mean_object_size,
            "min_bbox_area": self.min_bbox_area,
            "max_bbox_area": self.max_bbox_area,
            "horizontal_symmetry": self.horizontal_symmetry,
            "vertical_symmetry": self.vertical_symmetry,
            "diagonal_symmetry_main": self.diagonal_symmetry_main,
            "diagonal_symmetry_anti": self.diagonal_symmetry_anti,
            "spatial_occupancy": self.spatial_occupancy,
            "num_enclosed_regions": self.num_enclosed_regions,
            "num_holes": self.num_holes,
            "num_isolated_pixels": self.num_isolated_pixels,
            "background_color": self.background_color,
        }
