"""State difference analyzer for ARC grids.

Computes a structured difference between source GridState and target GridState,
extracting planning signals for sub-goal decomposition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.data.models import Grid
from src.reasoning.state import GridState


@dataclass(frozen=True)
class StateDifference:
    """Structured delta representation between source and target GridState."""
    dim_change: tuple[int, int]
    object_count_delta: int
    color_count_delta: int
    foreground_cell_delta: int
    enclosed_region_delta: int
    hole_delta: int
    symmetry_delta_h: float
    symmetry_delta_v: float
    entropy_delta: float
    is_size_preserving: bool
    is_object_extraction: bool
    is_object_filtering: bool
    is_cavity_filling: bool
    is_recoloring: bool
    removed_colors: set[int]
    added_colors: set[int]

    @classmethod
    def compute(cls, src_state: GridState, dst_state: GridState) -> StateDifference:
        dim_chg = (dst_state.height - src_state.height, dst_state.width - src_state.width)
        obj_delta = dst_state.object_count - src_state.object_count
        color_delta = dst_state.num_colors - src_state.num_colors
        fg_delta = dst_state.foreground_cell_count - src_state.foreground_cell_count
        enc_delta = dst_state.num_enclosed_regions - src_state.num_enclosed_regions
        hole_delta = dst_state.num_holes - src_state.num_holes

        sym_delta_h = round(dst_state.horizontal_symmetry - src_state.horizontal_symmetry, 4)
        sym_delta_v = round(dst_state.vertical_symmetry - src_state.vertical_symmetry, 4)
        ent_delta = round(dst_state.color_entropy - src_state.color_entropy, 4)

        is_size_pres = (dim_chg == (0, 0))

        # Heuristic difference flags
        is_obj_extract = not is_size_pres and (dst_state.height <= src_state.height) and (dst_state.width <= src_state.width)
        is_obj_filter = is_size_pres and (dst_state.object_count < src_state.object_count)
        is_cavity_fill = (src_state.num_enclosed_regions > 0 or src_state.num_holes > 0) and (dst_state.num_enclosed_regions < src_state.num_enclosed_regions or dst_state.num_holes < src_state.num_holes)

        src_colors = set(src_state.color_histogram.keys())
        dst_colors = set(dst_state.color_histogram.keys())

        removed = src_colors - dst_colors
        added = dst_colors - src_colors
        is_recolor = bool(removed or added)

        return cls(
            dim_change=dim_chg,
            object_count_delta=obj_delta,
            color_count_delta=color_delta,
            foreground_cell_delta=fg_delta,
            enclosed_region_delta=enc_delta,
            hole_delta=hole_delta,
            symmetry_delta_h=sym_delta_h,
            symmetry_delta_v=sym_delta_v,
            entropy_delta=ent_delta,
            is_size_preserving=is_size_pres,
            is_object_extraction=is_obj_extract,
            is_object_filtering=is_obj_filter,
            is_cavity_filling=is_cavity_fill,
            is_recoloring=is_recolor,
            removed_colors=removed,
            added_colors=added,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dim_change": list(self.dim_change),
            "object_count_delta": self.object_count_delta,
            "color_count_delta": self.color_count_delta,
            "foreground_cell_delta": self.foreground_cell_delta,
            "enclosed_region_delta": self.enclosed_region_delta,
            "hole_delta": self.hole_delta,
            "symmetry_delta_h": self.symmetry_delta_h,
            "symmetry_delta_v": self.symmetry_delta_v,
            "entropy_delta": self.entropy_delta,
            "is_size_preserving": self.is_size_preserving,
            "is_object_extraction": self.is_object_extraction,
            "is_object_filtering": self.is_object_filtering,
            "is_cavity_filling": self.is_cavity_filling,
            "is_recoloring": self.is_recoloring,
            "removed_colors": sorted(list(self.removed_colors)),
            "added_colors": sorted(list(self.added_colors)),
        }
