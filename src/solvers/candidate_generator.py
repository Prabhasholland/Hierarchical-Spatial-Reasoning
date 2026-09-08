"""Candidate transformation generator for ARC tasks.

Systematically proposes hypothesis transformations based on task dimensions,
observed color correspondences, spatial properties, symmetries, and compositions.
"""

from __future__ import annotations

from typing import Sequence

from src.data.models import ARCTask, Grid
from src.transformations.base import Transformation
from src.transformations.color import (
    ColorReplaceTransformation,
    ColorSubstitutionTransformation,
    ColorSwapTransformation,
    InvertForegroundTransformation,
)
from src.transformations.composite import CompositeTransformation
from src.transformations.geometric import (
    ALL_GEOMETRIC_TRANSFORMATIONS,
    IdentityTransformation,
)
from src.transformations.object import (
    GravityTransformation,
    SymmetryCompletionTransformation,
)
from src.transformations.spatial import (
    CropBoundingBoxTransformation,
    CropColorObjectTransformation,
    CropFixedTransformation,
    PadTransformation,
    TranslateTransformation,
)
from src.transformations.tiling import (
    FractalTilingTransformation,
    KroneckerScaleTransformation,
    TileTransformation,
)


def deduce_color_mapping(in_grids: Sequence[Grid], out_grids: Sequence[Grid]) -> dict[int, int] | None:
    """Deduce a consistent point-wise color mapping between input and output grids."""
    if len(in_grids) != len(out_grids):
        return None

    mapping: dict[int, int] = {}
    for in_g, out_g in zip(in_grids, out_grids):
        if in_g.shape != out_g.shape:
            return None
        for r in range(in_g.height):
            for c in range(in_g.width):
                src_val = in_g.cells[r][c]
                dst_val = out_g.cells[r][c]
                if src_val in mapping:
                    if mapping[src_val] != dst_val:
                        return None  # Inconsistent mapping
                else:
                    mapping[src_val] = dst_val

    return mapping


class CandidateGenerator:
    """Generates candidate transformation hypotheses for an ARC task."""

    def __init__(self, max_candidates: int = 500) -> None:
        self.max_candidates = max_candidates

    def generate(self, task: ARCTask) -> list[Transformation]:
        """Generate a list of candidate transformations to test against the task demonstrations.
        
        Args:
            task: The ARCTask containing training demonstration pairs.
            
        Returns:
            List of candidate Transformation instances.
        """
        candidates: list[Transformation] = []
        train_inputs = task.all_train_inputs
        train_outputs = task.all_train_outputs
        colors_used = sorted(list(task.unique_colors))

        # 1. Standard geometric transformations
        candidates.extend(ALL_GEOMETRIC_TRANSFORMATIONS)

        # 2. Inferred direct color substitutions
        direct_cmap = deduce_color_mapping(train_inputs, train_outputs)
        if direct_cmap is not None and direct_cmap:
            candidates.append(ColorSubstitutionTransformation(direct_cmap))

        # 3. Geometric + Color Substitution Compositions
        for geom in ALL_GEOMETRIC_TRANSFORMATIONS:
            if isinstance(geom, IdentityTransformation):
                continue
            # Apply geometric transform to inputs and check for color mapping
            try:
                geom_inputs = [geom.apply(g) for g in train_inputs]
                geom_cmap = deduce_color_mapping(geom_inputs, train_outputs)
                if geom_cmap is not None and geom_cmap:
                    candidates.append(
                        CompositeTransformation([geom, ColorSubstitutionTransformation(geom_cmap)])
                    )
            except Exception:
                pass

        # 4. Pairwise color swaps and single-color replacements
        for c1 in colors_used:
            for c2 in colors_used:
                if c1 < c2:
                    candidates.append(ColorSwapTransformation(c1, c2))
                if c1 != c2:
                    candidates.append(ColorReplaceTransformation(c1, c2))

        # 5. Foreground inversion / recoloring
        for target_c in colors_used:
            if target_c != 0:
                candidates.append(InvertForegroundTransformation(target_color=target_c, background=0))

        # 6. Spatial translations (small shifts dx, dy in [-3, 3])
        for dr in (-2, -1, 0, 1, 2):
            for dc in (-2, -1, 0, 1, 2):
                if dr != 0 or dc != 0:
                    candidates.append(TranslateTransformation(dr, dc, fill_color=0, wrap=False))
                    candidates.append(TranslateTransformation(dr, dc, fill_color=0, wrap=True))

        # 7. Symmetry completions and Gravity
        for axis in ("horizontal", "vertical", "both"):
            candidates.append(SymmetryCompletionTransformation(axis=axis, background=0))
        for direction in ("down", "up", "left", "right"):
            candidates.append(GravityTransformation(direction=direction, background=0))

        # 8. Bounding box cropping
        for pad in (0, 1):
            candidates.append(CropBoundingBoxTransformation(background=0, pad=pad))
            # Test Crop + Geometric compositions
            for geom in [
                IdentityTransformation(),
                ALL_GEOMETRIC_TRANSFORMATIONS[1],  # Rotate90
                ALL_GEOMETRIC_TRANSFORMATIONS[2],  # Rotate180
                ALL_GEOMETRIC_TRANSFORMATIONS[4],  # HFlip
            ]:
                if not isinstance(geom, IdentityTransformation):
                    candidates.append(
                        CompositeTransformation([CropBoundingBoxTransformation(background=0, pad=pad), geom])
                    )

        # 9. Color object cropping
        for c in colors_used:
            if c != 0:
                candidates.append(CropColorObjectTransformation(target_color=c))

        # 10. Fixed window crops if input > output dimensions consistently
        first_in = train_inputs[0]
        first_out = train_outputs[0]
        if all(
            p.input.height >= p.output.height and p.input.width >= p.output.width
            for p in task.train
        ):
            out_h, out_w = first_out.height, first_out.width
            in_h, in_w = first_in.height, first_in.width
            if out_h <= in_h and out_w <= in_w:
                candidates.extend([
                    CropFixedTransformation(0, out_h, 0, out_w),  # Top-left
                    CropFixedTransformation(0, out_h, in_w - out_w, in_w),  # Top-right
                    CropFixedTransformation(in_h - out_h, in_h, 0, out_w),  # Bottom-left
                    CropFixedTransformation(in_h - out_h, in_h, in_w - out_w, in_w),  # Bottom-right
                    CropFixedTransformation(
                        (in_h - out_h) // 2, (in_h - out_h) // 2 + out_h,
                        (in_w - out_w) // 2, (in_w - out_w) // 2 + out_w,
                    ),  # Center
                ])

        # 11. Tiling, scaling, and fractal expansions
        # Check if output is integer multiple of input
        if first_out.height % first_in.height == 0 and first_out.width % first_in.width == 0:
            scale_r = first_out.height // first_in.height
            scale_c = first_out.width // first_in.width
            if scale_r >= 1 and scale_c >= 1:
                candidates.append(TileTransformation(scale_r, scale_c))
                candidates.append(KroneckerScaleTransformation(scale_r, scale_c))

        # Fractal expansion (e.g. 3x3 -> 9x9)
        if first_out.height == first_in.height * first_in.height and first_out.width == first_in.width * first_in.width:
            candidates.append(FractalTilingTransformation(background=0))

        # Remove duplicate candidate transformations while preserving order
        unique_candidates: list[Transformation] = []
        seen = set()
        for cand in candidates:
            cand_repr = repr(cand)
            if cand_repr not in seen:
                seen.add(cand_repr)
                unique_candidates.append(cand)
                if len(unique_candidates) >= self.max_candidates:
                    break

        return unique_candidates
