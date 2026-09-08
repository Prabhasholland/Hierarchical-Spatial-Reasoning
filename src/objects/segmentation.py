"""Object segmentation and representation for ARC grids.

This module provides the ObjectInstance dataclass that represents a single
connected object extracted from an ARC grid, along with segmentation
utilities for identifying such objects under multiple hypotheses (4-connectivity,
8-connectivity, monochromatic, multicolor).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from src.data.models import Grid


def label_components(mask: np.ndarray, connectivity: int = 4) -> tuple[np.ndarray, int]:
    """Pure numpy/python connected component labeling.
    
    Args:
        mask: 2D boolean or integer binary array (non-zero is foreground).
        connectivity: 4 for cardinal neighbors, 8 for cardinal + diagonal.
        
    Returns:
        (labeled_array, num_components)
    """
    h, w = mask.shape
    labeled = np.zeros((h, w), dtype=np.int32)
    current_label = 0
    
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        offsets.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])

    for r in range(h):
        for c in range(w):
            if mask[r, c] and labeled[r, c] == 0:
                current_label += 1
                queue = deque([(r, c)])
                labeled[r, c] = current_label
                while queue:
                    curr_r, curr_c = queue.popleft()
                    for dr, dc in offsets:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            if mask[nr, nc] and labeled[nr, nc] == 0:
                                labeled[nr, nc] = current_label
                                queue.append((nr, nc))
    return labeled, current_label


def detect_background(grid: Grid) -> int:
    """Detect candidate background color from grid.
    
    If 0 is present in the grid, 0 is preferred as background.
    Otherwise, uses the most frequent color.
    """
    counts = grid.color_counts()
    if not counts:
        return 0
    if 0 in counts:
        return 0
    return max(counts.keys(), key=lambda k: counts[k])


@dataclass(frozen=True)
class ObjectInstance:
    """Immutable representation of a single segmented object from an ARC grid.

    Attributes:
        object_id: Unique integer identifier for this object.
        color: Primary color of the object, or None for multi-colored objects.
        colors: Frozenset of all distinct colors present in this object.
        pixel_coords: Frozenset of (row, col) coordinates comprising this object.
        bounding_box: (min_row, min_col, max_row, max_col) inclusive bounds.
        shape_signature: Normalised binary mask cropped to bounding box.
    """

    object_id: int
    color: int | None
    colors: frozenset[int]
    pixel_coords: frozenset[tuple[int, int]]
    bounding_box: tuple[int, int, int, int]
    shape_signature: tuple[tuple[int, ...], ...]

    # ------------------------------------------------------------------
    # Derived geometric properties
    # ------------------------------------------------------------------

    @property
    def min_row(self) -> int:
        """Minimum row index (top edge of bounding box)."""
        return self.bounding_box[0]

    @property
    def min_col(self) -> int:
        """Minimum column index (left edge of bounding box)."""
        return self.bounding_box[1]

    @property
    def max_row(self) -> int:
        """Maximum row index (bottom edge of bounding box)."""
        return self.bounding_box[2]

    @property
    def max_col(self) -> int:
        """Maximum column index (right edge of bounding box)."""
        return self.bounding_box[3]

    @property
    def height(self) -> int:
        """Height of the bounding box."""
        return self.max_row - self.min_row + 1

    @property
    def width(self) -> int:
        """Width of the bounding box."""
        return self.max_col - self.min_col + 1

    @property
    def area(self) -> int:
        """Number of pixels belonging to this object."""
        return len(self.pixel_coords)

    @property
    def centroid(self) -> tuple[float, float]:
        """Centroid (mean row, mean col) of the object's pixels."""
        if not self.pixel_coords:
            return (0.0, 0.0)
        rows = [r for r, _ in self.pixel_coords]
        cols = [c for _, c in self.pixel_coords]
        return (sum(rows) / len(rows), sum(cols) / len(cols))

    @property
    def aspect_ratio(self) -> float:
        """Aspect ratio (width / height). Returns 1.0 for degenerate cases."""
        if self.height == 0:
            return 1.0
        return self.width / self.height

    @property
    def density(self) -> float:
        """Fraction of bounding-box area occupied by object pixels."""
        bb_area = self.height * self.width
        if bb_area == 0:
            return 0.0
        return self.area / bb_area

    @property
    def perimeter(self) -> int:
        """Number of object pixels that are 4-adjacent to a non-object cell."""
        count = 0
        coords = self.pixel_coords
        for r, c in coords:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if (r + dr, c + dc) not in coords:
                    count += 1
                    break
        return count

    @property
    def is_symmetric_h(self) -> bool:
        """Whether the shape signature is horizontally symmetric (left-right)."""
        for row in self.shape_signature:
            if row != row[::-1]:
                return False
        return True

    @property
    def is_symmetric_v(self) -> bool:
        """Whether the shape signature is vertically symmetric (top-bottom)."""
        return self.shape_signature == self.shape_signature[::-1]

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the object instance to a plain dictionary."""
        return {
            "object_id": self.object_id,
            "color": self.color,
            "colors": sorted(list(self.colors)),
            "area": self.area,
            "bounding_box": list(self.bounding_box),
            "min_row": self.min_row,
            "max_row": self.max_row,
            "min_col": self.min_col,
            "max_col": self.max_col,
            "height": self.height,
            "width": self.width,
            "centroid": [round(self.centroid[0], 2), round(self.centroid[1], 2)],
            "aspect_ratio": round(self.aspect_ratio, 2),
            "density": round(self.density, 4),
            "perimeter": self.perimeter,
            "is_symmetric_h": self.is_symmetric_h,
            "is_symmetric_v": self.is_symmetric_v,
            "pixel_coords": sorted(list(self.pixel_coords)),
        }

    def render(self, height: int, width: int, background: int = 0) -> Grid:
        """Render this object onto a grid of the given dimensions."""
        grid = [[background] * width for _ in range(height)]
        paint_color = self.color if self.color is not None else (sorted(list(self.colors))[0] if self.colors else 0)
        for r, c in self.pixel_coords:
            if 0 <= r < height and 0 <= c < width:
                grid[r][c] = paint_color
        return Grid.from_list(grid)

    def crop(self) -> Grid:
        """Extract the object as a cropped Grid covering only the bounding box."""
        h, w = self.height, self.width
        grid = [[0] * w for _ in range(h)]
        paint_color = self.color if self.color is not None else (sorted(list(self.colors))[0] if self.colors else 0)
        for r, c in self.pixel_coords:
            grid[r - self.min_row][c - self.min_col] = paint_color
        return Grid.from_list(grid)


def _compute_shape_signature(
    pixel_coords: frozenset[tuple[int, int]],
    min_row: int,
    min_col: int,
    height: int,
    width: int,
) -> tuple[tuple[int, ...], ...]:
    """Build a normalised binary mask from pixel coordinates."""
    mask = [[0] * width for _ in range(height)]
    for r, c in pixel_coords:
        mask[r - min_row][c - min_col] = 1
    return tuple(tuple(row) for row in mask)


def make_object(
    object_id: int,
    pixel_coords: frozenset[tuple[int, int]],
    grid: Grid,
) -> ObjectInstance:
    """Create an ObjectInstance from raw pixel coordinates and a source grid."""
    if not pixel_coords:
        raise ValueError("Cannot create ObjectInstance from empty pixel set.")

    rows = [r for r, _ in pixel_coords]
    cols = [c for _, c in pixel_coords]
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)
    bb = (min_row, min_col, max_row, max_col)
    height = max_row - min_row + 1
    width = max_col - min_col + 1

    color_set: set[int] = set()
    for r, c in pixel_coords:
        color_set.add(grid.get(r, c))

    primary_color: int | None = list(color_set)[0] if len(color_set) == 1 else None

    shape_sig = _compute_shape_signature(
        pixel_coords, min_row, min_col, height, width
    )

    return ObjectInstance(
        object_id=object_id,
        color=primary_color,
        colors=frozenset(color_set),
        pixel_coords=pixel_coords,
        bounding_box=bb,
        shape_signature=shape_sig,
    )


class SegmentationStrategy(str, Enum):
    """Supported object segmentation strategies."""
    MONOCHROMATIC_4 = "MONOCHROMATIC_4"
    MONOCHROMATIC_8 = "MONOCHROMATIC_8"
    MULTICOLOR_4 = "MULTICOLOR_4"
    MULTICOLOR_8 = "MULTICOLOR_8"


@dataclass
class SegmentationHypothesis:
    """Represents a full segmentation of a grid under a specific strategy."""
    strategy: SegmentationStrategy
    objects: list[ObjectInstance]
    background: int
    grid_shape: tuple[int, int]

    @property
    def num_objects(self) -> int:
        return len(self.objects)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy.value if hasattr(self.strategy, "value") else str(self.strategy),
            "background": self.background,
            "grid_shape": list(self.grid_shape),
            "num_objects": self.num_objects,
            "objects": [obj.to_dict() for obj in self.objects],
        }

    def __str__(self) -> str:
        strat = self.strategy.value if hasattr(self.strategy, "value") else str(self.strategy)
        return (
            f"SegmentationHypothesis(strategy={strat}, "
            f"objects={len(self.objects)}, bg={self.background}, shape={self.grid_shape})"
        )


def segment_grid(
    grid: Grid,
    strategy: SegmentationStrategy | str = SegmentationStrategy.MONOCHROMATIC_4,
    background: int | None = None,
) -> SegmentationHypothesis:
    """Segment a grid into objects according to a chosen strategy.
    
    Args:
        grid: Input Grid to segment.
        strategy: SegmentationStrategy enum or string name.
        background: Background color (auto-detected if None).
        
    Returns:
        SegmentationHypothesis containing extracted ObjectInstances.
    """
    if isinstance(strategy, str):
        try:
            strategy = SegmentationStrategy(strategy)
        except ValueError:
            strategy = SegmentationStrategy.MONOCHROMATIC_4

    if background is None:
        background = detect_background(grid)

    arr = grid.to_numpy()
    h, w = arr.shape
    objects: list[ObjectInstance] = []
    obj_id = 0

    if strategy in (SegmentationStrategy.MONOCHROMATIC_4, SegmentationStrategy.MONOCHROMATIC_8):
        conn = 4 if strategy == SegmentationStrategy.MONOCHROMATIC_4 else 8
        for c in range(10):
            if c == background:
                continue
            mask = (arr == c)
            if not np.any(mask):
                continue
            labeled, num_features = label_components(mask, connectivity=conn)
            for f_idx in range(1, num_features + 1):
                coords = np.argwhere(labeled == f_idx)
                pixel_set = frozenset((int(r), int(col)) for r, col in coords)
                obj = make_object(obj_id, pixel_set, grid)
                objects.append(obj)
                obj_id += 1
    else:
        # Multicolor
        conn = 4 if strategy == SegmentationStrategy.MULTICOLOR_4 else 8
        mask = (arr != background)
        if np.any(mask):
            labeled, num_features = label_components(mask, connectivity=conn)
            for f_idx in range(1, num_features + 1):
                coords = np.argwhere(labeled == f_idx)
                pixel_set = frozenset((int(r), int(col)) for r, col in coords)
                obj = make_object(obj_id, pixel_set, grid)
                objects.append(obj)
                obj_id += 1

    return SegmentationHypothesis(
        strategy=strategy,
        objects=objects,
        background=background,
        grid_shape=(h, w),
    )


def segment_grid_multi(grid: Grid, background: int | None = None) -> list[SegmentationHypothesis]:
    """Generate multiple segmentation hypotheses across all 4 strategies."""
    if background is None:
        background = detect_background(grid)

    hypotheses = []
    for strat in SegmentationStrategy:
        hyp = segment_grid(grid, strategy=strat, background=background)
        hypotheses.append(hyp)
    return hypotheses
