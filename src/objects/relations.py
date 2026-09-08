"""Spatial and relational predicates for ARC objects.

Implements deterministic geometric and semantic predicates comparing pairs
of ObjectInstance objects.
"""

from __future__ import annotations

import math
from typing import Any

from src.objects.segmentation import ObjectInstance


def adjacent_to(a: ObjectInstance, b: ObjectInstance) -> bool:
    """True if any pixel in a is cardinally adjacent (4-connected) to any pixel in b without overlapping."""
    if overlaps(a, b):
        return False
    # Quick bounding box filter
    if (a.max_row < b.min_row - 1 or a.min_row > b.max_row + 1 or
        a.max_col < b.min_col - 1 or a.min_col > b.max_col + 1):
        return False

    b_coords = b.pixel_coords
    for r, c in a.pixel_coords:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if (r + dr, c + dc) in b_coords:
                return True
    return False


def touching(a: ObjectInstance, b: ObjectInstance) -> bool:
    """True if any pixel in a is 8-connected adjacent to any pixel in b without overlapping."""
    if overlaps(a, b):
        return False
    if (a.max_row < b.min_row - 1 or a.min_row > b.max_row + 1 or
        a.max_col < b.min_col - 1 or a.min_col > b.max_col + 1):
        return False

    b_coords = b.pixel_coords
    for r, c in a.pixel_coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                if (r + dr, c + dc) in b_coords:
                    return True
    return False


def overlaps(a: ObjectInstance, b: ObjectInstance) -> bool:
    """True if a and b share any pixel coordinates."""
    return bool(a.pixel_coords & b.pixel_coords)


def contains(a: ObjectInstance, b: ObjectInstance) -> bool:
    """True if all pixels of b are within the bounding box of a."""
    if not b.pixel_coords:
        return False
    return (a.min_row <= b.min_row and a.max_row >= b.max_row and
            a.min_col <= b.min_col and a.max_col >= b.max_col)


def inside(a: ObjectInstance, b: ObjectInstance) -> bool:
    """True if all pixels of a are within the bounding box of b."""
    return contains(b, a)


def left_of(a: ObjectInstance, b: ObjectInstance) -> bool:
    """True if a is strictly to the left of b (no column overlap)."""
    return a.max_col < b.min_col


def right_of(a: ObjectInstance, b: ObjectInstance) -> bool:
    """True if a is strictly to the right of b (no column overlap)."""
    return a.min_col > b.max_col


def above(a: ObjectInstance, b: ObjectInstance) -> bool:
    """True if a is strictly above b (no row overlap)."""
    return a.max_row < b.min_row


def below(a: ObjectInstance, b: ObjectInstance) -> bool:
    """True if a is strictly below b (no row overlap)."""
    return a.min_row > b.max_row


def horizontally_aligned(a: ObjectInstance, b: ObjectInstance, tolerance: float = 0.5) -> bool:
    """True if centroids share the same row within tolerance."""
    return abs(a.centroid[0] - b.centroid[0]) <= tolerance


def vertically_aligned(a: ObjectInstance, b: ObjectInstance, tolerance: float = 0.5) -> bool:
    """True if centroids share the same column within tolerance."""
    return abs(a.centroid[1] - b.centroid[1]) <= tolerance


def same_color(a: ObjectInstance, b: ObjectInstance) -> bool:
    """True if both objects have the same non-None primary color."""
    return a.color is not None and b.color is not None and a.color == b.color


def same_shape_signature(a: ObjectInstance, b: ObjectInstance) -> bool:
    """True if both objects have identical binary shape signatures."""
    return a.shape_signature == b.shape_signature


def same_size(a: ObjectInstance, b: ObjectInstance) -> bool:
    """True if both objects occupy the same number of pixels."""
    return a.area == b.area


def distance_between(a: ObjectInstance, b: ObjectInstance) -> float:
    """Euclidean distance between object centroids."""
    dr = a.centroid[0] - b.centroid[0]
    dc = a.centroid[1] - b.centroid[1]
    return math.sqrt(dr * dr + dc * dc)


def nearest_to(a: ObjectInstance, objects: list[ObjectInstance]) -> ObjectInstance | None:
    """Return the object in objects (excluding a) with minimum centroid distance to a."""
    candidates = [obj for obj in objects if obj.object_id != a.object_id]
    if not candidates:
        return None
    return min(candidates, key=lambda obj: distance_between(a, obj))


def symmetric_with(
    a: ObjectInstance,
    b: ObjectInstance,
    axis: str = "horizontal",
    grid_shape: tuple[int, int] = (30, 30),
) -> bool:
    """True if a and b are approximate reflections of each other across grid midpoint."""
    h, w = grid_shape
    if axis == "horizontal":
        # Reflection across vertical center line (left-right reflection)
        reflected_col = (w - 1) - a.centroid[1]
        return abs(b.centroid[1] - reflected_col) <= 1.0 and abs(b.centroid[0] - a.centroid[0]) <= 1.0
    elif axis == "vertical":
        # Reflection across horizontal center line (top-bottom reflection)
        reflected_row = (h - 1) - a.centroid[0]
        return abs(b.centroid[0] - reflected_row) <= 1.0 and abs(b.centroid[1] - a.centroid[1]) <= 1.0
    return False


def compute_all_pairwise_relations(
    objects: list[ObjectInstance],
    grid_shape: tuple[int, int] = (30, 30),
) -> list[dict[str, Any]]:
    """Compute all pairwise spatial and semantic relationships between objects."""
    relations: list[dict[str, Any]] = []
    n = len(objects)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a = objects[i]
            b = objects[j]

            predicates: list[str] = []
            if adjacent_to(a, b):
                predicates.append("adjacent_to")
            elif touching(a, b):
                predicates.append("touching")

            if overlaps(a, b):
                predicates.append("overlaps")
            if contains(a, b):
                predicates.append("contains")
            if inside(a, b):
                predicates.append("inside")
            if left_of(a, b):
                predicates.append("left_of")
            if right_of(a, b):
                predicates.append("right_of")
            if above(a, b):
                predicates.append("above")
            if below(a, b):
                predicates.append("below")
            if horizontally_aligned(a, b):
                predicates.append("horizontally_aligned")
            if vertically_aligned(a, b):
                predicates.append("vertically_aligned")
            if same_color(a, b):
                predicates.append("same_color")
            if same_shape_signature(a, b):
                predicates.append("same_shape_signature")
            if same_size(a, b):
                predicates.append("same_size")

            dist = distance_between(a, b)

            relations.append({
                "source_id": a.object_id,
                "target_id": b.object_id,
                "relations": predicates,
                "distance": round(dist, 2),
            })

    return relations
