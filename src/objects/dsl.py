"""Interpretable object-level Domain-Specific Language (DSL) for ARC reasoning.

Every DSL operation is deterministic, pure, composable, and operates on ObjectInstance entities.
"""

from __future__ import annotations

from typing import Sequence

from src.data.models import Grid
from src.objects.relations import adjacent_to, contains
from src.objects.segmentation import ObjectInstance, _compute_shape_signature


# --------------------------------------------------------------------------
# 1. Filtering Operations
# --------------------------------------------------------------------------


def filter_by_color(objects: Sequence[ObjectInstance], color: int) -> list[ObjectInstance]:
    """Keep only objects whose primary color matches the target color."""
    return [obj for obj in objects if obj.color == color or (obj.color is None and color in obj.colors)]


def filter_by_area(
    objects: Sequence[ObjectInstance],
    min_area: int = 1,
    max_area: int = 900,
) -> list[ObjectInstance]:
    """Keep objects whose pixel area falls in [min_area, max_area]."""
    return [obj for obj in objects if min_area <= obj.area <= max_area]


def filter_by_size(
    objects: Sequence[ObjectInstance],
    height: int | None = None,
    width: int | None = None,
) -> list[ObjectInstance]:
    """Keep objects matching given bounding box height and/or width."""
    res = []
    for obj in objects:
        if height is not None and obj.height != height:
            continue
        if width is not None and obj.width != width:
            continue
        res.append(obj)
    return res


def filter_by_position(
    objects: Sequence[ObjectInstance],
    region: str,
    grid_shape: tuple[int, int],
) -> list[ObjectInstance]:
    """Keep objects whose centroid falls within a specified region of the grid canvas.
    
    Regions: 'top', 'bottom', 'left', 'right', 'center'
    """
    h, w = grid_shape
    mid_r = h / 2.0
    mid_c = w / 2.0

    res = []
    for obj in objects:
        cr, cc = obj.centroid
        if region == "top" and cr < mid_r:
            res.append(obj)
        elif region == "bottom" and cr >= mid_r:
            res.append(obj)
        elif region == "left" and cc < mid_c:
            res.append(obj)
        elif region == "right" and cc >= mid_c:
            res.append(obj)
        elif region == "center":
            if (h * 0.25 <= cr <= h * 0.75) and (w * 0.25 <= cc <= w * 0.75):
                res.append(obj)
    return res


def filter_by_shape(
    objects: Sequence[ObjectInstance],
    shape_sig: tuple[tuple[int, ...], ...],
) -> list[ObjectInstance]:
    """Keep objects with matching binary shape signature."""
    return [obj for obj in objects if obj.shape_signature == shape_sig]


# --------------------------------------------------------------------------
# 2. Sorting Operations
# --------------------------------------------------------------------------


def sort_by_area(
    objects: Sequence[ObjectInstance],
    descending: bool = False,
) -> list[ObjectInstance]:
    """Sort objects by area."""
    return sorted(objects, key=lambda obj: (obj.area, obj.object_id), reverse=descending)


def sort_by_position(
    objects: Sequence[ObjectInstance],
    key: str = "top_to_bottom",
) -> list[ObjectInstance]:
    """Sort objects by coordinate position."""
    if key == "top_to_bottom":
        return sorted(objects, key=lambda obj: (obj.min_row, obj.min_col))
    elif key == "bottom_to_top":
        return sorted(objects, key=lambda obj: (-obj.max_row, obj.min_col))
    elif key == "left_to_right":
        return sorted(objects, key=lambda obj: (obj.min_col, obj.min_row))
    elif key == "right_to_left":
        return sorted(objects, key=lambda obj: (-obj.max_col, obj.min_row))
    elif key == "top_left_to_bottom_right":
        return sorted(objects, key=lambda obj: (obj.centroid[0] + obj.centroid[1]))
    return list(objects)


# --------------------------------------------------------------------------
# 3. Selection Operations
# --------------------------------------------------------------------------


def select_largest(objects: Sequence[ObjectInstance]) -> ObjectInstance | None:
    """Return the single object with the largest area."""
    if not objects:
        return None
    return max(objects, key=lambda obj: (obj.area, -obj.object_id))


def select_smallest(objects: Sequence[ObjectInstance]) -> ObjectInstance | None:
    """Return the single object with the smallest area."""
    if not objects:
        return None
    return min(objects, key=lambda obj: (obj.area, obj.object_id))


def select_unique(objects: Sequence[ObjectInstance]) -> list[ObjectInstance]:
    """Return objects whose shape_signature appears exactly once in the collection."""
    sig_counts: dict[tuple[tuple[int, ...], ...], int] = {}
    for obj in objects:
        sig_counts[obj.shape_signature] = sig_counts.get(obj.shape_signature, 0) + 1
    return [obj for obj in objects if sig_counts[obj.shape_signature] == 1]


def select_matching(
    objects: Sequence[ObjectInstance],
    template: ObjectInstance,
) -> list[ObjectInstance]:
    """Return objects sharing identical shape signature with template."""
    return [obj for obj in objects if obj.shape_signature == template.shape_signature]


def select_adjacent(
    objects: Sequence[ObjectInstance],
    target: ObjectInstance,
) -> list[ObjectInstance]:
    """Return objects 4-connected adjacent to target object."""
    return [obj for obj in objects if obj.object_id != target.object_id and adjacent_to(obj, target)]


def select_contained(
    objects: Sequence[ObjectInstance],
    container: ObjectInstance,
) -> list[ObjectInstance]:
    """Return objects whose bounding box is completely inside container."""
    return [obj for obj in objects if obj.object_id != container.object_id and contains(container, obj)]


# --------------------------------------------------------------------------
# 4. Counting & Aggregation
# --------------------------------------------------------------------------


def count_objects(objects: Sequence[ObjectInstance]) -> int:
    """Return the cardinality of the object set."""
    return len(objects)


# --------------------------------------------------------------------------
# 5. Mutation Operations (Pure functions returning new instances)
# --------------------------------------------------------------------------


def recolor_object(obj: ObjectInstance, new_color: int) -> ObjectInstance:
    """Return a new copy of obj recolored to new_color."""
    return ObjectInstance(
        object_id=obj.object_id,
        color=new_color,
        colors=frozenset({new_color}),
        pixel_coords=obj.pixel_coords,
        bounding_box=obj.bounding_box,
        shape_signature=obj.shape_signature,
    )


def recolor_objects(
    objects: Sequence[ObjectInstance],
    new_color: int,
) -> list[ObjectInstance]:
    """Return new copies of all objects recolored to new_color."""
    return [recolor_object(obj, new_color) for obj in objects]


def translate_object(obj: ObjectInstance, dr: int, dc: int) -> ObjectInstance:
    """Return a new copy of obj translated by (dr, dc) delta."""
    new_coords = frozenset((r + dr, c + dc) for r, c in obj.pixel_coords)
    new_bb = (
        obj.min_row + dr,
        obj.min_col + dc,
        obj.max_row + dr,
        obj.max_col + dc,
    )
    return ObjectInstance(
        object_id=obj.object_id,
        color=obj.color,
        colors=obj.colors,
        pixel_coords=new_coords,
        bounding_box=new_bb,
        shape_signature=obj.shape_signature,
    )


def translate_objects(
    objects: Sequence[ObjectInstance],
    dr: int,
    dc: int,
) -> list[ObjectInstance]:
    """Return new copies of all objects translated by (dr, dc)."""
    return [translate_object(obj, dr, dc) for obj in objects]


def extract_object(obj: ObjectInstance) -> Grid:
    """Extract object as a cropped Grid matching its bounding box."""
    return obj.crop()


# --------------------------------------------------------------------------
# 6. Rendering Operations
# --------------------------------------------------------------------------


def render_objects(
    objects: Sequence[ObjectInstance],
    grid_shape: tuple[int, int],
    background: int = 0,
) -> Grid:
    """Render a sequence of objects onto a blank canvas.
    
    Later objects paint over earlier ones at shared pixel locations.
    """
    h, w = grid_shape
    canvas = [[background] * w for _ in range(h)]

    for obj in objects:
        paint_color = obj.color if obj.color is not None else (sorted(list(obj.colors))[0] if obj.colors else 0)
        for r, c in obj.pixel_coords:
            if 0 <= r < h and 0 <= c < w:
                canvas[r][c] = paint_color

    return Grid.from_list(canvas)
