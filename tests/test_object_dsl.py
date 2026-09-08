"""Unit tests for object-level DSL operations."""

from __future__ import annotations

from src.data.models import Grid
from src.objects.dsl import (
    count_objects,
    extract_object,
    filter_by_area,
    filter_by_color,
    recolor_objects,
    render_objects,
    select_largest,
    select_smallest,
    sort_by_area,
    translate_objects,
)
from src.objects.segmentation import segment_grid


def test_filtering_and_sorting():
    grid = Grid.from_list([
        [1, 1, 0, 2],
        [1, 1, 0, 0],
    ])
    hyp = segment_grid(grid)
    objs = hyp.objects

    # Color filtering
    c1 = filter_by_color(objs, 1)
    assert len(c1) == 1
    assert c1[0].area == 4

    # Area filtering
    large = filter_by_area(objs, min_area=2)
    assert len(large) == 1
    assert large[0].color == 1

    # Selection
    largest = select_largest(objs)
    smallest = select_smallest(objs)
    assert largest.color == 1
    assert smallest.color == 2

    # Sorting
    sorted_asc = sort_by_area(objs, descending=False)
    assert sorted_asc[0].color == 2
    assert sorted_asc[1].color == 1


def test_mutation_and_rendering():
    grid = Grid.from_list([
        [1, 0],
        [0, 0],
    ])
    hyp = segment_grid(grid)
    objs = hyp.objects

    # Recolor
    recolored = recolor_objects(objs, new_color=3)
    assert recolored[0].color == 3
    # Original should be unchanged
    assert objs[0].color == 1

    # Translate
    moved = translate_objects(objs, dr=1, dc=1)
    assert moved[0].pixel_coords == frozenset({(1, 1)})

    # Render
    rendered = render_objects(moved, grid_shape=(2, 2), background=0)
    assert rendered.get(1, 1) == 1
    assert rendered.get(0, 0) == 0

    # Extract
    extracted = extract_object(objs[0])
    assert extracted.shape == (1, 1)
    assert extracted.get(0, 0) == 1
