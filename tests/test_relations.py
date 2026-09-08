"""Unit tests for spatial and relational predicates."""

from __future__ import annotations

from src.data.models import Grid
from src.objects.relations import (
    above,
    adjacent_to,
    below,
    compute_all_pairwise_relations,
    contains,
    distance_between,
    horizontally_aligned,
    inside,
    left_of,
    nearest_to,
    overlaps,
    right_of,
    same_color,
    same_shape_signature,
    same_size,
    touching,
    vertically_aligned,
)
from src.objects.segmentation import segment_grid


def test_spatial_predicates():
    grid = Grid.from_list([
        [1, 0, 2],
        [0, 0, 0],
        [3, 0, 4],
    ])

    hyp = segment_grid(grid)
    objs = sorted(hyp.objects, key=lambda o: o.color)
    o1, o2, o3, o4 = objs  # colors 1, 2, 3, 4

    assert left_of(o1, o2) is True
    assert right_of(o2, o1) is True
    assert above(o1, o3) is True
    assert below(o3, o1) is True

    assert horizontally_aligned(o1, o2) is True
    assert vertically_aligned(o1, o3) is True

    assert same_size(o1, o2) is True
    assert same_shape_signature(o1, o2) is True
    assert same_color(o1, o2) is False


def test_adjacency_and_touching():
    # 1 and 2 are adjacent
    # 1 and 3 are diagonally touching
    grid = Grid.from_list([
        [1, 2],
        [0, 3],
    ])
    hyp = segment_grid(grid)
    objs = {o.color: o for o in hyp.objects}

    assert adjacent_to(objs[1], objs[2]) is True
    assert touching(objs[1], objs[3]) is True
    assert adjacent_to(objs[1], objs[3]) is False


def test_contains_and_inside():
    # Outer frame 1 contains center 2 (with background=0)
    grid = Grid.from_list([
        [1, 1, 1],
        [1, 2, 1],
        [1, 1, 1],
    ])
    hyp = segment_grid(grid, background=0)
    objs = {o.color: o for o in hyp.objects}

    assert contains(objs[1], objs[2]) is True
    assert inside(objs[2], objs[1]) is True


def test_compute_all_pairwise_relations():
    grid = Grid.from_list([
        [1, 2],
        [0, 0],
    ])
    hyp = segment_grid(grid)
    relations = compute_all_pairwise_relations(hyp.objects, hyp.grid_shape)
    assert len(relations) == 2
    assert "adjacent_to" in relations[0]["relations"]
