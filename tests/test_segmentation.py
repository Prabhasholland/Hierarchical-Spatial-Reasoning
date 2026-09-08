"""Unit tests for object segmentation module."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.models import Grid
from src.objects.segmentation import (
    ObjectInstance,
    SegmentationHypothesis,
    SegmentationStrategy,
    detect_background,
    label_components,
    make_object,
    segment_grid,
    segment_grid_multi,
)


def test_label_components_4_and_8_connectivity():
    # Diagonal 2x2: (0,0) and (1,1)
    mask = np.array([
        [1, 0],
        [0, 1],
    ], dtype=bool)

    labeled_4, num_4 = label_components(mask, connectivity=4)
    assert num_4 == 2

    labeled_8, num_8 = label_components(mask, connectivity=8)
    assert num_8 == 1


def test_detect_background():
    # 0 is dominant
    grid_0 = Grid.from_list([
        [0, 0, 1],
        [0, 2, 0],
    ])
    assert detect_background(grid_0) == 0

    # 3 is dominant
    grid_3 = Grid.from_list([
        [3, 3, 1],
        [3, 3, 2],
    ])
    assert detect_background(grid_3) == 3


def test_object_attributes_computation():
    grid = Grid.from_list([
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
    ])

    hyp = segment_grid(grid, strategy=SegmentationStrategy.MONOCHROMATIC_4)
    assert len(hyp.objects) == 1

    obj = hyp.objects[0]
    assert obj.color == 1
    assert obj.area == 4
    assert obj.bounding_box == (0, 1, 1, 2)
    assert obj.min_row == 0
    assert obj.max_row == 1
    assert obj.min_col == 1
    assert obj.max_col == 2
    assert obj.height == 2
    assert obj.width == 2
    assert obj.centroid == (0.5, 1.5)
    assert obj.aspect_ratio == 1.0
    assert obj.density == 1.0
    assert obj.perimeter == 4
    assert obj.is_symmetric_h is True
    assert obj.is_symmetric_v is True


def test_segment_grid_multi_hypotheses():
    # Grid with touching different colors
    grid = Grid.from_list([
        [1, 2, 0],
        [0, 0, 0],
    ])

    hyps = segment_grid_multi(grid)
    assert len(hyps) == 4

    mono_4 = [h for h in hyps if h.strategy == SegmentationStrategy.MONOCHROMATIC_4][0]
    multi_4 = [h for h in hyps if h.strategy == SegmentationStrategy.MULTICOLOR_4][0]

    assert len(mono_4.objects) == 2
    assert len(multi_4.objects) == 1
