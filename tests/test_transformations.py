"""Unit tests for the transformation library."""

import pytest

from src.data.models import Grid
from src.transformations import (
    AntiTransposeTransformation,
    ColorReplaceTransformation,
    ColorSubstitutionTransformation,
    ColorSwapTransformation,
    CompositeTransformation,
    CropBoundingBoxTransformation,
    CropColorObjectTransformation,
    CropFixedTransformation,
    FractalTilingTransformation,
    GravityTransformation,
    HorizontalFlipTransformation,
    IdentityTransformation,
    InvertForegroundTransformation,
    KroneckerScaleTransformation,
    PadTransformation,
    Rotate90Transformation,
    Rotate180Transformation,
    Rotate270Transformation,
    SymmetryCompletionTransformation,
    TileTransformation,
    TranslateTransformation,
    TransposeTransformation,
    VerticalFlipTransformation,
)


def test_identity_transformation():
    g = Grid.from_list([[1, 2], [3, 4]])
    trans = IdentityTransformation()
    assert trans.apply(g) == g
    assert trans.complexity == 0.1


def test_rotations():
    g = Grid.from_list([
        [1, 2, 3],
        [4, 5, 6],
    ])
    # 90 clockwise: (2x3) -> (3x2)
    # [4, 1]
    # [5, 2]
    # [6, 3]
    rot90 = Rotate90Transformation().apply(g)
    assert rot90.shape == (3, 2)
    assert rot90.cells == ((4, 1), (5, 2), (6, 3))

    # 180: (2x3) -> (2x3)
    rot180 = Rotate180Transformation().apply(g)
    assert rot180.cells == ((6, 5, 4), (3, 2, 1))

    # 270: (2x3) -> (3x2)
    rot270 = Rotate270Transformation().apply(g)
    assert rot270.cells == ((3, 6), (2, 5), (1, 4))


def test_reflections_and_transpose():
    g = Grid.from_list([
        [1, 2, 3],
        [4, 5, 6],
    ])
    # Horizontal flip (left-right)
    hflip = HorizontalFlipTransformation().apply(g)
    assert hflip.cells == ((3, 2, 1), (6, 5, 4))

    # Vertical flip (up-down)
    vflip = VerticalFlipTransformation().apply(g)
    assert vflip.cells == ((4, 5, 6), (1, 2, 3))

    # Transpose
    tp = TransposeTransformation().apply(g)
    assert tp.cells == ((1, 4), (2, 5), (3, 6))

    # Anti-transpose
    atp = AntiTransposeTransformation().apply(g)
    assert atp.cells == ((6, 3), (5, 2), (4, 1))


def test_color_transformations():
    g = Grid.from_list([
        [1, 2],
        [0, 1],
    ])
    # Substitution
    csub = ColorSubstitutionTransformation({1: 3, 2: 4}).apply(g)
    assert csub.cells == ((3, 4), (0, 3))

    # Swap
    cswap = ColorSwapTransformation(1, 2).apply(g)
    assert cswap.cells == ((2, 1), (0, 2))

    # Replace
    crep = ColorReplaceTransformation(1, 7).apply(g)
    assert crep.cells == ((7, 2), (0, 7))

    # Invert foreground
    cinv = InvertForegroundTransformation(target_color=8, background=0).apply(g)
    assert cinv.cells == ((8, 8), (0, 8))


def test_spatial_translations():
    g = Grid.from_list([
        [0, 1, 0],
        [0, 0, 0],
        [0, 0, 0],
    ])
    # Shift down 1, right 1 (no wrap)
    t = TranslateTransformation(dr=1, dc=1, fill_color=0, wrap=False).apply(g)
    assert t.cells == (
        (0, 0, 0),
        (0, 0, 1),
        (0, 0, 0),
    )

    # Shift with wrap
    t_wrap = TranslateTransformation(dr=-1, dc=-1, wrap=True).apply(g)
    assert t_wrap.cells == (
        (0, 0, 0),
        (0, 0, 0),
        (1, 0, 0),
    )


def test_cropping():
    g = Grid.from_list([
        [0, 0, 0, 0],
        [0, 1, 2, 0],
        [0, 3, 4, 0],
        [0, 0, 0, 0],
    ])
    crop_bb = CropBoundingBoxTransformation(background=0).apply(g)
    assert crop_bb.shape == (2, 2)
    assert crop_bb.cells == ((1, 2), (3, 4))

    # Color object crop
    crop_color = CropColorObjectTransformation(target_color=1).apply(g)
    assert crop_color.cells == ((1,),)

    # Fixed crop
    crop_fixed = CropFixedTransformation(1, 3, 1, 3).apply(g)
    assert crop_fixed.cells == ((1, 2), (3, 4))


def test_padding():
    g = Grid.from_list([[1, 2], [3, 4]])
    pad = PadTransformation(top=1, bottom=1, left=1, right=1, fill_color=0).apply(g)
    assert pad.shape == (4, 4)
    assert pad.cells[0] == (0, 0, 0, 0)
    assert pad.cells[1] == (0, 1, 2, 0)
    assert pad.cells[2] == (0, 3, 4, 0)
    assert pad.cells[3] == (0, 0, 0, 0)


def test_tiling_and_scaling():
    g = Grid.from_list([[1, 2]])
    tiled = TileTransformation(n_rows=2, n_cols=2).apply(g)
    assert tiled.shape == (2, 4)
    assert tiled.cells == (
        (1, 2, 1, 2),
        (1, 2, 1, 2),
    )

    scaled = KroneckerScaleTransformation(scale_r=2, scale_c=2).apply(g)
    assert scaled.shape == (2, 4)
    assert scaled.cells == (
        (1, 1, 2, 2),
        (1, 1, 2, 2),
    )


def test_fractal_tiling():
    g = Grid.from_list([
        [1, 0],
        [0, 1],
    ])
    fractal = FractalTilingTransformation(background=0).apply(g)
    assert fractal.shape == (4, 4)
    assert fractal.cells == (
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 1, 0),
        (0, 0, 0, 1),
    )


def test_symmetry_and_gravity():
    g_sym = Grid.from_list([
        [1, 0, 0],
        [0, 2, 0],
        [0, 0, 0],
    ])
    sym = SymmetryCompletionTransformation(axis="horizontal", background=0).apply(g_sym)
    assert sym.cells == (
        (1, 0, 1),
        (0, 2, 0),
        (0, 0, 0),
    )

    g_grav = Grid.from_list([
        [0, 1, 0],
        [0, 0, 0],
        [2, 0, 0],
    ])
    grav_down = GravityTransformation(direction="down", background=0).apply(g_grav)
    assert grav_down.cells == (
        (0, 0, 0),
        (0, 0, 0),
        (2, 1, 0),
    )


def test_composite_transformation():
    g = Grid.from_list([
        [1, 2],
        [3, 4],
    ])
    # Rotate90 then replace 1 with 9
    comp = CompositeTransformation([
        Rotate90Transformation(),
        ColorReplaceTransformation(old_color=1, new_color=9),
    ])
    res = comp.apply(g)
    # rot90: ((3, 1), (4, 2)) -> replace 1 with 9: ((3, 9), (4, 2))
    assert res.cells == ((3, 9), (4, 2))
    assert comp.complexity > Rotate90Transformation().complexity
