"""Transformation library for the ARC reasoning solver."""

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
    AntiTransposeTransformation,
    HorizontalFlipTransformation,
    IdentityTransformation,
    Rotate90Transformation,
    Rotate180Transformation,
    Rotate270Transformation,
    TransposeTransformation,
    VerticalFlipTransformation,
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

__all__ = [
    "Transformation",
    "IdentityTransformation",
    "Rotate90Transformation",
    "Rotate180Transformation",
    "Rotate270Transformation",
    "HorizontalFlipTransformation",
    "VerticalFlipTransformation",
    "TransposeTransformation",
    "AntiTransposeTransformation",
    "ALL_GEOMETRIC_TRANSFORMATIONS",
    "ColorSubstitutionTransformation",
    "ColorSwapTransformation",
    "ColorReplaceTransformation",
    "InvertForegroundTransformation",
    "TranslateTransformation",
    "CropBoundingBoxTransformation",
    "CropColorObjectTransformation",
    "CropFixedTransformation",
    "PadTransformation",
    "TileTransformation",
    "KroneckerScaleTransformation",
    "FractalTilingTransformation",
    "SymmetryCompletionTransformation",
    "GravityTransformation",
    "CompositeTransformation",
]
