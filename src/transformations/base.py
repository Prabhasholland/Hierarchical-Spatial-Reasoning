"""Base interface and abstractions for ARC transformations.

Every transformation implements a deterministic mapping from an input Grid
to an output Grid, with explicit mathematical definition, complexity scoring,
and parameter inspection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.data.models import Grid


class Transformation(ABC):
    """Abstract base class for all ARC grid transformations."""

    @abstractmethod
    def apply(self, grid: Grid) -> Grid:
        """Apply the transformation to an input grid.
        
        Args:
            grid: Input Grid instance.
            
        Returns:
            Transformed Grid instance.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the transformation."""
        ...

    @property
    def complexity(self) -> float:
        """Complexity score for candidate ranking (Occam's razor).
        
        Lower complexity means simpler, more general transformation.
        """
        return 1.0

    @property
    def params(self) -> dict[str, Any]:
        """Dictionary of transformation parameters."""
        return {}

    def __call__(self, grid: Grid) -> Grid:
        """Convenience caller for apply()."""
        return self.apply(grid)

    def __repr__(self) -> str:
        param_str = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{self.name}({param_str})" if param_str else f"{self.name}()"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.name == other.name and self.params == other.params

    def __hash__(self) -> int:
        param_tuple = tuple(sorted((k, str(v)) for k, v in self.params.items()))
        return hash((self.name, param_tuple))
