"""Composition of multiple transformations into a sequential pipeline."""

from __future__ import annotations

from typing import Any, Sequence

from src.data.models import Grid
from src.transformations.base import Transformation


class CompositeTransformation(Transformation):
    """Sequential composition of transformations: (T_k ∘ ... ∘ T_1)(G)."""

    def __init__(self, steps: Sequence[Transformation]) -> None:
        if not steps:
            raise ValueError("CompositeTransformation requires at least one step.")
        # Flatten nested composite transformations
        flattened: list[Transformation] = []
        for s in steps:
            if isinstance(s, CompositeTransformation):
                flattened.extend(s.steps)
            else:
                flattened.append(s)
        self._steps = tuple(flattened)

    @property
    def steps(self) -> tuple[Transformation, ...]:
        return self._steps

    def apply(self, grid: Grid) -> Grid:
        current = grid
        for step in self._steps:
            current = step.apply(current)
        return current

    @property
    def name(self) -> str:
        step_names = " -> ".join(s.name for s in self._steps)
        return f"Composite[{step_names}]"

    @property
    def params(self) -> dict[str, Any]:
        return {"steps": [s.params for s in self._steps]}

    @property
    def complexity(self) -> float:
        # Sum of step complexities plus composition penalty
        return sum(s.complexity for s in self._steps) + 0.3 * (len(self._steps) - 1)

    def __repr__(self) -> str:
        steps_repr = " -> ".join(repr(s) for s in self._steps)
        return f"Composite({steps_repr})"
