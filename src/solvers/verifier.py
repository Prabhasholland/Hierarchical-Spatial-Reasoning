"""Candidate verification and ranking for the ARC rule-based solver."""

from __future__ import annotations

from typing import Sequence

from src.data.models import ARCTask
from src.evaluation.metrics import grid_match
from src.transformations.base import Transformation


class CandidateVerifier:
    """Verifies candidate transformations against task training pairs."""

    def verify(self, transformation: Transformation, task: ARCTask) -> bool:
        """Check whether a candidate transformation produces exact matches for ALL training pairs.
        
        Args:
            transformation: The candidate transformation to test.
            task: The ARCTask containing training demonstrations.
            
        Returns:
            True if transformation exactly reproduces every training output from its input.
        """
        for pair in task.train:
            try:
                predicted = transformation.apply(pair.input)
                if not grid_match(predicted.to_list(), pair.output.to_list()):
                    return False
            except Exception:
                # Any transformation failure (e.g. invalid bounds) immediately invalidates candidate
                return False
        return True

    def find_all_consistent(
        self, candidates: Sequence[Transformation], task: ARCTask
    ) -> list[Transformation]:
        """Filter a collection of candidates to only those that explain all training pairs."""
        return [cand for cand in candidates if self.verify(cand, task)]


class CandidateRanker:
    """Ranks consistent transformation candidates by simplicity (Occam's razor) and specificity."""

    def rank(
        self, candidates: Sequence[Transformation], task: ARCTask
    ) -> list[Transformation]:
        """Sort consistent candidates in order of preference (lowest complexity first)."""
        return sorted(
            candidates,
            key=lambda c: (
                c.complexity,
                len(repr(c)),  # Shorter parameter representations preferred
            ),
        )
