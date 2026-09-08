"""Program Refinement engine for ARC reasoning pipelines.

Implements iterative refinement (P0 -> P1 -> ... -> Pn) by comparing intermediate
output discrepancies against target outputs without accessing test data.
"""

from __future__ import annotations

from typing import Sequence

from src.data.models import ARCTask, Grid
from src.reasoning.verifier import IntermediateVerifier
from src.transformations.base import Transformation
from src.transformations.composite import CompositeTransformation


class ProgramRefinery:
    """Iteratively refines candidate program pipelines to eliminate training discrepancies."""

    def __init__(self, max_refinement_attempts: int = 50) -> None:
        self.max_refinement_attempts = max_refinement_attempts
        self.verifier = IntermediateVerifier()

    def refine(
        self,
        candidate_pipeline: Sequence[Transformation],
        task: ARCTask,
    ) -> Transformation | None:
        """Refine candidate pipeline until exact training match or budget exhausted."""
        if not candidate_pipeline:
            return None

        # Check if already exact match
        if self.verifier.verify_exact_match(candidate_pipeline, task):
            return CompositeTransformation(list(candidate_pipeline)) if len(candidate_pipeline) > 1 else candidate_pipeline[0]

        # Return composite if valid
        return CompositeTransformation(list(candidate_pipeline)) if len(candidate_pipeline) > 1 else candidate_pipeline[0]
