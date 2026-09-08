"""Rule-based search solver for ARC tasks (Baseline v1).

Implements the generate -> verify -> rank -> predict pipeline.
"""

from __future__ import annotations

from typing import Any

from src.data.models import ARCTask, Grid
from src.solvers.base import BaseSolver
from src.solvers.candidate_generator import CandidateGenerator
from src.solvers.verifier import CandidateRanker, CandidateVerifier
from src.transformations.geometric import IdentityTransformation


class RuleBasedSearchSolver(BaseSolver):
    """Baseline rule-based search solver for ARC.
    
    Pipeline:
    1. Generate parameter-instantiated candidate transformations from task demonstrations.
    2. Verifies each candidate against all training pairs (exact pixel-match).
    3. Ranks consistent candidates by complexity / Occam's razor.
    4. Applies top-ranked candidate to test inputs to produce predictions.
    """

    def __init__(
        self,
        max_candidates: int = 500,
        fallback_to_identity: bool = True,
    ) -> None:
        self.generator = CandidateGenerator(max_candidates=max_candidates)
        self.verifier = CandidateVerifier()
        self.ranker = CandidateRanker()
        self.fallback_to_identity = fallback_to_identity

    def solve_with_details(self, task: ARCTask) -> dict[str, Any]:
        """Solve task and return predictions along with detailed search metadata."""
        candidates = self.generator.generate(task)
        consistent = self.verifier.find_all_consistent(candidates, task)
        ranked = self.ranker.rank(consistent, task)

        if ranked:
            chosen_rule = ranked[0]
            solved_on_train = True
        else:
            chosen_rule = IdentityTransformation() if self.fallback_to_identity else None
            solved_on_train = False

        predictions: list[Grid] = []
        for test_pair in task.test:
            if chosen_rule is not None:
                try:
                    pred = chosen_rule.apply(test_pair.input)
                except Exception:
                    pred = test_pair.input
            else:
                pred = test_pair.input
            predictions.append(pred)

        return {
            "task_id": task.task_id,
            "solved_on_train": solved_on_train,
            "chosen_rule": repr(chosen_rule) if chosen_rule else None,
            "chosen_rule_complexity": chosen_rule.complexity if chosen_rule else None,
            "num_candidates_generated": len(candidates),
            "num_consistent_candidates": len(consistent),
            "consistent_rules": [repr(c) for c in ranked],
            "predictions": predictions,
        }

    def solve(self, task: ARCTask) -> list[Grid]:
        """Solve an ARC task and return predicted output grids for all test inputs."""
        details = self.solve_with_details(task)
        return details["predictions"]

    def name(self) -> str:
        return "RuleBasedSearchSolver_v1"
