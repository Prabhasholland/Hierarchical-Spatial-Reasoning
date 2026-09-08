"""Object-aware rule-based solver for ARC tasks (RuleBasedObjectSolver_v1).

Combines grid-level baseline transformations with object-centric perception,
relational predicates, and object-level DSL programs.
"""

from __future__ import annotations

from typing import Any

from src.data.models import ARCTask, Grid
from src.solvers.base import BaseSolver
from src.solvers.candidate_generator import CandidateGenerator
from src.solvers.object_reasoning import ObjectCandidateGenerator
from src.solvers.verifier import CandidateRanker, CandidateVerifier
from src.transformations.base import Transformation
from src.transformations.geometric import IdentityTransformation


class RuleBasedObjectSolver_v1(BaseSolver):
    """Object-aware rule-based search solver for ARC tasks.
    
    Generates hypotheses from both baseline grid transformations and
    object-centric DSL operations, strictly verifies on demonstration pairs,
    and predicts unseen test outputs.
    """

    def __init__(
        self,
        max_candidates: int = 1000,
        fallback_to_identity: bool = True,
        enable_baseline_rules: bool = True,
        enable_object_rules: bool = True,
    ) -> None:
        self.max_candidates = max_candidates
        self.fallback_to_identity = fallback_to_identity
        self.enable_baseline_rules = enable_baseline_rules
        self.enable_object_rules = enable_object_rules

        self.grid_generator = CandidateGenerator(max_candidates=max_candidates // 2)
        self.object_generator = ObjectCandidateGenerator(max_candidates=max_candidates // 2)
        self.verifier = CandidateVerifier()
        self.ranker = CandidateRanker()

    def generate_all_candidates(self, task: ARCTask) -> list[Transformation]:
        """Aggregate candidates from enabled generators."""
        candidates: list[Transformation] = []
        if self.enable_grid_rules and self.enable_baseline_rules:
            candidates.extend(self.grid_generator.generate(task))
        if self.enable_object_rules:
            candidates.extend(self.object_generator.generate(task))

        # Deduplicate
        unique: list[Transformation] = []
        seen = set()
        for cand in candidates:
            cand_repr = repr(cand)
            if cand_repr not in seen:
                seen.add(cand_repr)
                unique.append(cand)
                if len(unique) >= self.max_candidates:
                    break
        return unique

    @property
    def enable_grid_rules(self) -> bool:
        return self.enable_baseline_rules

    def solve_with_details(self, task: ARCTask) -> dict[str, Any]:
        """Solve task and return predictions with rich diagnostic metadata."""
        candidates = self.generate_all_candidates(task)
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
        """Solve an ARC task and return predicted output grids."""
        details = self.solve_with_details(task)
        return details["predictions"]

    def name(self) -> str:
        return "RuleBasedObjectSolver_v1"
