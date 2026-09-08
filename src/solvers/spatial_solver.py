"""RuleBasedSpatialObjectSolver_v1 for ARC tasks.

Integrates Grid baseline, Object segmentation DSL, Topology engine, and Spatial raycasting primitives.
"""

from __future__ import annotations

from typing import Any

from src.data.models import ARCTask, Grid
from src.solvers.base import BaseSolver
from src.solvers.candidate_generator import CandidateGenerator
from src.solvers.object_reasoning import ObjectCandidateGenerator
from src.solvers.spatial_reasoning import (
    FillEnclosedTransformation,
    RaycastProjectionTransformation,
    SpatialCandidateGenerator,
    TraceBoundaryTransformation,
)
from src.solvers.verifier import CandidateRanker, CandidateVerifier
from src.transformations.base import Transformation
from src.transformations.geometric import IdentityTransformation


class RuleBasedSpatialObjectSolver_v1(BaseSolver):
    """Full Spatial-Object Solver integrating baseline, object, topology, and raycasting.
    
    Modular solver configured with ablation switches.
    """

    def __init__(
        self,
        max_candidates: int = 1200,
        fallback_to_identity: bool = True,
        enable_baseline_rules: bool = True,
        enable_object_rules: bool = True,
        enable_topology_rules: bool = True,
        enable_raycast_rules: bool = True,
    ) -> None:
        self.max_candidates = max_candidates
        self.fallback_to_identity = fallback_to_identity
        self.enable_baseline_rules = enable_baseline_rules
        self.enable_object_rules = enable_object_rules
        self.enable_topology_rules = enable_topology_rules
        self.enable_raycast_rules = enable_raycast_rules

        self.grid_generator = CandidateGenerator(max_candidates=400)
        self.object_generator = ObjectCandidateGenerator(max_candidates=400)
        self.spatial_generator = SpatialCandidateGenerator(max_candidates=400)
        self.verifier = CandidateVerifier()
        self.ranker = CandidateRanker()

    def generate_all_candidates(self, task: ARCTask) -> list[Transformation]:
        candidates: list[Transformation] = []

        if self.enable_baseline_rules:
            candidates.extend(self.grid_generator.generate(task))

        if self.enable_object_rules:
            candidates.extend(self.object_generator.generate(task))

        if self.enable_topology_rules or self.enable_raycast_rules:
            raw_spatial = self.spatial_generator.generate(task)
            for cand in raw_spatial:
                is_topo = isinstance(cand, (FillEnclosedTransformation, TraceBoundaryTransformation))
                is_ray = isinstance(cand, RaycastProjectionTransformation)

                if is_topo and not self.enable_topology_rules:
                    continue
                if is_ray and not self.enable_raycast_rules:
                    continue
                candidates.append(cand)

        # Deduplicate while preserving priority order
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

    def solve_with_details(self, task: ARCTask) -> dict[str, Any]:
        """Solve task and return predictions alongside rich diagnostic metadata."""
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
        return "RuleBasedSpatialObjectSolver_v1"
