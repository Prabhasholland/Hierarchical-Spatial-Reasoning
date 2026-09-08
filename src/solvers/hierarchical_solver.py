"""RuleBasedHierarchicalSolver_v1 for ARC tasks.

Integrates state representation, state difference analysis, sub-goal generation,
and property-guided A* search across multi-step depth k=1..4 composition spaces.
"""

from __future__ import annotations

from typing import Any

from src.data.models import ARCTask, Grid
from src.reasoning.planner import HierarchicalPlanner
from src.reasoning.state import GridState
from src.reasoning.state_difference import StateDifference
from src.reasoning.subgoals import propose_subgoals
from src.solvers.base import BaseSolver
from src.solvers.candidate_generator import CandidateGenerator
from src.solvers.object_reasoning import ObjectCandidateGenerator
from src.solvers.spatial_reasoning import SpatialCandidateGenerator
from src.solvers.verifier import CandidateRanker, CandidateVerifier
from src.transformations.base import Transformation
from src.transformations.composite import CompositeTransformation
from src.transformations.geometric import IdentityTransformation


class RuleBasedHierarchicalSolver_v1(BaseSolver):
    """Hierarchical Sub-Goal Decomposition & Multi-Step Program Synthesis Solver."""

    def __init__(
        self,
        max_depth: int = 3,
        max_nodes_expanded: int = 500,
        enable_hierarchical_heuristics: bool = True,
        fallback_to_identity: bool = True,
    ) -> None:
        self.max_depth = max_depth
        self.max_nodes_expanded = max_nodes_expanded
        self.enable_hierarchical_heuristics = enable_hierarchical_heuristics
        self.fallback_to_identity = fallback_to_identity

        self.planner = HierarchicalPlanner(
            max_depth=max_depth,
            max_nodes_expanded=max_nodes_expanded,
            enable_hierarchical_heuristics=enable_hierarchical_heuristics,
        )
        self.verifier = CandidateVerifier()
        self.ranker = CandidateRanker()

    def solve_with_details(self, task: ARCTask) -> dict[str, Any]:
        """Solve task using hierarchical sub-goal planning and return rich metadata."""
        # 1. State difference analysis on first demonstration pair
        src_state = GridState.from_grid(task.train[0].input)
        dst_state = GridState.from_grid(task.train[0].output)
        diff = StateDifference.compute(src_state, dst_state)
        subgoals = propose_subgoals(src_state, dst_state)

        # 2. Execute hierarchical planner
        pipeline = self.planner.plan(task)

        if pipeline:
            if len(pipeline) == 1:
                chosen_rule = pipeline[0]
            else:
                chosen_rule = CompositeTransformation(pipeline)
            solved_on_train = True
            rule_name = repr(chosen_rule)
            complexity = chosen_rule.complexity
            successful_depth = len(pipeline)
        else:
            chosen_rule = IdentityTransformation() if self.fallback_to_identity else None
            solved_on_train = False
            rule_name = repr(chosen_rule) if chosen_rule else None
            complexity = chosen_rule.complexity if chosen_rule else None
            successful_depth = 0

        # 3. Generate test predictions
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
            "chosen_rule": rule_name,
            "chosen_rule_complexity": complexity,
            "successful_depth": successful_depth,
            "state_difference": diff.to_dict(),
            "proposed_subgoals": [sg.to_dict() for sg in subgoals],
            "predictions": predictions,
        }

    def solve(self, task: ARCTask) -> list[Grid]:
        """Solve task and return test predictions."""
        details = self.solve_with_details(task)
        return details["predictions"]

    def name(self) -> str:
        h_flag = "Hierarchical" if self.enable_hierarchical_heuristics else "Blind"
        return f"RuleBasedHierarchicalSolver_v1_{h_flag}_D{self.max_depth}"
