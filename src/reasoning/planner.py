"""Hierarchical Planner for ARC sub-goal decomposition and multi-step search.

Implements property-guided A* search across depth k=1..4 composition spaces.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any, Sequence

from src.data.models import ARCTask, Grid, TrainingPair
from src.reasoning.state import GridState
from src.reasoning.state_difference import StateDifference
from src.reasoning.subgoals import SubGoal, propose_subgoals
from src.reasoning.verifier import IntermediateVerifier
from src.solvers.candidate_generator import CandidateGenerator
from src.solvers.object_reasoning import ObjectCandidateGenerator
from src.solvers.spatial_reasoning import SpatialCandidateGenerator
from src.transformations.base import Transformation
from src.transformations.composite import CompositeTransformation


@dataclass
class SearchNode:
    """Node in A* property-guided search tree."""
    state_distance: float
    depth: int
    current_grids: list[Grid]
    pipeline: list[Transformation]
    subgoal_history: list[str]

    def __lt__(self, other: SearchNode) -> bool:
        # A* priority: f(n) = depth + heuristic_distance
        return (self.depth + self.state_distance) < (other.depth + other.state_distance)


def compute_state_distance(current_states: list[GridState], target_states: list[GridState]) -> float:
    """Compute heuristic distance between current training grid states and target states."""
    total_dist = 0.0
    for cur, tgt in zip(current_states, target_states):
        d_dim = abs(cur.height - tgt.height) + abs(cur.width - tgt.width)
        d_colors = abs(cur.num_colors - tgt.num_colors)
        d_fg = abs(cur.foreground_cell_count - tgt.foreground_cell_count) / 10.0
        d_enc = abs(cur.num_enclosed_regions - tgt.num_enclosed_regions)
        d_sym = abs(cur.horizontal_symmetry - tgt.horizontal_symmetry) + abs(cur.vertical_symmetry - tgt.vertical_symmetry)
        total_dist += d_dim * 5.0 + d_colors * 2.0 + d_fg + d_enc * 3.0 + d_sym * 2.0
    return total_dist / len(current_states)


class HierarchicalPlanner:
    """Property-guided hierarchical planner for multi-step program synthesis."""

    def __init__(
        self,
        max_depth: int = 3,
        max_nodes_expanded: int = 500,
        enable_hierarchical_heuristics: bool = True,
    ) -> None:
        self.max_depth = max_depth
        self.max_nodes_expanded = max_nodes_expanded
        self.enable_hierarchical_heuristics = enable_hierarchical_heuristics

        self.verifier = IntermediateVerifier()
        self.grid_gen = CandidateGenerator(max_candidates=150)
        self.obj_gen = ObjectCandidateGenerator(max_candidates=150)
        self.spatial_gen = SpatialCandidateGenerator(max_candidates=150)

    def _get_primitive_library(self, task: ARCTask) -> list[Transformation]:
        """Aggregate primitive candidates across generators."""
        primitives: list[Transformation] = []
        primitives.extend(self.grid_gen.generate(task))
        primitives.extend(self.obj_gen.generate(task))
        primitives.extend(self.spatial_gen.generate(task))

        unique: list[Transformation] = []
        seen = set()
        for p in primitives:
            p_repr = repr(p)
            if p_repr not in seen:
                seen.add(p_repr)
                unique.append(p)
        return unique

    def plan(self, task: ARCTask) -> list[Transformation] | None:
        """Search for a sequence of transformations T1..Tk explaining all train demonstration pairs."""
        train_inputs = task.all_train_inputs
        train_outputs = task.all_train_outputs
        target_states = [GridState.from_grid(g) for g in train_outputs]

        primitives = self._get_primitive_library(task)
        initial_distance = compute_state_distance(
            [GridState.from_grid(g) for g in train_inputs], target_states
        )

        start_node = SearchNode(
            state_distance=initial_distance,
            depth=0,
            current_grids=train_inputs,
            pipeline=[],
            subgoal_history=[],
        )

        frontier: list[SearchNode] = [start_node]
        nodes_expanded = 0

        while frontier and nodes_expanded < self.max_nodes_expanded:
            curr_node = heapq.heappop(frontier) if self.enable_hierarchical_heuristics else frontier.pop(0)
            nodes_expanded += 1

            # Check exact match on all train pairs
            if curr_node.pipeline:
                if self.verifier.verify_exact_match(curr_node.pipeline, task):
                    return curr_node.pipeline

            # Stop expansion if depth limit reached
            if curr_node.depth >= self.max_depth:
                continue

            # Branch expansion
            for prim in primitives:
                try:
                    next_grids = [prim.apply(g) for g in curr_node.current_grids]
                except Exception:
                    continue

                # Temporary task with next_grids as inputs
                temp_train = [TrainingPair(input=ng, output=out) for ng, out in zip(next_grids, train_outputs)]
                temp_task = ARCTask(task_id=task.task_id, train=temp_train, test=task.test)

                # Intermediate verification
                if not self.verifier.verify_step_progress(prim, temp_task):
                    continue

                next_states = [GridState.from_grid(g) for g in next_grids]
                dist = compute_state_distance(next_states, target_states)

                new_pipeline = list(curr_node.pipeline) + [prim]

                # Exact match check after step
                if dist == 0.0:
                    if self.verifier.verify_exact_match(new_pipeline, task):
                        return new_pipeline

                # Prune if distance increased significantly and heuristics enabled
                if self.enable_hierarchical_heuristics and curr_node.depth > 0:
                    if dist > curr_node.state_distance + 15.0:
                        continue

                next_node = SearchNode(
                    state_distance=dist,
                    depth=curr_node.depth + 1,
                    current_grids=next_grids,
                    pipeline=new_pipeline,
                    subgoal_history=curr_node.subgoal_history + [prim.name],
                )

                if self.enable_hierarchical_heuristics:
                    heapq.heappush(frontier, next_node)
                else:
                    frontier.append(next_node)

        return None
