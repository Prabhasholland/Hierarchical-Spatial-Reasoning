"""Intermediate Verifier for hierarchical ARC planning.

Verifies intermediate transformation steps across training demonstration pairs,
ensuring progress towards sub-goals without introducing contradictions.
"""

from __future__ import annotations

from typing import Any, Sequence

from src.data.models import ARCTask, Grid
from src.evaluation.metrics import grid_match
from src.reasoning.state import GridState
from src.reasoning.subgoals import SubGoal
from src.transformations.base import Transformation


class IntermediateVerifier:
    """Verifies intermediate execution steps across all training demonstration pairs."""

    def verify_step_progress(
        self,
        transformation: Transformation,
        task: ARCTask,
        subgoal: SubGoal | None = None,
    ) -> bool:
        """Check if applying transformation to all train inputs yields valid intermediate grids.
        
        Args:
            transformation: The candidate step transformation.
            task: The ARCTask containing training pairs.
            subgoal: Optional SubGoal specifying verification criteria.
            
        Returns:
            True if step executes cleanly without exceptions and makes valid progress across ALL train inputs.
        """
        for pair in task.train:
            try:
                out_grid = transformation.apply(pair.input)
                if not isinstance(out_grid, Grid):
                    return False

                # Basic validity checks
                if out_grid.height < 1 or out_grid.width < 1:
                    return False

                # Subgoal criterion check if specified
                if subgoal and subgoal.verification_criteria:
                    st = GridState.from_grid(out_grid)
                    crit = subgoal.verification_criteria
                    if "height" in crit and st.height != crit["height"]:
                        return False
                    if "width" in crit and st.width != crit["width"]:
                        return False
                    if "max_enclosed" in crit and st.num_enclosed_regions > crit["max_enclosed"]:
                        return False

            except Exception:
                return False

        return True

    def verify_exact_match(
        self,
        transformation_pipeline: Sequence[Transformation],
        task: ARCTask,
    ) -> bool:
        """Check if pipeline produces exact pixel match on ALL training output pairs."""
        for pair in task.train:
            curr_grid = pair.input
            try:
                for t in transformation_pipeline:
                    curr_grid = t.apply(curr_grid)
                if not grid_match(curr_grid.to_list(), pair.output.to_list()):
                    return False
            except Exception:
                return False
        return True
