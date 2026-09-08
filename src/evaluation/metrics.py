"""Evaluation metrics for ARC tasks.

The primary metric for ARC is exact match accuracy:
a task is considered solved only if the predicted output grid
is pixel-perfect identical to the ground truth.
"""

from typing import Any


Grid = list[list[int]]


def grid_match(predicted: Grid, expected: Grid) -> bool:
    """Check if two grids are exactly identical.
    
    Args:
        predicted: The predicted output grid.
        expected: The expected output grid.
        
    Returns:
        True if grids are identical in dimensions and values.
    """
    if len(predicted) != len(expected):
        return False
    for pred_row, exp_row in zip(predicted, expected):
        if len(pred_row) != len(exp_row):
            return False
        if pred_row != exp_row:
            return False
    return True


def evaluate_solver(
    solver,
    tasks: dict[str, dict[str, Any]],
    max_attempts: int = 2,
) -> dict[str, Any]:
    """Evaluate a solver on a set of ARC tasks.
    
    Args:
        solver: An ARC solver implementing the BaseSolver interface.
        tasks: Dictionary mapping task IDs to task data.
        max_attempts: Maximum number of solution attempts per task.
        
    Returns:
        Dictionary with evaluation results including accuracy,
        per-task results, and timing information.
    """
    # TODO: Implement evaluation loop
    # - Run solver on each task
    # - Compare predictions to ground truth
    # - Track solve rate, timing, etc.
    raise NotImplementedError("Evaluation loop not yet implemented")
