"""Adapter bridging the core solver logic to the Kaggle submission format."""

import traceback
from typing import Any

from src.data.models import ARCTask, Grid
from src.solvers.hierarchical_solver import RuleBasedHierarchicalSolver_v1


def solve_task(task: ARCTask, solver=None) -> list[dict[str, list[list[int]]]]:
    """
    Solve a single ARC task and format predictions for the official Kaggle submission format.
    
    Args:
        task: ARCTask instance containing train and test pairs.
        solver: The solver to use. If None, instantiates the default best solver.
        
    Returns:
        List of dictionaries containing attempt_1 and attempt_2.
        One dictionary for each test pair in the task.
    """
    if solver is None:
        solver = RuleBasedHierarchicalSolver_v1(
            max_depth=4,
            max_nodes_expanded=250,
            enable_hierarchical_heuristics=True
        )

    try:
        predictions = solver.solve(task)
        
        # Ensure we have exactly as many predictions as test pairs
        if len(predictions) != len(task.test):
            raise ValueError(f"Solver returned {len(predictions)} predictions, expected {len(task.test)}")
            
        formatted_predictions = []
        for pred in predictions:
            pred_list = pred.to_list()
            # The Kaggle format requires attempt_1 and attempt_2
            # Since our solver only generates one top prediction, we duplicate it.
            formatted_predictions.append({
                "attempt_1": pred_list,
                "attempt_2": pred_list
            })
            
        return formatted_predictions
        
    except Exception:
        # Re-raise to be handled by fallback logic
        raise


def run_solver_with_fallback(task: ARCTask, solver=None) -> tuple[list[dict[str, list[list[int]]]], dict[str, Any]]:
    """
    Run solver with a guaranteed valid fallback output upon failure.
    Returns the predictions and a stats dictionary.
    """
    stats = {
        "task_id": task.task_id,
        "status": "success",
        "error": None
    }
    
    try:
        predictions = solve_task(task, solver=solver)
        return predictions, stats
    except Exception as e:
        stats["status"] = "fallback"
        stats["error"] = str(e)
        
        # Generate valid fallback: Just copy the test input for each test pair.
        fallback_predictions = []
        for test_pair in task.test:
            fallback_grid = test_pair.input.to_list()
            fallback_predictions.append({
                "attempt_1": fallback_grid,
                "attempt_2": fallback_grid
            })
        return fallback_predictions, stats
