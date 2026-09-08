"""Generates the full submission dictionary across all tasks."""

import time
from typing import Any

from src.data.models import ARCTask
from src.submission.solver_adapter import run_solver_with_fallback
from src.solvers.hierarchical_solver import RuleBasedHierarchicalSolver_v1


def generate_submission_dict(tasks: dict[str, ARCTask]) -> tuple[dict[str, list[dict[str, list[list[int]]]]], dict[str, Any]]:
    """
    Run solver on a set of tasks and compile the submission dictionary.
    
    Args:
        tasks: Dictionary mapping task_id to ARCTask.
        
    Returns:
        A tuple of (submission_dict, report_dict).
    """
    submission = {}
    report = {
        "total_tasks": len(tasks),
        "successful_tasks": 0,
        "failed_tasks": 0,
        "exceptions": [],
        "runtime_seconds": 0.0,
        "fallback_count": 0
    }
    
    start_time = time.time()
    
    # Initialize one solver instance to avoid overhead if applicable, 
    # though it doesn't store state between runs.
    solver = RuleBasedHierarchicalSolver_v1(
        max_depth=4,
        max_nodes_expanded=250,
        enable_hierarchical_heuristics=True
    )
    
    for task_id, task in tasks.items():
        predictions, stats = run_solver_with_fallback(task, solver=solver)
        submission[task_id] = predictions
        
        if stats["status"] == "success":
            report["successful_tasks"] += 1
        else:
            report["failed_tasks"] += 1
            report["fallback_count"] += 1
            report["exceptions"].append({
                "task_id": task_id,
                "error": stats["error"]
            })
            
    report["runtime_seconds"] = round(time.time() - start_time, 2)
    
    return submission, report
