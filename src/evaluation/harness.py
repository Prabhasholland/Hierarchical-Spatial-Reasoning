"""Evaluation framework and benchmark harness for ARC solvers.

Ensures strict evaluation integrity (zero leakage of test outputs to solvers),
computes task-level exact match accuracy, runtime, search efficiency,
and detailed failure taxonomy.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from src.data.models import ARCTask, Grid
from src.evaluation.metrics import grid_match


class TaskOutcome(str, Enum):
    """Categorization of solver performance on an ARC task."""
    SOLVED = "SOLVED"
    GENERALIZATION_ERROR = "GENERALIZATION_ERROR"
    UNEXPLAINED_SIZE_CHANGE = "UNEXPLAINED_SIZE_CHANGE"
    UNSOLVED_SIZE_PRESERVING = "UNSOLVED_SIZE_PRESERVING"
    NO_GROUND_TRUTH = "NO_GROUND_TRUTH"
    EXECUTION_ERROR = "EXECUTION_ERROR"


@dataclass
class TaskEvaluationResult:
    """Detailed evaluation result for a single task."""
    task_id: str
    outcome: TaskOutcome
    is_task_solved: bool
    num_test_pairs: int
    test_pairs_solved: int
    solved_on_train: bool
    chosen_rule: str | None
    num_candidates_generated: int
    num_consistent_candidates: int
    runtime_ms: float
    error_message: str | None = None
    predicted_shapes: list[tuple[int, int]] = field(default_factory=list)
    expected_shapes: list[tuple[int, int]] = field(default_factory=list)


def evaluate_single_task(
    solver: Any,
    task: ARCTask,
) -> TaskEvaluationResult:
    """Evaluate a solver on a single task without leaking ground-truth test outputs."""
    start_time = time.perf_counter()
    error_msg = None

    try:
        if hasattr(solver, "solve_with_details"):
            details = solver.solve_with_details(task)
            predictions = details.get("predictions", [])
            solved_on_train = details.get("solved_on_train", False)
            chosen_rule = details.get("chosen_rule")
            num_cand = details.get("num_candidates_generated", 0)
            num_cons = details.get("num_consistent_candidates", 0)
        else:
            predictions = solver.solve(task)
            solved_on_train = False
            chosen_rule = None
            num_cand = 0
            num_cons = 0
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return TaskEvaluationResult(
            task_id=task.task_id,
            outcome=TaskOutcome.EXECUTION_ERROR,
            is_task_solved=False,
            num_test_pairs=task.num_test,
            test_pairs_solved=0,
            solved_on_train=False,
            chosen_rule=None,
            num_candidates_generated=0,
            num_consistent_candidates=0,
            runtime_ms=elapsed_ms,
            error_message=str(e),
        )

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    test_pairs_solved = 0
    has_all_gt = True

    pred_shapes = [p.shape if isinstance(p, Grid) else (len(p), len(p[0])) for p in predictions]
    exp_shapes: list[tuple[int, int]] = []

    for t_idx, test_pair in enumerate(task.test):
        if test_pair.output is None:
            has_all_gt = False
            continue

        exp_shapes.append(test_pair.output.shape)
        if t_idx < len(predictions):
            pred = predictions[t_idx]
            pred_list = pred.to_list() if isinstance(pred, Grid) else pred
            if grid_match(pred_list, test_pair.output.to_list()):
                test_pairs_solved += 1

    is_task_solved = has_all_gt and (test_pairs_solved == task.num_test)

    # Determine failure category
    if is_task_solved:
        outcome = TaskOutcome.SOLVED
    elif not has_all_gt:
        outcome = TaskOutcome.NO_GROUND_TRUTH
    elif solved_on_train:
        # Rule fit training data perfectly, but failed on test
        outcome = TaskOutcome.GENERALIZATION_ERROR
    else:
        # No rule found on train
        if not task.is_size_preserving:
            outcome = TaskOutcome.UNEXPLAINED_SIZE_CHANGE
        else:
            outcome = TaskOutcome.UNSOLVED_SIZE_PRESERVING

    return TaskEvaluationResult(
        task_id=task.task_id,
        outcome=outcome,
        is_task_solved=is_task_solved,
        num_test_pairs=task.num_test,
        test_pairs_solved=test_pairs_solved,
        solved_on_train=solved_on_train,
        chosen_rule=chosen_rule,
        num_candidates_generated=num_cand,
        num_consistent_candidates=num_cons,
        runtime_ms=elapsed_ms,
        error_message=error_msg,
        predicted_shapes=pred_shapes,
        expected_shapes=exp_shapes,
    )


def evaluate_dataset(
    solver: Any,
    tasks: Sequence[ARCTask] | dict[str, ARCTask],
    max_tasks: int | None = None,
) -> dict[str, Any]:
    """Evaluate a solver on a full ARC dataset and compute aggregate benchmarks."""
    task_list = list(tasks.values()) if isinstance(tasks, dict) else list(tasks)
    if max_tasks is not None:
        task_list = task_list[:max_tasks]

    total_tasks = len(task_list)
    results: list[TaskEvaluationResult] = []
    
    total_start = time.perf_counter()
    for task in task_list:
        res = evaluate_single_task(solver, task)
        results.append(res)
    total_runtime_sec = time.perf_counter() - total_start

    eval_with_gt = [r for r in results if r.outcome != TaskOutcome.NO_GROUND_TRUTH]
    num_with_gt = len(eval_with_gt)
    solved_tasks = [r for r in eval_with_gt if r.is_task_solved]
    num_solved = len(solved_tasks)

    total_test_pairs = sum(r.num_test_pairs for r in eval_with_gt)
    solved_test_pairs = sum(r.test_pairs_solved for r in eval_with_gt)
    train_consistent_tasks = sum(1 for r in eval_with_gt if r.solved_on_train)

    task_acc = (num_solved / num_with_gt * 100) if num_with_gt > 0 else 0.0
    pair_acc = (solved_test_pairs / total_test_pairs * 100) if total_test_pairs > 0 else 0.0
    train_acc = (train_consistent_tasks / num_with_gt * 100) if num_with_gt > 0 else 0.0

    outcome_counts = Counter(r.outcome.value for r in results)

    return {
        "solver_name": getattr(solver, "name", lambda: solver.__class__.__name__)(),
        "total_tasks_evaluated": total_tasks,
        "tasks_with_ground_truth": num_with_gt,
        "tasks_solved": num_solved,
        "task_level_accuracy_pct": task_acc,
        "total_test_pairs": total_test_pairs,
        "test_pairs_solved": solved_test_pairs,
        "test_pair_accuracy_pct": pair_acc,
        "train_consistency_count": train_consistent_tasks,
        "train_consistency_pct": train_acc,
        "generalization_gap_pct": train_acc - task_acc,
        "total_runtime_seconds": total_runtime_sec,
        "avg_runtime_ms_per_task": (total_runtime_sec / total_tasks * 1000.0) if total_tasks > 0 else 0.0,
        "avg_candidates_generated": (sum(r.num_candidates_generated for r in results) / total_tasks) if total_tasks > 0 else 0.0,
        "avg_consistent_candidates": (sum(r.num_consistent_candidates for r in results) / total_tasks) if total_tasks > 0 else 0.0,
        "failure_breakdown": dict(outcome_counts),
        "solved_task_ids": [r.task_id for r in solved_tasks],
        "solved_task_rules": {r.task_id: r.chosen_rule for r in solved_tasks},
        "per_task_results": [
            {
                "task_id": r.task_id,
                "outcome": r.outcome.value,
                "is_solved": r.is_task_solved,
                "solved_on_train": r.solved_on_train,
                "chosen_rule": r.chosen_rule,
                "runtime_ms": round(r.runtime_ms, 2),
            }
            for r in results
        ],
    }
