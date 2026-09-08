"""Evaluation and benchmarking module for ARC."""

from src.evaluation.harness import (
    TaskEvaluationResult,
    TaskOutcome,
    evaluate_dataset,
    evaluate_single_task,
)
from src.evaluation.metrics import grid_match

__all__ = [
    "grid_match",
    "TaskOutcome",
    "TaskEvaluationResult",
    "evaluate_single_task",
    "evaluate_dataset",
]
