"""Submission adapter and generator for Kaggle ARC competitions."""

from src.submission.solver_adapter import solve_task, run_solver_with_fallback
from src.submission.submission_generator import generate_submission_dict
from src.submission.validation import validate_submission

__all__ = [
    "solve_task",
    "run_solver_with_fallback",
    "generate_submission_dict",
    "validate_submission",
]
