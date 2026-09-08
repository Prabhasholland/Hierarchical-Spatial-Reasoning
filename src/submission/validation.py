"""Validator for ARC-AGI-2 submission format."""

import json
from pathlib import Path


def validate_grid(grid: list[list[int]]) -> None:
    """Validate that a grid is a valid 2D list of integers 0-9."""
    if not isinstance(grid, list):
        raise ValueError("Grid is not a list")
    
    if len(grid) == 0:
        raise ValueError("Grid has 0 rows")
    if len(grid) > 30:
        raise ValueError(f"Grid has too many rows ({len(grid)} > 30)")
        
    cols = len(grid[0])
    if cols == 0:
        raise ValueError("Grid has 0 columns")
    if cols > 30:
        raise ValueError(f"Grid has too many columns ({cols} > 30)")
        
    for row_idx, row in enumerate(grid):
        if not isinstance(row, list):
            raise ValueError(f"Row {row_idx} is not a list")
        if len(row) != cols:
            raise ValueError(f"Row {row_idx} length {len(row)} doesn't match expected {cols}")
        for col_idx, val in enumerate(row):
            if not isinstance(val, int) or isinstance(val, bool):
                raise ValueError(f"Cell ({row_idx}, {col_idx}) is not an int: {val}")
            if val < 0 or val > 9:
                raise ValueError(f"Cell ({row_idx}, {col_idx}) has invalid value: {val}")


def validate_submission(submission: dict, expected_tasks: dict | None = None) -> bool:
    """
    Validate the complete submission dictionary against ARC-AGI-2 requirements.
    
    Args:
        submission: The parsed submission dictionary.
        expected_tasks: Optional dictionary of expected tasks to verify missing tasks.
        
    Raises:
        ValueError if the submission is invalid.
    """
    if not isinstance(submission, dict):
        raise ValueError("Submission must be a dictionary.")
        
    if expected_tasks is not None:
        missing = set(expected_tasks.keys()) - set(submission.keys())
        if missing:
            raise ValueError(f"Missing predictions for {len(missing)} tasks, e.g. {list(missing)[:5]}")
            
    for task_id, predictions in submission.items():
        if not isinstance(predictions, list):
            raise ValueError(f"Task {task_id} value is not a list of predictions.")
            
        if expected_tasks is not None and task_id in expected_tasks:
            expected_pairs = len(expected_tasks[task_id].test)
            if len(predictions) != expected_pairs:
                raise ValueError(f"Task {task_id} has {len(predictions)} predictions, expected {expected_pairs}.")
                
        for i, pred in enumerate(predictions):
            if not isinstance(pred, dict):
                raise ValueError(f"Task {task_id} prediction {i} is not a dict.")
                
            if set(pred.keys()) != {"attempt_1", "attempt_2"}:
                raise ValueError(f"Task {task_id} prediction {i} has keys {list(pred.keys())}, expected exactly ['attempt_1', 'attempt_2'].")
                
            try:
                validate_grid(pred["attempt_1"])
            except ValueError as e:
                raise ValueError(f"Task {task_id} prediction {i} attempt_1 invalid: {e}")
                
            try:
                validate_grid(pred["attempt_2"])
            except ValueError as e:
                raise ValueError(f"Task {task_id} prediction {i} attempt_2 invalid: {e}")
                
    return True
