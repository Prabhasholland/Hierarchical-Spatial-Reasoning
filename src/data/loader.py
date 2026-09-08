"""ARC dataset loader and data access utilities.

Handles loading, parsing, and validating ARC tasks from individual JSON files
or whole directories.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Sequence

from src.data.models import (
    ARCTask,
    Grid,
    MalformedTaskError,
    TestPair,
    TrainingPair,
)
from src.data.stats import compute_dataset_statistics


def parse_grid(raw_grid: Any) -> Grid:
    """Parse raw nested list data into a validated Grid instance."""
    if not isinstance(raw_grid, (list, tuple)):
        raise MalformedTaskError(f"Grid must be a list of lists, got {type(raw_grid)}")
    return Grid.from_list(raw_grid)


def load_task_from_dict(
    data: dict[str, Any],
    task_id: str = "unknown",
    metadata: dict[str, Any] | None = None,
) -> ARCTask:
    """Parse a task dictionary into an ARCTask model.
    
    Args:
        data: Dictionary containing 'train' and 'test' keys.
        task_id: Identifier string for the task.
        metadata: Optional metadata dictionary.
        
    Returns:
        Validated ARCTask instance.
    """
    if not isinstance(data, dict):
        raise MalformedTaskError(f"Task data must be a JSON dict, got {type(data)}")
    
    if "train" not in data:
        raise MalformedTaskError(f"Task '{task_id}' is missing required 'train' field.")
    if "test" not in data:
        raise MalformedTaskError(f"Task '{task_id}' is missing required 'test' field.")
    
    train_data = data["train"]
    test_data = data["test"]
    
    if not isinstance(train_data, list):
        raise MalformedTaskError(f"Task '{task_id}' 'train' must be a list, got {type(train_data)}")
    if not isinstance(test_data, list):
        raise MalformedTaskError(f"Task '{task_id}' 'test' must be a list, got {type(test_data)}")
    
    train_pairs: list[TrainingPair] = []
    for idx, pair in enumerate(train_data):
        if not isinstance(pair, dict):
            raise MalformedTaskError(
                f"Task '{task_id}' train[{idx}] must be a dict with 'input' and 'output'"
            )
        if "input" not in pair or "output" not in pair:
            raise MalformedTaskError(
                f"Task '{task_id}' train[{idx}] missing 'input' or 'output' keys"
            )
        in_grid = parse_grid(pair["input"])
        out_grid = parse_grid(pair["output"])
        train_pairs.append(TrainingPair(input=in_grid, output=out_grid))
        
    test_pairs: list[TestPair] = []
    for idx, pair in enumerate(test_data):
        if not isinstance(pair, dict):
            raise MalformedTaskError(
                f"Task '{task_id}' test[{idx}] must be a dict with 'input' (and optional 'output')"
            )
        if "input" not in pair:
            raise MalformedTaskError(
                f"Task '{task_id}' test[{idx}] missing required 'input' key"
            )
        in_grid = parse_grid(pair["input"])
        out_grid = parse_grid(pair["output"]) if "output" in pair else None
        test_pairs.append(TestPair(input=in_grid, output=out_grid))
        
    return ARCTask(
        task_id=task_id,
        train=train_pairs,
        test=test_pairs,
        metadata=metadata or {},
    )


def load_task(task_path: str | Path) -> ARCTask:
    """Load and validate a single ARC task from a JSON file path.
    
    Args:
        task_path: Path to the task JSON file.
        
    Returns:
        Validated ARCTask instance.
    """
    path = Path(task_path)
    if not path.exists():
        raise FileNotFoundError(f"Task file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    task_id = path.stem
    return load_task_from_dict(data, task_id=task_id, metadata={"file_path": str(path)})


def load_dataset(
    dataset_dir: str | Path,
    pattern: str = "*.json",
    recursive: bool = False,
) -> dict[str, ARCTask]:
    """Load all ARC tasks from a directory.
    
    Args:
        dataset_dir: Directory containing task JSON files.
        pattern: File glob pattern (defaults to '*.json').
        recursive: Whether to search subdirectories recursively.
        
    Returns:
        Dictionary mapping task_id to ARCTask instances.
    """
    dir_path = Path(dataset_dir)
    if not dir_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dir_path}")
    
    files = dir_path.rglob(pattern) if recursive else dir_path.glob(pattern)
    tasks: dict[str, ARCTask] = {}
    
    for file_path in sorted(files):
        try:
            task = load_task(file_path)
            tasks[task.task_id] = task
        except Exception as e:
            print(f"Warning: Failed to load task from {file_path}: {e}")
            
    return tasks


def list_task_ids(dataset_dir: str | Path, recursive: bool = False) -> list[str]:
    """List all task IDs present in a dataset directory without loading full tasks."""
    dir_path = Path(dataset_dir)
    if not dir_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dir_path}")
    
    files = dir_path.rglob("*.json") if recursive else dir_path.glob("*.json")
    return sorted(f.stem for f in files)


def get_random_task(
    tasks_or_dir: dict[str, ARCTask] | Sequence[ARCTask] | str | Path,
    seed: int | None = None,
) -> ARCTask:
    """Select a random task from a loaded dataset or directory path."""
    if seed is not None:
        random.seed(seed)
        
    if isinstance(tasks_or_dir, (str, Path)):
        task_ids = list_task_ids(tasks_or_dir)
        if not task_ids:
            raise ValueError(f"No task JSON files found in {tasks_or_dir}")
        chosen_id = random.choice(task_ids)
        task_path = Path(tasks_or_dir) / f"{chosen_id}.json"
        return load_task(task_path)
    elif isinstance(tasks_or_dir, dict):
        if not tasks_or_dir:
            raise ValueError("Task dictionary is empty.")
        return random.choice(list(tasks_or_dir.values()))
    elif isinstance(tasks_or_dir, Sequence):
        if not tasks_or_dir:
            raise ValueError("Task sequence is empty.")
        return random.choice(tasks_or_dir)
    else:
        raise TypeError(f"Unsupported type for tasks_or_dir: {type(tasks_or_dir)}")


def get_dataset_statistics(tasks: Sequence[ARCTask] | dict[str, ARCTask]) -> dict[str, Any]:
    """Compute aggregate dataset statistics."""
    return compute_dataset_statistics(tasks)
