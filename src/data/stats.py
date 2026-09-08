"""Dataset statistics calculation utilities for ARC."""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

from src.data.models import ARCTask, COLOR_NAMES


def compute_dataset_statistics(tasks: Sequence[ARCTask] | dict[str, ARCTask]) -> dict[str, Any]:
    """Compute aggregate statistics across a collection of ARC tasks.
    
    Args:
        tasks: Sequence or dict mapping task IDs to ARCTask objects.
        
    Returns:
        Dictionary containing comprehensive dataset metrics.
    """
    if isinstance(tasks, dict):
        task_list = list(tasks.values())
    else:
        task_list = list(tasks)

    total_tasks = len(task_list)
    if total_tasks == 0:
        return {
            "total_tasks": 0,
            "total_train_pairs": 0,
            "total_test_pairs": 0,
        }

    train_pair_counts: Counter[int] = Counter()
    test_pair_counts: Counter[int] = Counter()
    
    grid_heights: list[int] = []
    grid_widths: list[int] = []
    grid_shapes: Counter[tuple[int, int]] = Counter()
    
    color_cell_counts: Counter[int] = Counter()
    color_task_presence: Counter[int] = Counter()
    
    size_preserving_tasks = 0
    fixed_output_dim_tasks = 0

    for task in task_list:
        train_pair_counts[task.num_train] += 1
        test_pair_counts[task.num_test] += 1
        
        if task.is_size_preserving:
            size_preserving_tasks += 1
        if task.has_fixed_output_dim:
            fixed_output_dim_tasks += 1
            
        task_colors = task.unique_colors
        for c in task_colors:
            color_task_presence[c] += 1
            
        for grid in task.all_grids:
            grid_heights.append(grid.height)
            grid_widths.append(grid.width)
            grid_shapes[grid.shape] += 1
            for color, count in grid.color_counts().items():
                color_cell_counts[color] += count

    total_train_pairs = sum(t.num_train for t in task_list)
    total_test_pairs = sum(t.num_test for t in task_list)
    total_cells = sum(color_cell_counts.values())

    color_frequencies = {
        c: {
            "name": COLOR_NAMES.get(c, f"color_{c}"),
            "cell_count": color_cell_counts[c],
            "cell_percentage": (color_cell_counts[c] / total_cells * 100) if total_cells > 0 else 0.0,
            "task_presence_count": color_task_presence[c],
            "task_presence_percentage": (color_task_presence[c] / total_tasks * 100) if total_tasks > 0 else 0.0,
        }
        for c in range(10)
    }

    return {
        "total_tasks": total_tasks,
        "total_train_pairs": total_train_pairs,
        "total_test_pairs": total_test_pairs,
        "avg_train_pairs_per_task": total_train_pairs / total_tasks,
        "avg_test_pairs_per_task": total_test_pairs / total_tasks,
        "train_pair_distribution": dict(sorted(train_pair_counts.items())),
        "test_pair_distribution": dict(sorted(test_pair_counts.items())),
        "size_preserving_task_count": size_preserving_tasks,
        "size_preserving_percentage": (size_preserving_tasks / total_tasks * 100),
        "fixed_output_dim_task_count": fixed_output_dim_tasks,
        "fixed_output_dim_percentage": (fixed_output_dim_tasks / total_tasks * 100),
        "grid_dimension_stats": {
            "min_height": min(grid_heights) if grid_heights else 0,
            "max_height": max(grid_heights) if grid_heights else 0,
            "avg_height": (sum(grid_heights) / len(grid_heights)) if grid_heights else 0.0,
            "min_width": min(grid_widths) if grid_widths else 0,
            "max_width": max(grid_widths) if grid_widths else 0,
            "avg_width": (sum(grid_widths) / len(grid_widths)) if grid_widths else 0.0,
        },
        "top_10_grid_shapes": grid_shapes.most_common(10),
        "color_statistics": color_frequencies,
    }
