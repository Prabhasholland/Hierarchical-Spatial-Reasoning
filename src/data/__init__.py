"""ARC data loading, validation, representation, and visualization."""

from src.data.loader import (
    get_dataset_statistics,
    get_random_task,
    list_task_ids,
    load_dataset,
    load_task,
    load_task_from_dict,
)
from src.data.models import (
    ARC_COLORS,
    COLOR_NAMES,
    ARCTask,
    ARCValidationError,
    Grid,
    InvalidGridError,
    MalformedTaskError,
    TestPair,
    TrainingPair,
)
from src.data.stats import compute_dataset_statistics
from src.data.visualizer import (
    ARC_CMAP,
    ARC_NORM,
    get_arc_colormap,
    plot_grid,
    plot_task,
    render_task_ascii,
)

__all__ = [
    "ARC_COLORS",
    "COLOR_NAMES",
    "ARC_CMAP",
    "ARC_NORM",
    "ARCTask",
    "ARCValidationError",
    "Grid",
    "InvalidGridError",
    "MalformedTaskError",
    "TestPair",
    "TrainingPair",
    "load_task",
    "load_task_from_dict",
    "load_dataset",
    "list_task_ids",
    "get_random_task",
    "get_dataset_statistics",
    "compute_dataset_statistics",
    "get_arc_colormap",
    "plot_grid",
    "plot_task",
    "render_task_ascii",
]
