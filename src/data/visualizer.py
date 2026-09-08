"""ARC Grid and Task Visualizer using Matplotlib and ASCII rendering.

Provides clear visual plots for single grids, train input-output pairs side-by-side,
test inputs/outputs, and full ARC tasks with official colormaps and dimension labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm

from src.data.models import (
    ARC_COLORS,
    ARCTask,
    Grid,
    TestPair,
    TrainingPair,
)


# Official ARC hex color list in order 0-9
ARC_COLOR_LIST = [ARC_COLORS[i] for i in range(10)]
ARC_CMAP = ListedColormap(ARC_COLOR_LIST, name="ARC_Colormap")
ARC_NORM = BoundaryNorm(boundaries=np.arange(-0.5, 10.5, 1), ncolors=10)


def get_arc_colormap() -> tuple[ListedColormap, BoundaryNorm]:
    """Return standard ARC ListedColormap and BoundaryNorm."""
    return ARC_CMAP, ARC_NORM


def plot_grid(
    grid: Grid | Sequence[Sequence[int]] | np.ndarray,
    title: str | None = None,
    ax: plt.Axes | None = None,
    show_grid_lines: bool = True,
    show_dims: bool = True,
    cell_size: float = 0.4,
) -> plt.Axes:
    """Plot a single ARC grid.
    
    Args:
        grid: Grid instance or 2D array/list of integers.
        title: Optional title string above the grid.
        ax: Matplotlib axes to draw on. If None, creates a new figure.
        show_grid_lines: Whether to draw cell grid borders.
        show_dims: Whether to include (H, W) in title or labels.
        cell_size: Visual scale factor for figure creation when ax is None.
        
    Returns:
        The matplotlib Axes object.
    """
    if isinstance(grid, Grid):
        arr = grid.to_numpy()
        h, w = grid.height, grid.width
    elif isinstance(grid, np.ndarray):
        arr = grid
        h, w = arr.shape
    else:
        grid_obj = Grid.from_list(grid)
        arr = grid_obj.to_numpy()
        h, w = grid_obj.height, grid_obj.width

    if ax is None:
        fig_w = max(2.5, w * cell_size)
        fig_h = max(2.5, h * cell_size)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)

    ax.imshow(arr, cmap=ARC_CMAP, norm=ARC_NORM, interpolation="nearest")

    if show_grid_lines:
        ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
        ax.grid(which="minor", color="#555555", linestyle="-", linewidth=0.75)
        ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_xticks([])
    ax.set_yticks([])

    dim_str = f" ({h}×{w})" if show_dims else ""
    full_title = f"{title}{dim_str}" if title else (f"{h}×{w}" if show_dims else "")
    if full_title:
        ax.set_title(full_title, fontsize=9, fontweight="bold", pad=4)

    return ax


def plot_task(
    task: ARCTask,
    show_task_id: bool = True,
    show_dims: bool = True,
    save_path: str | Path | None = None,
    dpi: int = 120,
) -> plt.Figure:
    """Plot an entire ARC task showing train pairs and test pairs side-by-side.
    
    Layout:
    Row i: Train Pair i [Input | Output]
    Row k: Test Pair j [Test Input | Test Output (or Prediction)]
    
    Args:
        task: ARCTask instance to visualize.
        show_task_id: Whether to display task ID as suptitle.
        show_dims: Whether to include grid dimensions in sub-titles.
        save_path: Optional path to save the generated figure.
        dpi: DPI resolution for output plot.
        
    Returns:
        The matplotlib Figure object.
    """
    num_train = task.num_train
    num_test = task.num_test
    total_rows = num_train + num_test
    
    # 2 columns: Input (left) and Output (right)
    fig, axes = plt.subplots(
        nrows=total_rows,
        ncols=2,
        figsize=(7, max(4, total_rows * 2.8)),
        dpi=dpi,
        squeeze=False,
    )
    
    # Render training pairs
    for r_idx, pair in enumerate(task.train):
        plot_grid(
            pair.input,
            title=f"Train {r_idx + 1} Input",
            ax=axes[r_idx, 0],
            show_dims=show_dims,
        )
        plot_grid(
            pair.output,
            title=f"Train {r_idx + 1} Output",
            ax=axes[r_idx, 1],
            show_dims=show_dims,
        )

    # Render test pairs
    for t_idx, pair in enumerate(task.test):
        row = num_train + t_idx
        plot_grid(
            pair.input,
            title=f"Test {t_idx + 1} Input",
            ax=axes[row, 0],
            show_dims=show_dims,
        )
        if pair.output is not None:
            plot_grid(
                pair.output,
                title=f"Test {t_idx + 1} Output (Ground Truth)",
                ax=axes[row, 1],
                show_dims=show_dims,
            )
        else:
            axes[row, 1].text(
                0.5,
                0.5,
                "Test Output\n(Hidden / Unsolved)",
                ha="center",
                va="center",
                fontsize=10,
                color="#666666",
                style="italic",
            )
            axes[row, 1].set_xticks([])
            axes[row, 1].set_yticks([])
            axes[row, 1].set_title(f"Test {t_idx + 1} Output", fontsize=9, pad=4)

    if show_task_id:
        size_info = "Size-Preserving" if task.is_size_preserving else "Size-Changing"
        fig.suptitle(
            f"ARC Task: {task.task_id}  |  {num_train} Train, {num_test} Test  |  {size_info}",
            fontsize=12,
            fontweight="bold",
            y=0.995,
        )

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)

    return fig


def plot_task_summary(
    tasks: Sequence[ARCTask],
    max_tasks: int = 10,
    save_path: str | Path | None = None,
) -> list[plt.Figure]:
    """Plot multiple tasks and return list of figures."""
    figs = []
    for idx, task in enumerate(tasks[:max_tasks]):
        fig = plot_task(task)
        figs.append(fig)
    return figs


def render_task_ascii(task: ARCTask) -> str:
    """Produce an ASCII string summary of an entire ARC task."""
    lines = [
        f"=== Task: {task.task_id} ===",
        f"Train pairs: {task.num_train} | Test pairs: {task.num_test}",
        f"Unique colors: {sorted(list(task.unique_colors))}",
        "",
    ]
    for idx, p in enumerate(task.train, 1):
        lines.append(f"--- Train {idx} Input {p.input.shape} ---")
        lines.append(p.input.to_ascii())
        lines.append(f"--- Train {idx} Output {p.output.shape} ---")
        lines.append(p.output.to_ascii())
        lines.append("")
    for idx, p in enumerate(task.test, 1):
        lines.append(f"--- Test {idx} Input {p.input.shape} ---")
        lines.append(p.input.to_ascii())
        if p.output is not None:
            lines.append(f"--- Test {idx} Output {p.output.shape} ---")
            lines.append(p.output.to_ascii())
        else:
            lines.append(f"--- Test {idx} Output: [HIDDEN] ---")
        lines.append("")
    return "\n".join(lines)
