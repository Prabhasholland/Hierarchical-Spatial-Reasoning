"""Unit tests for ARC visualizer (matplotlib and ASCII rendering)."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless testing
import matplotlib.pyplot as plt

from src.data.models import (
    ARCTask,
    Grid,
    TestPair,
    TrainingPair,
)
from src.data.visualizer import (
    ARC_CMAP,
    ARC_NORM,
    get_arc_colormap,
    plot_grid,
    plot_task,
    render_task_ascii,
)


def sample_task() -> ARCTask:
    train_pair = TrainingPair(
        input=Grid.from_list([[0, 1], [2, 3]]),
        output=Grid.from_list([[3, 2], [1, 0]]),
    )
    test_pair_1 = TestPair(
        input=Grid.from_list([[4, 5], [6, 7]]),
        output=Grid.from_list([[7, 6], [5, 4]]),
    )
    test_pair_2 = TestPair(
        input=Grid.from_list([[8, 9], [0, 1]]),
        output=None,
    )
    return ARCTask(
        task_id="vis_test_01",
        train=[train_pair],
        test=[test_pair_1, test_pair_2],
    )


def test_colormap_validity():
    cmap, norm = get_arc_colormap()
    assert cmap.N == 10
    assert norm.Ncmap == 10
    assert len(norm.boundaries) == 11


def test_plot_grid():
    grid = Grid.from_list([[0, 1, 2], [3, 4, 5]])
    fig, ax = plt.subplots()
    ret_ax = plot_grid(grid, title="Test Grid", ax=ax, show_dims=True)
    assert ret_ax is ax
    plt.close(fig)


def test_plot_task(tmp_path: Path):
    task = sample_task()
    fig = plot_task(task, show_task_id=True, show_dims=True)
    assert fig is not None
    
    # Test saving to file
    out_img = tmp_path / "task_plot.png"
    plot_task(task, save_path=out_img)
    assert out_img.exists()
    assert out_img.stat().st_size > 0
    plt.close("all")


def test_render_task_ascii():
    task = sample_task()
    ascii_out = render_task_ascii(task)
    assert "vis_test_01" in ascii_out
    assert "Train 1 Input" in ascii_out
    assert "Test 1 Output" in ascii_out
    assert "[HIDDEN]" in ascii_out
