"""Tests for core ARC data models (Grid, TrainingPair, TestPair, ARCTask)."""

import pytest

from src.data.models import (
    ARCTask,
    Grid,
    InvalidGridError,
    MalformedTaskError,
    TestPair,
    TrainingPair,
)


def test_grid_creation_and_properties():
    raw = [
        [0, 1, 2],
        [3, 4, 5],
    ]
    grid = Grid.from_list(raw)
    assert grid.height == 2
    assert grid.width == 3
    assert grid.shape == (2, 3)
    assert grid.size == 6
    assert grid.unique_colors == {0, 1, 2, 3, 4, 5}
    assert grid.to_list() == raw
    assert grid.get(0, 1) == 1
    assert grid.get(1, 2) == 5
    assert len(grid) == 2


def test_grid_equality_and_immutability():
    g1 = Grid.from_list([[1, 2], [3, 4]])
    g2 = Grid.from_list([[1, 2], [3, 4]])
    g3 = Grid.from_list([[1, 2], [3, 5]])

    assert g1 == g2
    assert g1 != g3
    assert hash(g1) == hash(g2)

    # Ensure immutability
    with pytest.raises(Exception):
        g1.cells = ((1, 1), (1, 1))  # type: ignore


def test_grid_validation_non_rectangular():
    raw = [
        [1, 2, 3],
        [4, 5],
    ]
    with pytest.raises(InvalidGridError, match="not rectangular"):
        Grid.from_list(raw)


def test_grid_validation_empty():
    with pytest.raises(InvalidGridError):
        Grid.from_list([])


def test_grid_validation_color_out_of_range():
    # Value 10 is invalid
    with pytest.raises(InvalidGridError, match="invalid color value"):
        Grid.from_list([[0, 10]])

    # Negative value is invalid
    with pytest.raises(InvalidGridError, match="invalid color value"):
        Grid.from_list([[-1, 2]])


def test_grid_validation_non_integer():
    with pytest.raises(InvalidGridError, match="non-integer"):
        Grid.from_list([[1, "red"]])  # type: ignore


def test_grid_validation_dimensions_bounds():
    # Max dimension is 30
    too_wide = [[0] * 31]
    with pytest.raises(InvalidGridError, match="out of bounds"):
        Grid.from_list(too_wide)

    too_tall = [[0]] * 31
    with pytest.raises(InvalidGridError, match="out of bounds"):
        Grid.from_list(too_tall)


def test_grid_ascii_and_counts():
    grid = Grid.from_list([[0, 1], [1, 2]])
    counts = grid.color_counts()
    assert counts == {0: 1, 1: 2, 2: 1}
    assert grid.background_color == 1
    ascii_str = grid.to_ascii()
    assert "+" in ascii_str
    assert "." in ascii_str  # 0 renders as .


def test_training_pair():
    in_grid = Grid.from_list([[1, 2], [3, 4]])
    out_grid = Grid.from_list([[1, 2], [3, 4]])
    pair = TrainingPair(input=in_grid, output=out_grid)
    assert pair.is_size_preserving is True
    assert pair.input_shape == (2, 2)
    assert pair.output_shape == (2, 2)
    assert pair.to_dict() == {
        "input": [[1, 2], [3, 4]],
        "output": [[1, 2], [3, 4]],
    }


def test_training_pair_size_changing():
    in_grid = Grid.from_list([[1, 2], [3, 4]])
    out_grid = Grid.from_list([[1, 2, 3]])
    pair = TrainingPair(input=in_grid, output=out_grid)
    assert pair.is_size_preserving is False


def test_test_pair_with_and_without_output():
    in_grid = Grid.from_list([[1, 2], [3, 4]])
    out_grid = Grid.from_list([[5, 6], [7, 8]])

    pair_with_out = TestPair(input=in_grid, output=out_grid)
    assert pair_with_out.has_output is True
    assert pair_with_out.is_size_preserving is True
    assert "output" in pair_with_out.to_dict()

    pair_no_out = TestPair(input=in_grid, output=None)
    assert pair_no_out.has_output is False
    assert pair_no_out.is_size_preserving is None
    assert "output" not in pair_no_out.to_dict()


def test_arc_task_creation_and_properties():
    train_pair1 = TrainingPair(
        input=Grid.from_list([[1, 0], [0, 1]]),
        output=Grid.from_list([[2, 0], [0, 2]]),
    )
    train_pair2 = TrainingPair(
        input=Grid.from_list([[3, 0], [0, 3]]),
        output=Grid.from_list([[4, 0], [0, 4]]),
    )
    test_pair = TestPair(
        input=Grid.from_list([[5, 0], [0, 5]]),
        output=Grid.from_list([[6, 0], [0, 6]]),
    )

    task = ARCTask(
        task_id="test_task_01",
        train=[train_pair1, train_pair2],
        test=[test_pair],
    )

    assert task.task_id == "test_task_01"
    assert task.num_train == 2
    assert task.num_test == 1
    assert task.is_size_preserving is True
    assert task.has_fixed_output_dim is True
    assert task.unique_colors == {0, 1, 2, 3, 4, 5, 6}

    summary = task.summary()
    assert summary["task_id"] == "test_task_01"
    assert summary["num_train"] == 2
    assert summary["num_test"] == 1
    assert summary["is_size_preserving"] is True


def test_arc_task_validation_missing_train_or_test():
    grid = Grid.from_list([[0]])
    pair = TrainingPair(input=grid, output=grid)
    t_pair = TestPair(input=grid)

    with pytest.raises(MalformedTaskError, match="no training examples"):
        ARCTask(task_id="empty_train", train=[], test=[t_pair])

    with pytest.raises(MalformedTaskError, match="no test examples"):
        ARCTask(task_id="empty_test", train=[pair], test=[])
