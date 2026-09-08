"""Unit tests for dataset statistics computation."""

from src.data.models import (
    ARCTask,
    Grid,
    TestPair,
    TrainingPair,
)
from src.data.stats import compute_dataset_statistics


def create_mock_tasks() -> list[ARCTask]:
    # Task 1: Size preserving, colors 0, 1, 2
    t1 = ARCTask(
        task_id="t1",
        train=[
            TrainingPair(
                input=Grid.from_list([[0, 1], [1, 0]]),
                output=Grid.from_list([[0, 2], [2, 0]]),
            )
        ],
        test=[
            TestPair(
                input=Grid.from_list([[1, 0], [0, 1]]),
                output=Grid.from_list([[2, 0], [0, 2]]),
            )
        ],
    )
    
    # Task 2: Size changing (3x3 to 1x1), colors 0, 5
    t2 = ARCTask(
        task_id="t2",
        train=[
            TrainingPair(
                input=Grid.from_list([[0, 0, 0], [0, 5, 0], [0, 0, 0]]),
                output=Grid.from_list([[5]]),
            )
        ],
        test=[
            TestPair(
                input=Grid.from_list([[0, 5, 0], [5, 5, 5], [0, 5, 0]]),
                output=Grid.from_list([[5]]),
            )
        ],
    )
    return [t1, t2]


def test_compute_dataset_statistics():
    tasks = create_mock_tasks()
    stats = compute_dataset_statistics(tasks)

    assert stats["total_tasks"] == 2
    assert stats["total_train_pairs"] == 2
    assert stats["total_test_pairs"] == 2
    assert stats["avg_train_pairs_per_task"] == 1.0
    assert stats["avg_test_pairs_per_task"] == 1.0
    assert stats["size_preserving_task_count"] == 1
    assert stats["size_preserving_percentage"] == 50.0

    color_stats = stats["color_statistics"]
    assert 0 in color_stats
    assert color_stats[0]["cell_count"] > 0
    assert color_stats[5]["task_presence_count"] == 1
    assert color_stats[5]["task_presence_percentage"] == 50.0


def test_compute_dataset_statistics_empty():
    stats = compute_dataset_statistics([])
    assert stats["total_tasks"] == 0
    assert stats["total_train_pairs"] == 0
