"""Unit tests for ARC task loader and dataset parsing."""

import json
from pathlib import Path
import pytest

from src.data.loader import (
    get_random_task,
    list_task_ids,
    load_dataset,
    load_task,
    load_task_from_dict,
)
from src.data.models import (
    ARCTask,
    InvalidGridError,
    MalformedTaskError,
)


@pytest.fixture
def sample_valid_task_dict() -> dict:
    return {
        "train": [
            {
                "input": [[0, 1], [2, 3]],
                "output": [[3, 2], [1, 0]],
            },
            {
                "input": [[4, 5], [6, 7]],
                "output": [[7, 6], [5, 4]],
            },
        ],
        "test": [
            {
                "input": [[1, 1], [2, 2]],
                "output": [[2, 2], [1, 1]],
            }
        ],
    }


def test_load_task_from_dict_valid(sample_valid_task_dict):
    task = load_task_from_dict(sample_valid_task_dict, task_id="sample_01")
    assert isinstance(task, ARCTask)
    assert task.task_id == "sample_01"
    assert task.num_train == 2
    assert task.num_test == 1
    assert task.train[0].input.shape == (2, 2)
    assert task.train[0].output.shape == (2, 2)
    assert task.test[0].input.shape == (2, 2)
    assert task.test[0].output.shape == (2, 2)


def test_load_task_from_dict_test_without_output():
    task_dict = {
        "train": [{"input": [[1]], "output": [[2]]}],
        "test": [{"input": [[3]]}],
    }
    task = load_task_from_dict(task_dict, task_id="test_hidden")
    assert task.num_train == 1
    assert task.num_test == 1
    assert task.test[0].output is None
    assert task.test[0].has_output is False


def test_load_task_missing_train_or_test():
    with pytest.raises(MalformedTaskError, match="missing required 'train' field"):
        load_task_from_dict({"test": [{"input": [[1]]}]})

    with pytest.raises(MalformedTaskError, match="missing required 'test' field"):
        load_task_from_dict({"train": [{"input": [[1]], "output": [[1]]}]})


def test_load_task_malformed_pair_structure():
    with pytest.raises(MalformedTaskError, match="missing 'input' or 'output' keys"):
        load_task_from_dict({
            "train": [{"input": [[1]]}],  # Missing output
            "test": [{"input": [[1]]}],
        })

    with pytest.raises(MalformedTaskError, match="missing required 'input' key"):
        load_task_from_dict({
            "train": [{"input": [[1]], "output": [[1]]}],
            "test": [{"output": [[1]]}],  # Missing input
        })


def test_load_task_invalid_grid_data():
    with pytest.raises(InvalidGridError):
        load_task_from_dict({
            "train": [{"input": [[1, 2], [3]], "output": [[1, 2], [3, 4]]}],  # Non-rectangular
            "test": [{"input": [[1, 2], [3, 4]]}],
        })


def test_load_task_from_file(tmp_path: Path, sample_valid_task_dict):
    task_file = tmp_path / "task_1234abcd.json"
    with open(task_file, "w") as f:
        json.dump(sample_valid_task_dict, f)

    task = load_task(task_file)
    assert task.task_id == "task_1234abcd"
    assert task.num_train == 2


def test_load_task_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_task(Path("non_existent_directory_12345/no_file.json"))


def test_load_dataset_and_list_ids(tmp_path: Path, sample_valid_task_dict):
    # Create 3 task files
    for i in range(3):
        task_file = tmp_path / f"task_{i:02d}.json"
        with open(task_file, "w") as f:
            json.dump(sample_valid_task_dict, f)

    task_ids = list_task_ids(tmp_path)
    assert task_ids == ["task_00", "task_01", "task_02"]

    dataset = load_dataset(tmp_path)
    assert len(dataset) == 3
    assert "task_00" in dataset
    assert isinstance(dataset["task_00"], ARCTask)


def test_get_random_task(tmp_path: Path, sample_valid_task_dict):
    for i in range(5):
        task_file = tmp_path / f"task_{i:02d}.json"
        with open(task_file, "w") as f:
            json.dump(sample_valid_task_dict, f)

    dataset = load_dataset(tmp_path)
    random_task_1 = get_random_task(dataset, seed=42)
    random_task_2 = get_random_task(dataset, seed=42)
    assert random_task_1.task_id == random_task_2.task_id

    # Test random task from dir
    random_from_dir = get_random_task(tmp_path, seed=42)
    assert isinstance(random_from_dir, ARCTask)
