"""Unit tests for the evaluation harness."""

from src.data.models import ARCTask, Grid, TestPair, TrainingPair
from src.evaluation.harness import TaskOutcome, evaluate_dataset, evaluate_single_task
from src.solvers.rule_based import RuleBasedSearchSolver


def test_evaluate_single_task_solved():
    # Construct task solved by HorizontalFlip
    train_in = Grid.from_list([[1, 2], [3, 4]])
    train_out = Grid.from_list([[2, 1], [4, 3]])
    test_in = Grid.from_list([[5, 6], [7, 8]])
    test_out = Grid.from_list([[6, 5], [8, 7]])

    task = ARCTask(
        task_id="test_hflip",
        train=[TrainingPair(train_in, train_out)],
        test=[TestPair(test_in, test_out)],
    )

    solver = RuleBasedSearchSolver()
    res = evaluate_single_task(solver, task)

    assert res.is_task_solved is True
    assert res.outcome == TaskOutcome.SOLVED
    assert res.test_pairs_solved == 1
    assert "HorizontalFlip" in res.chosen_rule


def test_evaluate_dataset_aggregate():
    # 2 tasks: 1 solvable (HFlip), 1 unsolvable (random/arbitrary)
    t1 = ARCTask(
        task_id="t1",
        train=[TrainingPair(Grid.from_list([[1, 2]]), Grid.from_list([[2, 1]]))],
        test=[TestPair(Grid.from_list([[3, 4]]), Grid.from_list([[4, 3]]))],
    )
    t2 = ARCTask(
        task_id="t2",
        train=[TrainingPair(Grid.from_list([[1, 2]]), Grid.from_list([[9, 9]]))],
        test=[TestPair(Grid.from_list([[3, 4]]), Grid.from_list([[8, 8]]))],
    )

    solver = RuleBasedSearchSolver()
    stats = evaluate_dataset(solver, [t1, t2])

    assert stats["total_tasks_evaluated"] == 2
    assert stats["tasks_solved"] == 1
    assert stats["task_level_accuracy_pct"] == 50.0
    assert "t1" in stats["solved_task_ids"]
    assert stats["total_runtime_seconds"] > 0
