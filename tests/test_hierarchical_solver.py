"""Unit tests for RuleBasedHierarchicalSolver_v1."""

from __future__ import annotations

from src.data.models import ARCTask, Grid, TestPair, TrainingPair
from src.solvers.hierarchical_solver import RuleBasedHierarchicalSolver_v1


def test_hierarchical_solver_end_to_end():
    task = ARCTask(
        task_id="h_solver_test",
        train=[
            TrainingPair(
                input=Grid.from_list([
                    [1, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1],
                ]),
                output=Grid.from_list([
                    [1, 1, 1],
                    [1, 4, 1],
                    [1, 1, 1],
                ]),
            )
        ],
        test=[
            TestPair(
                input=Grid.from_list([
                    [2, 2, 2],
                    [2, 0, 2],
                    [2, 2, 2],
                ]),
                output=Grid.from_list([
                    [2, 2, 2],
                    [2, 4, 2],
                    [2, 2, 2],
                ]),
            )
        ],
    )

    solver = RuleBasedHierarchicalSolver_v1(max_depth=3)
    details = solver.solve_with_details(task)

    assert details["solved_on_train"] is True
    assert details["predictions"][0].to_list() == [
        [2, 2, 2],
        [2, 4, 2],
        [2, 2, 2],
    ]
