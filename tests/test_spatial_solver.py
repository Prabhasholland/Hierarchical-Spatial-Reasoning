"""Unit tests for RuleBasedSpatialObjectSolver_v1."""

from __future__ import annotations

from src.data.models import ARCTask, Grid, TestPair, TrainingPair
from src.solvers.spatial_solver import RuleBasedSpatialObjectSolver_v1


def test_spatial_solver_solves_enclosed_fill():
    # Task: fill enclosed cavities with color 3
    task = ARCTask(
        task_id="spatial_fill_task",
        train=[
            TrainingPair(
                input=Grid.from_list([
                    [1, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1],
                ]),
                output=Grid.from_list([
                    [1, 1, 1],
                    [1, 3, 1],
                    [1, 1, 1],
                ]),
            )
        ],
        test=[
            TestPair(
                input=Grid.from_list([
                    [2, 2, 2, 2],
                    [2, 0, 0, 2],
                    [2, 2, 2, 2],
                ]),
                output=Grid.from_list([
                    [2, 2, 2, 2],
                    [2, 3, 3, 2],
                    [2, 2, 2, 2],
                ]),
            )
        ],
    )

    solver = RuleBasedSpatialObjectSolver_v1()
    details = solver.solve_with_details(task)

    assert details["solved_on_train"] is True
    assert details["predictions"][0].to_list() == [
        [2, 2, 2, 2],
        [2, 3, 3, 2],
        [2, 2, 2, 2],
    ]
