"""Unit tests for RuleBasedObjectSolver_v1 and object-aware candidate generation."""

from __future__ import annotations

from src.data.models import ARCTask, Grid, TestPair, TrainingPair
from src.solvers.object_reasoning import ObjectCandidateGenerator
from src.solvers.object_solver import RuleBasedObjectSolver_v1


def test_object_candidate_generator():
    task = ARCTask(
        task_id="obj_test_1",
        train=[
            TrainingPair(
                input=Grid.from_list([
                    [1, 1, 0, 2],
                    [1, 1, 0, 0],
                ]),
                output=Grid.from_list([
                    [1, 1],
                    [1, 1],
                ]),
            )
        ],
        test=[
            TestPair(
                input=Grid.from_list([
                    [3, 3, 0, 4],
                    [3, 3, 0, 0],
                ]),
                output=Grid.from_list([
                    [3, 3],
                    [3, 3],
                ]),
            )
        ],
    )

    gen = ObjectCandidateGenerator()
    candidates = gen.generate(task)
    assert len(candidates) > 0


def test_object_solver_solves_extraction():
    # Task: Extract largest object
    task = ARCTask(
        task_id="extract_largest_task",
        train=[
            TrainingPair(
                input=Grid.from_list([
                    [1, 1, 0, 2],
                    [1, 1, 0, 0],
                ]),
                output=Grid.from_list([
                    [1, 1],
                    [1, 1],
                ]),
            ),
            TrainingPair(
                input=Grid.from_list([
                    [0, 5, 0, 0],
                    [0, 0, 3, 3],
                    [0, 0, 3, 3],
                ]),
                output=Grid.from_list([
                    [3, 3],
                    [3, 3],
                ]),
            ),
        ],
        test=[
            TestPair(
                input=Grid.from_list([
                    [8, 0, 0],
                    [0, 4, 4],
                    [0, 4, 4],
                ]),
                output=Grid.from_list([
                    [4, 4],
                    [4, 4],
                ]),
            )
        ],
    )

    solver = RuleBasedObjectSolver_v1()
    details = solver.solve_with_details(task)

    assert details["solved_on_train"] is True
    assert details["predictions"][0].to_list() == [[4, 4], [4, 4]]
