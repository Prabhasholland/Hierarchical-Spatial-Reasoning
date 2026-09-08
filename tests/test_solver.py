"""Unit tests for the rule-based search solver and components."""

from src.data.models import ARCTask, Grid, TestPair, TrainingPair
from src.solvers.candidate_generator import CandidateGenerator
from src.solvers.rule_based import RuleBasedSearchSolver
from src.solvers.verifier import CandidateRanker, CandidateVerifier
from src.transformations.geometric import Rotate90Transformation


def create_rotation_task() -> ARCTask:
    # 90-degree clockwise rotation task
    train1_in = Grid.from_list([[1, 2], [3, 4]])
    train1_out = Grid.from_list([[3, 1], [4, 2]])

    train2_in = Grid.from_list([[5, 6], [7, 8]])
    train2_out = Grid.from_list([[7, 5], [8, 6]])

    test_in = Grid.from_list([[9, 0], [1, 2]])
    test_out = Grid.from_list([[1, 9], [2, 0]])

    return ARCTask(
        task_id="rotation_task",
        train=[
            TrainingPair(train1_in, train1_out),
            TrainingPair(train2_in, train2_out),
        ],
        test=[TestPair(test_in, test_out)],
    )


def create_color_swap_task() -> ARCTask:
    train1_in = Grid.from_list([[1, 2], [2, 1]])
    train1_out = Grid.from_list([[3, 4], [4, 3]])

    test_in = Grid.from_list([[2, 1], [1, 2]])
    test_out = Grid.from_list([[4, 3], [3, 4]])

    return ARCTask(
        task_id="color_sub_task",
        train=[TrainingPair(train1_in, train1_out)],
        test=[TestPair(test_in, test_out)],
    )


def test_candidate_generator():
    task = create_rotation_task()
    gen = CandidateGenerator()
    candidates = gen.generate(task)
    assert len(candidates) > 0
    # Check that Rotate90 is in candidates
    has_rot90 = any(isinstance(c, Rotate90Transformation) for c in candidates)
    assert has_rot90


def test_candidate_verifier():
    task = create_rotation_task()
    verifier = CandidateVerifier()
    assert verifier.verify(Rotate90Transformation(), task) is True

    from src.transformations.geometric import Rotate180Transformation
    assert verifier.verify(Rotate180Transformation(), task) is False


def test_candidate_ranker():
    task = create_rotation_task()
    from src.transformations.geometric import IdentityTransformation, Rotate90Transformation
    ranker = CandidateRanker()
    ranked = ranker.rank([Rotate90Transformation(), IdentityTransformation()], task)
    assert ranked[0].name == "Identity"  # Lowest complexity first


def test_rule_based_solver_solves_rotation():
    task = create_rotation_task()
    solver = RuleBasedSearchSolver()
    details = solver.solve_with_details(task)

    assert details["solved_on_train"] is True
    assert "Rotate90" in details["chosen_rule"]
    assert len(details["predictions"]) == 1
    assert details["predictions"][0] == task.test[0].output


def test_rule_based_solver_solves_color_substitution():
    task = create_color_swap_task()
    solver = RuleBasedSearchSolver()
    details = solver.solve_with_details(task)

    assert details["solved_on_train"] is True
    assert details["predictions"][0] == task.test[0].output


def test_solver_fallback_on_unsolvable():
    # Impossible contradictory task
    train1_in = Grid.from_list([[1]])
    train1_out = Grid.from_list([[2]])
    train2_in = Grid.from_list([[1]])
    train2_out = Grid.from_list([[3]])  # Contradiction: same input -> different output

    task = ARCTask(
        task_id="contradiction",
        train=[TrainingPair(train1_in, train1_out), TrainingPair(train2_in, train2_out)],
        test=[TestPair(Grid.from_list([[1]]))],
    )
    solver = RuleBasedSearchSolver(fallback_to_identity=True)
    details = solver.solve_with_details(task)
    assert details["solved_on_train"] is False
    assert details["predictions"][0] == Grid.from_list([[1]])
