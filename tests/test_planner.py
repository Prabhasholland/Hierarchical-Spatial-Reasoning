"""Unit tests for HierarchicalPlanner."""

from __future__ import annotations

from src.data.models import ARCTask, Grid, TestPair, TrainingPair
from src.reasoning.planner import HierarchicalPlanner


def test_planner_single_step():
    task = ARCTask(
        task_id="plan_single_step",
        train=[
            TrainingPair(
                input=Grid.from_list([
                    [1, 0],
                    [0, 0],
                ]),
                output=Grid.from_list([
                    [0, 1],
                    [0, 0],
                ]),
            )
        ],
        test=[
            TestPair(
                input=Grid.from_list([
                    [2, 0],
                    [0, 0],
                ])
            )
        ],
    )

    planner = HierarchicalPlanner(max_depth=2, max_nodes_expanded=100)
    pipeline = planner.plan(task)
    assert pipeline is not None
    assert len(pipeline) >= 1
