"""Unit tests for Relational Object Graph G=(V, E)."""

from __future__ import annotations

import json

from src.data.models import Grid
from src.objects.graph import ObjectGraph


def test_object_graph_construction():
    grid = Grid.from_list([
        [1, 2, 0],
        [0, 0, 0],
    ])
    graph = ObjectGraph.from_grid(grid)
    assert len(graph) == 2
    assert len(graph.nodes) == 2
    assert len(graph.edges) >= 2

    node0 = graph.get_node(0)
    assert node0 is not None
    assert node0.object_id == 0

    neighbors = graph.get_neighbors(0, relation="adjacent_to")
    assert len(neighbors) == 1
    assert neighbors[0].object_id == 1


def test_graph_serialization():
    grid = Grid.from_list([
        [1, 0],
        [0, 2],
    ])
    graph = ObjectGraph.from_grid(grid)
    json_str = graph.to_json()
    parsed = json.loads(json_str)

    assert "num_nodes" in parsed
    assert parsed["num_nodes"] == 2
    assert len(parsed["nodes"]) == 2
    assert "strategy" in parsed
