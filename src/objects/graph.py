"""Relational Object Graph representation for ARC grids.

Represents an ARC grid as G = (V, E) where:
  V = ObjectInstance nodes containing complete attribute vectors
  E = Relational edges capturing spatial and semantic relationships
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from src.data.models import Grid
from src.objects.relations import compute_all_pairwise_relations
from src.objects.segmentation import (
    ObjectInstance,
    SegmentationHypothesis,
    SegmentationStrategy,
    segment_grid,
)


@dataclass
class ObjectNode:
    """A node in the Relational Object Graph."""
    object_id: int
    attributes: dict[str, Any]
    instance: ObjectInstance | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "attributes": self.attributes,
        }


@dataclass
class ObjectEdge:
    """A directed edge representing spatial/semantic relations between two object nodes."""
    source_id: int
    target_id: int
    relations: list[str]
    distance: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relations": self.relations,
            "distance": self.distance,
        }


@dataclass
class ObjectGraph:
    """Relational graph G = (V, E) of an ARC grid."""
    nodes: list[ObjectNode]
    edges: list[ObjectEdge]
    background: int
    grid_shape: tuple[int, int]
    strategy: str

    @classmethod
    def from_grid(
        cls,
        grid: Grid,
        strategy: SegmentationStrategy | str = SegmentationStrategy.MONOCHROMATIC_4,
        background: int | None = None,
    ) -> ObjectGraph:
        """Construct a Relational Object Graph directly from an ARC grid."""
        hyp = segment_grid(grid, strategy=strategy, background=background)
        return cls.from_hypothesis(hyp)

    @classmethod
    def from_hypothesis(cls, hyp: SegmentationHypothesis) -> ObjectGraph:
        """Construct graph from an existing SegmentationHypothesis."""
        nodes = [
            ObjectNode(
                object_id=obj.object_id,
                attributes=obj.to_dict(),
                instance=obj,
            )
            for obj in hyp.objects
        ]

        raw_relations = compute_all_pairwise_relations(hyp.objects, hyp.grid_shape)
        edges = [
            ObjectEdge(
                source_id=r["source_id"],
                target_id=r["target_id"],
                relations=r["relations"],
                distance=r["distance"],
            )
            for r in raw_relations
            if r["relations"]  # only store edges with at least one active relation
        ]

        strat_str = hyp.strategy.value if hasattr(hyp.strategy, "value") else str(hyp.strategy)

        return cls(
            nodes=nodes,
            edges=edges,
            background=hyp.background,
            grid_shape=hyp.grid_shape,
            strategy=strat_str,
        )

    def get_node(self, object_id: int) -> ObjectNode | None:
        """Retrieve node by object_id."""
        for n in self.nodes:
            if n.object_id == object_id:
                return n
        return None

    def get_neighbors(self, object_id: int, relation: str | None = None) -> list[ObjectNode]:
        """Find neighboring nodes connected by edges."""
        target_ids = []
        for e in self.edges:
            if e.source_id == object_id:
                if relation is None or relation in e.relations:
                    target_ids.append(e.target_id)
        return [n for n in self.nodes if n.object_id in target_ids]

    def get_edges_for(self, object_id: int) -> list[ObjectEdge]:
        """Get all outgoing edges from a specific node."""
        return [e for e in self.edges if e.source_id == object_id]

    def filter_nodes(self, predicate: Callable[[ObjectNode], bool]) -> list[ObjectNode]:
        """Filter graph nodes matching a given attribute predicate."""
        return [n for n in self.nodes if predicate(n)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "background": self.background,
            "grid_shape": list(self.grid_shape),
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize graph to a JSON formatted string for inspection."""
        return json.dumps(self.to_dict(), indent=indent)

    def __len__(self) -> int:
        return len(self.nodes)

    def __str__(self) -> str:
        lines = [
            f"ObjectGraph(nodes={len(self.nodes)}, edges={len(self.edges)}, "
            f"strategy={self.strategy}, bg={self.background}, shape={self.grid_shape})"
        ]
        for n in self.nodes:
            attr = n.attributes
            c = attr.get("color")
            area = attr.get("area")
            bb = attr.get("bounding_box")
            lines.append(f"  Node {n.object_id}: color={c}, area={area}, bbox={bb}")
        for e in self.edges[:10]:
            lines.append(f"  Edge {e.source_id}->{e.target_id}: {e.relations} (d={e.distance})")
        if len(self.edges) > 10:
            lines.append(f"  ... ({len(self.edges) - 10} more edges)")
        return "\n".join(lines)
