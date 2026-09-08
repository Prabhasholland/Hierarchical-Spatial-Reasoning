"""Topology package for ARC grid reasoning."""

from src.topology.engine import (
    boundary_trace,
    connected_regions,
    enclosure_detection,
    flood_fill,
    hole_detection,
    region_adjacency,
)

__all__ = [
    "flood_fill",
    "connected_regions",
    "enclosure_detection",
    "boundary_trace",
    "hole_detection",
    "region_adjacency",
]
