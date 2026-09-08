"""Raycasting engine for ARC spatial reasoning.

Implements directional ray propagation, path traversal, and collision detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.data.models import Grid


DIRECTION_VECTORS: dict[str, tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
    "up_left": (-1, -1),
    "up_right": (-1, 1),
    "down_left": (1, -1),
    "down_right": (1, 1),
}


@dataclass
class RaycastResult:
    """Dataclass holding raycasting traversal trajectory and termination metadata."""
    direction: str
    start: tuple[int, int]
    path: list[tuple[int, int]]
    hit_coord: tuple[int, int] | None
    hit_color: int | None
    stop_reason: str
    steps: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "start": list(self.start),
            "path": [list(p) for p in self.path],
            "hit_coord": list(self.hit_coord) if self.hit_coord else None,
            "hit_color": self.hit_color,
            "stop_reason": self.stop_reason,
            "steps": self.steps,
        }


def raycast(
    grid: Grid,
    start: tuple[int, int],
    direction: str,
    stop_condition: str = "boundary",
    background: int = 0,
    target_color: int | None = None,
    obstacle_coords: set[tuple[int, int]] | None = None,
    max_steps: int = 30,
) -> RaycastResult:
    """Cast a ray from start position in specified direction until stop condition is met.
    
    Args:
        grid: Input Grid.
        start: (row, col) origin.
        direction: Ray direction ('up', 'down', 'left', 'right', 'up_left', 'up_right', 'down_left', 'down_right').
        stop_condition: 'boundary', 'non_background', 'target_color', 'collision_with_object', 'landmark'.
        background: Background color (default 0).
        target_color: Specific color value for 'target_color' stop condition.
        obstacle_coords: Set of (r, c) tuples for 'collision_with_object' stop condition.
        max_steps: Maximum propagation steps.
        
    Returns:
        RaycastResult with path and termination details.
    """
    if direction not in DIRECTION_VECTORS:
        raise ValueError(f"Invalid ray direction '{direction}'. Must be one of {list(DIRECTION_VECTORS.keys())}")

    dr, dc = DIRECTION_VECTORS[direction]
    curr_r, curr_c = start
    h, w = grid.height, grid.width

    path: list[tuple[int, int]] = [(curr_r, curr_c)]
    hit_coord: tuple[int, int] | None = None
    hit_color: int | None = None
    stop_reason = "max_steps"

    steps = 0
    while steps < max_steps:
        next_r, next_c = curr_r + dr, curr_c + dc

        # 1. Boundary check
        if not (0 <= next_r < h and 0 <= next_c < w):
            stop_reason = "boundary"
            break

        curr_r, curr_c = next_r, next_c
        cell_color = grid.get(curr_r, curr_c)
        path.append((curr_r, curr_c))
        steps += 1

        # 2. Obstacle / collision check
        if stop_condition in ("collision_with_object", "landmark") and obstacle_coords:
            if (curr_r, curr_c) in obstacle_coords:
                hit_coord = (curr_r, curr_c)
                hit_color = cell_color
                stop_reason = "collision_with_object"
                break

        # 3. Target color check
        if stop_condition == "target_color" and target_color is not None:
            if cell_color == target_color:
                hit_coord = (curr_r, curr_c)
                hit_color = cell_color
                stop_reason = "target_color"
                break

        # 4. Non-background check
        if stop_condition in ("non_background", "landmark") and cell_color != background:
            hit_coord = (curr_r, curr_c)
            hit_color = cell_color
            stop_reason = "non_background"
            break

    return RaycastResult(
        direction=direction,
        start=start,
        path=path,
        hit_coord=hit_coord,
        hit_color=hit_color,
        stop_reason=stop_reason,
        steps=steps,
    )
