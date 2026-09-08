"""Spatial hypothesis generator and transformation rules for ARC tasks.

Implements spatial transformation classes including:
- Enclosed cavity filling (FillEnclosedTransformation)
- Boundary tracing (TraceBoundaryTransformation)
- Raycasting projection (RaycastProjectionTransformation)
- Landmark connection (ConnectLandmarksTransformation)
- Object extension until collision (ExtendCollisionTransformation)
"""

from __future__ import annotations

from typing import Any, Sequence

from src.data.models import ARCTask, Grid
from src.spatial.dsl import (
    connect_same_color_op,
    extend_until_collision_op,
    fill_enclosed_op,
    project_until_boundary_op,
    project_until_object_op,
    trace_boundary_op,
)
from src.transformations.base import Transformation


class FillEnclosedTransformation(Transformation):
    """Fill enclosed background cavities with fill_color."""

    def __init__(self, fill_color: int, background: int = 0) -> None:
        self.fill_color = fill_color
        self.background = background

    def apply(self, grid: Grid) -> Grid:
        return fill_enclosed_op(grid, fill_color=self.fill_color, background=self.background)

    @property
    def name(self) -> str:
        return f"FillEnclosed(color={self.fill_color})"

    @property
    def complexity(self) -> float:
        return 2.1

    @property
    def params(self) -> dict[str, Any]:
        return {"fill_color": self.fill_color, "background": self.background}


class TraceBoundaryTransformation(Transformation):
    """Trace and recolor object boundaries to boundary_color."""

    def __init__(
        self,
        boundary_color: int,
        target_color: int | None = None,
        background: int = 0,
    ) -> None:
        self.boundary_color = boundary_color
        self.target_color = target_color
        self.background = background

    def apply(self, grid: Grid) -> Grid:
        return trace_boundary_op(
            grid,
            target_color=self.target_color,
            boundary_color=self.boundary_color,
            background=self.background,
        )

    @property
    def name(self) -> str:
        return f"TraceBoundary(boundary_color={self.boundary_color}, target_color={self.target_color})"

    @property
    def complexity(self) -> float:
        return 2.3

    @property
    def params(self) -> dict[str, Any]:
        return {
            "boundary_color": self.boundary_color,
            "target_color": self.target_color,
            "background": self.background,
        }


class RaycastProjectionTransformation(Transformation):
    """Project ray from landmark points/objects in specified direction."""

    def __init__(
        self,
        direction: str,
        fill_color: int,
        source_color: int | None = None,
        stop_condition: str = "boundary",  # 'boundary' or 'object'
        background: int = 0,
    ) -> None:
        self.direction = direction
        self.fill_color = fill_color
        self.source_color = source_color
        self.stop_condition = stop_condition
        self.background = background

    def apply(self, grid: Grid) -> Grid:
        h, w = grid.height, grid.width
        curr_grid = grid

        # Identify source seed coordinates
        coords = []
        for r in range(h):
            for c in range(w):
                val = grid.get(r, c)
                if self.source_color is not None:
                    if val == self.source_color:
                        coords.append((r, c))
                elif val != self.background:
                    coords.append((r, c))

        for sr, sc in coords:
            if self.stop_condition == "boundary":
                curr_grid = project_until_boundary_op(curr_grid, start=(sr, sc), direction=self.direction, fill_color=self.fill_color)
            else:
                curr_grid = project_until_object_op(curr_grid, start=(sr, sc), direction=self.direction, fill_color=self.fill_color, background=self.background)

        return curr_grid

    @property
    def name(self) -> str:
        return f"RaycastProjection(dir={self.direction}, fill={self.fill_color}, stop={self.stop_condition})"

    @property
    def complexity(self) -> float:
        return 2.4

    @property
    def params(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "fill_color": self.fill_color,
            "source_color": self.source_color,
            "stop_condition": self.stop_condition,
        }


class ConnectLandmarksTransformation(Transformation):
    """Connect matching landmark dots of target_color with straight lines."""

    def __init__(
        self,
        target_color: int,
        line_color: int | None = None,
        background: int = 0,
    ) -> None:
        self.target_color = target_color
        self.line_color = line_color
        self.background = background

    def apply(self, grid: Grid) -> Grid:
        return connect_same_color_op(
            grid,
            target_color=self.target_color,
            line_color=self.line_color,
            background=self.background,
        )

    @property
    def name(self) -> str:
        return f"ConnectLandmarks(target={self.target_color}, line={self.line_color})"

    @property
    def complexity(self) -> float:
        return 2.5

    @property
    def params(self) -> dict[str, Any]:
        return {
            "target_color": self.target_color,
            "line_color": self.line_color,
            "background": self.background,
        }


class ExtendCollisionTransformation(Transformation):
    """Extend objects of target_color in direction until hitting obstacle."""

    def __init__(
        self,
        target_color: int,
        direction: str,
        fill_color: int | None = None,
        background: int = 0,
    ) -> None:
        self.target_color = target_color
        self.direction = direction
        self.fill_color = fill_color
        self.background = background

    def apply(self, grid: Grid) -> Grid:
        return extend_until_collision_op(
            grid,
            target_color=self.target_color,
            direction=self.direction,
            fill_color=self.fill_color,
            background=self.background,
        )

    @property
    def name(self) -> str:
        return f"ExtendCollision(target={self.target_color}, dir={self.direction})"

    @property
    def complexity(self) -> float:
        return 2.6

    @property
    def params(self) -> dict[str, Any]:
        return {
            "target_color": self.target_color,
            "direction": self.direction,
            "fill_color": self.fill_color,
        }


class SpatialCandidateGenerator:
    """Proposes spatial and topological transformation hypotheses for an ARC task."""

    def __init__(self, max_candidates: int = 500) -> None:
        self.max_candidates = max_candidates

    def generate(self, task: ARCTask) -> list[Transformation]:
        candidates: list[Transformation] = []
        colors_used = sorted(list(task.unique_colors))
        directions = ("up", "down", "left", "right", "up_left", "up_right", "down_left", "down_right")

        # 1. Enclosed cavity fills
        for fill_c in colors_used:
            if fill_c != 0:
                candidates.append(FillEnclosedTransformation(fill_color=fill_c, background=0))

        # 2. Boundary trace transformations
        for b_color in colors_used:
            if b_color != 0:
                candidates.append(TraceBoundaryTransformation(boundary_color=b_color, background=0))
                for target_c in colors_used:
                    if target_c not in (0, b_color):
                        candidates.append(
                            TraceBoundaryTransformation(boundary_color=b_color, target_color=target_c, background=0)
                        )

        # 3. Raycast projection transformations
        for d in ("up", "down", "left", "right"):
            for fill_c in colors_used:
                if fill_c != 0:
                    candidates.append(
                        RaycastProjectionTransformation(direction=d, fill_color=fill_c, stop_condition="boundary")
                    )
                    candidates.append(
                        RaycastProjectionTransformation(direction=d, fill_color=fill_c, stop_condition="object")
                    )
                    for src_c in colors_used:
                        if src_c not in (0, fill_c):
                            candidates.append(
                                RaycastProjectionTransformation(
                                    direction=d,
                                    fill_color=fill_c,
                                    source_color=src_c,
                                    stop_condition="boundary",
                                )
                            )

        # 4. Landmark connection transformations
        for target_c in colors_used:
            if target_c != 0:
                candidates.append(ConnectLandmarksTransformation(target_color=target_c))
                for line_c in colors_used:
                    if line_c not in (0, target_c):
                        candidates.append(ConnectLandmarksTransformation(target_color=target_c, line_color=line_c))

        # 5. Extend collision transformations
        for d in ("up", "down", "left", "right"):
            for target_c in colors_used:
                if target_c != 0:
                    candidates.append(ExtendCollisionTransformation(target_color=target_c, direction=d))

        # Deduplicate
        unique: list[Transformation] = []
        seen = set()
        for cand in candidates:
            cand_repr = repr(cand)
            if cand_repr not in seen:
                seen.add(cand_repr)
                unique.append(cand)
                if len(unique) >= self.max_candidates:
                    break

        return unique
