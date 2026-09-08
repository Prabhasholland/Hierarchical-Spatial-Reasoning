"""Object-aware hypothesis generation and transformation rules for ARC tasks.

Implements object-level program hypotheses including:
- Object extraction (largest, smallest, unique, color-specific)
- Object filtering and canvas rendering
- Object sorting by area / position
- Object recoloring (conditional, size-based, containment-based)
- Object counting and cardinality-to-dimension mapping
- Object movement and alignment
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from src.data.models import ARCTask, Grid
from src.objects.dsl import (
    count_objects,
    extract_object,
    filter_by_area,
    filter_by_color,
    filter_by_position,
    recolor_objects,
    render_objects,
    select_adjacent,
    select_contained,
    select_largest,
    select_matching,
    select_smallest,
    select_unique,
    sort_by_area,
    sort_by_position,
    translate_objects,
)
from src.objects.segmentation import (
    ObjectInstance,
    SegmentationHypothesis,
    SegmentationStrategy,
    detect_background,
    segment_grid,
)
from src.transformations.base import Transformation


# --------------------------------------------------------------------------
# Object Transformation Wrappers (implementing Transformation interface)
# --------------------------------------------------------------------------


class ObjectExtractionTransformation(Transformation):
    """Extract a single object as a cropped Grid based on selection criterion."""

    def __init__(
        self,
        criterion: str,
        param: Any = None,
        strategy: SegmentationStrategy = SegmentationStrategy.MONOCHROMATIC_4,
    ) -> None:
        self.criterion = criterion  # 'largest', 'smallest', 'unique', 'color', 'position'
        self.param = param
        self.strategy = strategy

    def apply(self, grid: Grid) -> Grid:
        hyp = segment_grid(grid, strategy=self.strategy)
        objs = hyp.objects
        if not objs:
            return grid

        target: ObjectInstance | None = None
        if self.criterion == "largest":
            target = select_largest(objs)
        elif self.criterion == "smallest":
            target = select_smallest(objs)
        elif self.criterion == "unique":
            uniques = select_unique(objs)
            target = uniques[0] if uniques else None
        elif self.criterion == "color":
            filtered = filter_by_color(objs, int(self.param))
            target = filtered[0] if filtered else None
        elif self.criterion == "position":
            filtered = filter_by_position(objs, str(self.param), grid.shape)
            target = filtered[0] if filtered else None

        if target is None:
            return grid
        return extract_object(target)

    @property
    def name(self) -> str:
        return f"ObjectExtraction_{self.criterion}"

    @property
    def complexity(self) -> float:
        return 2.0

    @property
    def params(self) -> dict[str, Any]:
        return {"criterion": self.criterion, "param": self.param, "strategy": self.strategy.value}


class ObjectFilterAndRenderTransformation(Transformation):
    """Filter objects by property and render onto the output canvas."""

    def __init__(
        self,
        filter_type: str,
        filter_param: Any,
        output_shape_mode: str = "same",  # 'same', or explicit (h, w)
        background: int = 0,
        strategy: SegmentationStrategy = SegmentationStrategy.MONOCHROMATIC_4,
    ) -> None:
        self.filter_type = filter_type  # 'color_keep', 'color_remove', 'area_min', 'area_max', 'position'
        self.filter_param = filter_param
        self.output_shape_mode = output_shape_mode
        self.background = background
        self.strategy = strategy

    def apply(self, grid: Grid) -> Grid:
        hyp = segment_grid(grid, strategy=self.strategy, background=self.background)
        objs = hyp.objects

        filtered: list[ObjectInstance] = []
        if self.filter_type == "color_keep":
            filtered = filter_by_color(objs, int(self.filter_param))
        elif self.filter_type == "color_remove":
            filtered = [o for o in objs if o.color != int(self.filter_param)]
        elif self.filter_type == "area_min":
            filtered = filter_by_area(objs, min_area=int(self.filter_param))
        elif self.filter_type == "area_max":
            filtered = filter_by_area(objs, max_area=int(self.filter_param))
        elif self.filter_type == "position":
            filtered = filter_by_position(objs, str(self.filter_param), grid.shape)
        else:
            filtered = objs

        target_shape = grid.shape if self.output_shape_mode == "same" else self.output_shape_mode
        return render_objects(filtered, target_shape, background=self.background)

    @property
    def name(self) -> str:
        return f"ObjectFilterRender_{self.filter_type}"

    @property
    def complexity(self) -> float:
        return 2.2

    @property
    def params(self) -> dict[str, Any]:
        return {
            "filter_type": self.filter_type,
            "filter_param": self.filter_param,
            "strategy": self.strategy.value,
        }


class ObjectRecolorTransformation(Transformation):
    """Recolor objects matching a structural or relational condition."""

    def __init__(
        self,
        condition: str,
        target_color: int,
        param: Any = None,
        strategy: SegmentationStrategy = SegmentationStrategy.MONOCHROMATIC_4,
    ) -> None:
        self.condition = condition  # 'largest', 'smallest', 'by_color', 'contained'
        self.target_color = target_color
        self.param = param
        self.strategy = strategy

    def apply(self, grid: Grid) -> Grid:
        hyp = segment_grid(grid, strategy=self.strategy)
        objs = hyp.objects
        if not objs:
            return grid

        new_objs: list[ObjectInstance] = []
        if self.condition == "largest":
            largest = select_largest(objs)
            for o in objs:
                if largest and o.object_id == largest.object_id:
                    new_objs.append(recolor_objects([o], self.target_color)[0])
                else:
                    new_objs.append(o)
        elif self.condition == "smallest":
            smallest = select_smallest(objs)
            for o in objs:
                if smallest and o.object_id == smallest.object_id:
                    new_objs.append(recolor_objects([o], self.target_color)[0])
                else:
                    new_objs.append(o)
        elif self.condition == "by_color":
            src_color = int(self.param)
            for o in objs:
                if o.color == src_color:
                    new_objs.append(recolor_objects([o], self.target_color)[0])
                else:
                    new_objs.append(o)
        else:
            new_objs = objs

        return render_objects(new_objs, grid.shape, background=hyp.background)

    @property
    def name(self) -> str:
        return f"ObjectRecolor_{self.condition}"

    @property
    def complexity(self) -> float:
        return 2.3

    @property
    def params(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "target_color": self.target_color,
            "param": self.param,
        }


class ObjectCountingTransformation(Transformation):
    """Count objects and map cardinality to a structured output grid."""

    def __init__(
        self,
        count_target: str,  # 'all', 'by_color', 'unique_colors'
        target_color: int | None = None,
        output_format: str = "1xN_color",  # '1xN_color', 'Nx1_color', '1x1_count', 'bar_sorted'
        paint_color: int = 1,
        strategy: SegmentationStrategy = SegmentationStrategy.MONOCHROMATIC_4,
    ) -> None:
        self.count_target = count_target
        self.target_color = target_color
        self.output_format = output_format
        self.paint_color = paint_color
        self.strategy = strategy

    def apply(self, grid: Grid) -> Grid:
        hyp = segment_grid(grid, strategy=self.strategy)
        objs = hyp.objects

        if self.count_target == "all":
            k = len(objs)
        elif self.count_target == "by_color" and self.target_color is not None:
            k = len(filter_by_color(objs, self.target_color))
        elif self.count_target == "unique_colors":
            k = len(set(o.color for o in objs if o.color is not None))
        else:
            k = len(objs)

        # Minimum dimension of 1
        k = max(1, min(30, k))

        if self.output_format == "1xN_color":
            return Grid.from_list([[self.paint_color] * k])
        elif self.output_format == "Nx1_color":
            return Grid.from_list([[self.paint_color] for _ in range(k)])
        elif self.output_format == "1x1_count":
            # Direct color code = k
            color_val = min(9, max(0, k))
            return Grid.from_list([[color_val]])
        elif self.output_format == "bar_sorted":
            # Bar of colors sorted by count
            counts: dict[int, int] = {}
            for o in objs:
                if o.color is not None:
                    counts[o.color] = counts.get(o.color, 0) + 1
            sorted_colors = [c for c, _ in sorted(counts.items(), key=lambda x: x[1])]
            if not sorted_colors:
                sorted_colors = [self.paint_color]
            return Grid.from_list([sorted_colors])

        return Grid.from_list([[self.paint_color] * k])

    @property
    def name(self) -> str:
        return f"ObjectCounting_{self.count_target}_{self.output_format}"

    @property
    def complexity(self) -> float:
        return 2.5

    @property
    def params(self) -> dict[str, Any]:
        return {
            "count_target": self.count_target,
            "target_color": self.target_color,
            "output_format": self.output_format,
            "paint_color": self.paint_color,
        }


# --------------------------------------------------------------------------
# Object Candidate Generator
# --------------------------------------------------------------------------


class ObjectCandidateGenerator:
    """Generates object-centric transformation candidates for an ARC task."""

    def __init__(self, max_candidates: int = 500) -> None:
        self.max_candidates = max_candidates

    def generate(self, task: ARCTask) -> list[Transformation]:
        """Propose object-aware transformation hypotheses from task demonstrations."""
        candidates: list[Transformation] = []
        colors_used = sorted(list(task.unique_colors))
        train_inputs = task.all_train_inputs
        train_outputs = task.all_train_outputs

        first_in = train_inputs[0]
        first_out = train_outputs[0]
        is_size_preserving = task.is_size_preserving

        # Try multiple segmentation strategies
        strategies = [
            SegmentationStrategy.MONOCHROMATIC_4,
            SegmentationStrategy.MONOCHROMATIC_8,
            SegmentationStrategy.MULTICOLOR_4,
        ]

        for strat in strategies:
            # 1. Object Extraction Candidates
            if not is_size_preserving:
                for crit in ("largest", "smallest", "unique"):
                    candidates.append(ObjectExtractionTransformation(criterion=crit, strategy=strat))
                for c in colors_used:
                    if c != 0:
                        candidates.append(ObjectExtractionTransformation(criterion="color", param=c, strategy=strat))
                for pos in ("top", "bottom", "left", "right", "center"):
                    candidates.append(ObjectExtractionTransformation(criterion="position", param=pos, strategy=strat))

            # 2. Object Filtering + Canvas Rendering Candidates
            if is_size_preserving:
                for c in colors_used:
                    candidates.append(ObjectFilterAndRenderTransformation("color_keep", c, strategy=strat))
                    candidates.append(ObjectFilterAndRenderTransformation("color_remove", c, strategy=strat))
                for pos in ("top", "bottom", "left", "right", "center"):
                    candidates.append(ObjectFilterAndRenderTransformation("position", pos, strategy=strat))
                for area in (1, 2, 3, 4, 5, 6, 8, 9):
                    candidates.append(ObjectFilterAndRenderTransformation("area_min", area, strategy=strat))
                    candidates.append(ObjectFilterAndRenderTransformation("area_max", area, strategy=strat))

            # 3. Object Recoloring Candidates
            if is_size_preserving:
                for target_c in colors_used:
                    if target_c != 0:
                        candidates.append(ObjectRecolorTransformation("largest", target_c, strategy=strat))
                        candidates.append(ObjectRecolorTransformation("smallest", target_c, strategy=strat))
                        for src_c in colors_used:
                            if src_c != target_c:
                                candidates.append(
                                    ObjectRecolorTransformation("by_color", target_c, param=src_c, strategy=strat)
                                )

            # 4. Object Counting & Cardinality Mappings
            if not is_size_preserving:
                for paint_c in colors_used:
                    if paint_c != 0:
                        for fmt in ("1xN_color", "Nx1_color", "1x1_count", "bar_sorted"):
                            candidates.append(
                                ObjectCountingTransformation(
                                    count_target="all",
                                    output_format=fmt,
                                    paint_color=paint_c,
                                    strategy=strat,
                                )
                            )
                            candidates.append(
                                ObjectCountingTransformation(
                                    count_target="unique_colors",
                                    output_format=fmt,
                                    paint_color=paint_c,
                                    strategy=strat,
                                )
                            )
                            for target_c in colors_used:
                                if target_c != 0:
                                    candidates.append(
                                        ObjectCountingTransformation(
                                            count_target="by_color",
                                            target_color=target_c,
                                            output_format=fmt,
                                            paint_color=paint_c,
                                            strategy=strat,
                                        )
                                    )

        # Deduplicate candidates while preserving order
        unique_candidates: list[Transformation] = []
        seen = set()
        for cand in candidates:
            cand_repr = repr(cand)
            if cand_repr not in seen:
                seen.add(cand_repr)
                unique_candidates.append(cand)
                if len(unique_candidates) >= self.max_candidates:
                    break

        return unique_candidates
