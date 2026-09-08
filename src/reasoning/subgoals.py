"""Sub-Goal Generation module for ARC planning.

Proposes intermediate target sub-goals based on state differences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.reasoning.state import GridState
from src.reasoning.state_difference import StateDifference


@dataclass
class SubGoal:
    """Represents a targeted intermediate sub-goal in hierarchical planning."""
    goal_type: str
    desired_property_change: str
    target_value: Any
    applicable_primitives: list[str] = field(default_factory=list)
    verification_criteria: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_type": self.goal_type,
            "desired_property_change": self.desired_property_change,
            "target_value": self.target_value,
            "applicable_primitives": self.applicable_primitives,
            "verification_criteria": self.verification_criteria,
        }


def propose_subgoals(src_state: GridState, dst_state: GridState) -> list[SubGoal]:
    """Propose deterministic sub-goals from source state to target state."""
    diff = StateDifference.compute(src_state, dst_state)
    subgoals: list[SubGoal] = []

    # 1. Dimension change / Subgrid extraction
    if not diff.is_size_preserving:
        subgoals.append(
            SubGoal(
                goal_type="change_dimensions",
                desired_property_change="grid_shape",
                target_value=(dst_state.height, dst_state.width),
                applicable_primitives=["CropBoundingBox", "ObjectExtraction", "CropFixed"],
                verification_criteria={"height": dst_state.height, "width": dst_state.width},
            )
        )

    # 2. Cavity filling
    if diff.is_cavity_filling or src_state.num_enclosed_regions > 0:
        subgoals.append(
            SubGoal(
                goal_type="fill_enclosed_region",
                desired_property_change="num_enclosed_regions",
                target_value=dst_state.num_enclosed_regions,
                applicable_primitives=["FillEnclosed", "flood_fill_op"],
                verification_criteria={"max_enclosed": dst_state.num_enclosed_regions},
            )
        )

    # 3. Object reduction / filtering
    if diff.is_object_filtering or diff.object_count_delta < 0:
        subgoals.append(
            SubGoal(
                goal_type="reduce_object_count",
                desired_property_change="object_count",
                target_value=dst_state.object_count,
                applicable_primitives=["ObjectFilterRender", "filter_by_color", "filter_by_area"],
                verification_criteria={"target_object_count": dst_state.object_count},
            )
        )

    # 4. Color substitution / recoloring
    if diff.is_recoloring or diff.removed_colors or diff.added_colors:
        subgoals.append(
            SubGoal(
                goal_type="change_color",
                desired_property_change="color_histogram",
                target_value=dst_state.color_histogram,
                applicable_primitives=["ColorSubstitution", "ColorReplace", "ObjectRecolor"],
                verification_criteria={"num_colors": dst_state.num_colors},
            )
        )

    # 5. Symmetry increase
    if diff.symmetry_delta_h > 0.1 or diff.symmetry_delta_v > 0.1:
        subgoals.append(
            SubGoal(
                goal_type="increase_symmetry",
                desired_property_change="symmetry_score",
                target_value=(dst_state.horizontal_symmetry, dst_state.vertical_symmetry),
                applicable_primitives=["SymmetryCompletion", "HorizontalFlip", "VerticalFlip"],
                verification_criteria={"min_h_sym": dst_state.horizontal_symmetry},
            )
        )

    # Default general transformation sub-goal
    subgoals.append(
        SubGoal(
            goal_type="general_transformation",
            desired_property_change="exact_match",
            target_value=dst_state,
            applicable_primitives=["ALL"],
            verification_criteria={},
        )
    )

    return subgoals
