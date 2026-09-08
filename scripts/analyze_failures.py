"""Systematic failure analysis script for ARC baseline v1.

Classifies all failed tasks from the baseline evaluation into precise
failure categories, extracts quantitative metrics, identifies representative
examples, renders failure visualization images, and generates failure_analysis.json.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.loader import load_dataset, load_task
from src.data.models import ARCTask, Grid
from src.data.visualizer import plot_task
from src.solvers.rule_based import RuleBasedSearchSolver


def label_components(mask: np.ndarray, connectivity: int = 4) -> tuple[np.ndarray, int]:
    """Pure numpy/python connected component labeling."""
    h, w = mask.shape
    labeled = np.zeros((h, w), dtype=np.int32)
    current_label = 0
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        offsets.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])

    for r in range(h):
        for c in range(w):
            if mask[r, c] and labeled[r, c] == 0:
                current_label += 1
                queue = [(r, c)]
                labeled[r, c] = current_label
                while queue:
                    curr_r, curr_c = queue.pop()
                    for dr, dc in offsets:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            if mask[nr, nc] and labeled[nr, nc] == 0:
                                labeled[nr, nc] = current_label
                                queue.append((nr, nc))
    return labeled, current_label


def count_connected_components(grid: Grid, background: int = 0) -> tuple[int, list[int]]:
    """Count connected components (objects) and their sizes."""
    arr = grid.to_numpy()
    mask = (arr != background)
    if not np.any(mask):
        return 0, []
    labeled_arr, num_features = label_components(mask)
    sizes = [int(np.sum(labeled_arr == i)) for i in range(1, num_features + 1)]
    return num_features, sizes


def count_single_color_objects(grid: Grid, background: int = 0) -> int:
    """Count connected components grouped by color."""
    arr = grid.to_numpy()
    total_objs = 0
    for c in range(10):
        if c == background:
            continue
        mask = (arr == c)
        if np.any(mask):
            _, num_features = label_components(mask)
            total_objs += num_features
    return total_objs


def has_enclosed_holes(grid: Grid, background: int = 0) -> bool:
    """Check if the grid contains background pixels completely enclosed by foreground."""
    arr = grid.to_numpy()
    h, w = arr.shape
    fg_mask = (arr != background)
    bg_mask = (arr == background)
    if not np.any(bg_mask) or not np.any(fg_mask):
        return False
    # Label background components
    labeled_bg, num_bg = label_components(bg_mask)
    # Check if any bg component does not touch the outer boundary
    for i in range(1, num_bg + 1):
        comp_mask = (labeled_bg == i)
        # Check boundary touch
        touches_top = np.any(comp_mask[0, :])
        touches_bottom = np.any(comp_mask[h - 1, :])
        touches_left = np.any(comp_mask[:, 0])
        touches_right = np.any(comp_mask[:, w - 1])
        if not (touches_top or touches_bottom or touches_left or touches_right):
            return True
    return False


def classify_task_failure(task: ARCTask, solver_details: dict[str, Any]) -> tuple[str, list[str]]:
    """Classify a failed task into primary failure category and secondary attributes."""
    tags = []
    
    # 1. Ambiguity / Generalization Error
    if solver_details.get("solved_on_train", False):
        return "ambiguity_between_hypotheses", ["overfitted_on_train", "spurious_consistency"]

    train_inputs = task.all_train_inputs
    train_outputs = task.all_train_outputs
    test_inputs = task.all_test_inputs
    test_outputs = task.all_test_outputs

    # Check size behavior
    is_size_preserving = task.is_size_preserving
    out_shapes = [p.output.shape for p in task.train]
    in_shapes = [p.input.shape for p in task.train]

    # Check counting signatures
    # (e.g. output is 1x1 color representing count, or 1xK grid with size equal to number of objects/colors)
    is_counting = False
    for p in task.train:
        num_in_objs, obj_sizes = count_connected_components(p.input)
        num_colors = len(p.input.unique_colors - {0})
        if p.output.shape in ((1, 1), (1, num_in_objs), (num_in_objs, 1), (1, num_colors), (num_colors, 1)):
            is_counting = True
            break
        # Check if output dimension equals count of specific elements
        if p.output.height in (num_in_objs, num_colors) or p.output.width in (num_in_objs, num_colors):
            is_counting = True
            break

    if is_counting and not is_size_preserving:
        return "counting_failure", ["count_to_dimension", "cardinality_inference"]

    # Check object extraction failure (e.g. output is one specific object from input)
    is_object_extraction = False
    for p in task.train:
        num_objs, _ = count_connected_components(p.input)
        if num_objs > 1 and p.output.height <= p.input.height and p.output.width <= p.input.width:
            # Check if output colors are a strict subset of input
            if p.output.unique_colors.issubset(p.input.unique_colors):
                is_object_extraction = True
                break

    if is_object_extraction and not is_size_preserving:
        return "object_detection_failure", ["multi_object_filtering", "subgrid_selection"]

    # Check spatial relationships / topology / containment
    has_holes = any(has_enclosed_holes(p.input) or has_enclosed_holes(p.output) for p in task.train)
    if has_holes and is_size_preserving:
        return "spatial_relationship_failure", ["topology", "containment", "flood_fill"]

    # Check multi-object movement / gravity / alignment
    multi_objs = any(count_connected_components(p.input)[0] >= 2 for p in task.train)
    if multi_objs and is_size_preserving:
        # Check if color distribution is preserved but coordinates moved
        color_hist_preserved = all(
            p.input.color_counts() == p.output.color_counts() for p in task.train
        )
        if color_hist_preserved:
            return "spatial_relationship_failure", ["object_movement", "collision", "rearrangement"]
        else:
            return "multi_step_reasoning_failure", ["object_interaction", "conditional_recoloring"]

    # Check color relationship failure (contextual recoloring, adjacency recoloring)
    input_colors = set().union(*(p.input.unique_colors for p in task.train))
    output_colors = set().union(*(p.output.unique_colors for p in task.train))
    if is_size_preserving and (output_colors - input_colors):
        # New colors introduced conditionally
        return "color_relationship_failure", ["new_color_introduction", "contextual_recoloring"]

    # Check symmetry / pattern completion
    if is_size_preserving and any(p.input.height == p.input.width for p in task.train):
        return "symmetry_failure", ["partial_symmetry", "pattern_completion"]

    # Check dimension change that is not covered
    if not is_size_preserving:
        return "transformation_composition_failure", ["unmodeled_dimension_mapping", "scale_crop_composite"]

    # Default size-preserving
    return "insufficient_transformation_library", ["local_cellular_automata", "line_drawing", "path_finding"]


def run_failure_analysis() -> None:
    eval_dir = PROJECT_ROOT / "data" / "ARC-AGI-1" / "data" / "evaluation"
    print(f"Loading evaluation tasks from {eval_dir}...")
    tasks = load_dataset(eval_dir)
    print(f"Loaded {len(tasks)} tasks.")

    solver = RuleBasedSearchSolver(fallback_to_identity=True)
    print("Running solver and diagnosing failure causes...")

    category_tasks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    solved_count = 0
    total_failures = 0

    for task_id, task in tasks.items():
        details = solver.solve_with_details(task)
        
        # Verify test correctness
        all_test_correct = True
        for t_idx, test_pair in enumerate(task.test):
            if test_pair.output is None:
                all_test_correct = False
                break
            pred = details["predictions"][t_idx]
            if pred.to_list() != test_pair.output.to_list():
                all_test_correct = False
                break

        if all_test_correct:
            solved_count += 1
            continue

        total_failures += 1
        category, tags = classify_task_failure(task, details)
        category_tasks[category].append({
            "task_id": task_id,
            "num_train": task.num_train,
            "is_size_preserving": task.is_size_preserving,
            "colors_used": sorted(list(task.unique_colors)),
            "train_input_shapes": [p.input.shape for p in task.train],
            "train_output_shapes": [p.output.shape for p in task.train],
            "tags": tags,
            "chosen_rule_on_train": details.get("chosen_rule"),
        })

    # Compute category statistics
    category_summary = {}
    for cat, failed_list in sorted(category_tasks.items(), key=lambda x: len(x[1]), reverse=True):
        count = len(failed_list)
        pct = (count / total_failures * 100) if total_failures > 0 else 0.0
        representative_ids = [item["task_id"] for item in failed_list[:5]]
        
        # Common characteristics
        size_pres_count = sum(1 for item in failed_list if item["is_size_preserving"])
        avg_colors = sum(len(item["colors_used"]) for item in failed_list) / count
        
        category_summary[cat] = {
            "failure_count": count,
            "percentage_of_failures": round(pct, 2),
            "representative_examples": representative_ids,
            "characteristics": {
                "size_preserving_percentage": round(size_pres_count / count * 100, 1),
                "avg_colors_used": round(avg_colors, 2),
                "common_tags": [t for t, _ in Counter(tag for item in failed_list for tag in item["tags"]).most_common(3)],
            },
            "task_details": failed_list,
        }

    output_data = {
        "analysis_date": "2026-08-21",
        "benchmark_dataset": "ARC-AGI-1/data/evaluation",
        "total_tasks": len(tasks),
        "tasks_solved": solved_count,
        "total_failures": total_failures,
        "failure_categories": category_summary,
    }

    # Save results JSON
    json_path = PROJECT_ROOT / "results" / "failure_analysis.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved failure analysis JSON to: {json_path.resolve()}")

    # Render visualizations of representative tasks for each failure category
    img_dir = PROJECT_ROOT / "results" / "failure_visualizations"
    img_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating representative task visualizations...")
    saved_images = {}
    for cat, cat_info in category_summary.items():
        rep_ids = cat_info["representative_examples"]
        if rep_ids:
            rep_id = rep_ids[0]
            task_obj = tasks[rep_id]
            img_path = img_dir / f"{cat}_{rep_id}.png"
            plot_task(task_obj, save_path=img_path)
            saved_images[cat] = {
                "task_id": rep_id,
                "image_path": str(img_path.relative_to(PROJECT_ROOT)),
            }
            print(f"  [{cat}] Saved visualization for {rep_id} -> {img_path.name}")

    output_data["representative_visualizations"] = saved_images
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print("\nFailure breakdown across 398 failed tasks:")
    for cat, info in category_summary.items():
        print(f"  - {cat:35s}: {info['failure_count']:3d} tasks ({info['percentage_of_failures']:5.1f}%)")


if __name__ == "__main__":
    run_failure_analysis()
