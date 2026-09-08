"""Command-line utility for inspecting and visualizing ARC tasks.

Usage examples:
    python scripts/inspect_task.py --task 007bbfb7
    python scripts/inspect_task.py --random --dataset data/ARC-AGI-1/data/training
    python scripts/inspect_task.py --task 007bbfb7 --ascii
    python scripts/inspect_task.py --task 007bbfb7 --save results/task_007bbfb7.png
    python scripts/inspect_task.py --stats --dataset data/ARC-AGI-1/data/training
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import (
    get_dataset_statistics,
    get_random_task,
    list_task_ids,
    load_dataset,
    load_task,
)
from src.data.visualizer import plot_task, render_task_ascii


def find_task_file(task_id: str, dataset_dir: Path | None = None) -> Path:
    """Find the path to a task file by ID."""
    # Direct path provided
    direct_path = Path(task_id)
    if direct_path.exists() and direct_path.is_file():
        return direct_path
    
    # Check within specific dataset directory if provided
    if dataset_dir and dataset_dir.exists():
        direct = dataset_dir / f"{task_id}.json"
        if direct.exists():
            return direct
        matches = list(dataset_dir.rglob(f"{task_id}.json"))
        if matches:
            return matches[0]

    # Search standard data paths
    data_root = PROJECT_ROOT / "data"
    if data_root.exists():
        matches = list(data_root.rglob(f"{task_id}.json"))
        if matches:
            return matches[0]
        matches = list(data_root.rglob(f"*{task_id}*.json"))
        if matches:
            return matches[0]

    raise FileNotFoundError(f"Could not locate task file for ID: '{task_id}'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and visualize ARC tasks.")
    parser.add_argument("--task", "-t", type=str, help="Task ID or path to JSON file")
    parser.add_argument("--dataset", "-d", type=str, help="Dataset directory path")
    parser.add_argument("--random", "-r", action="store_true", help="Pick a random task")
    parser.add_argument("--ascii", "-a", action="store_true", help="Print task in ASCII format")
    parser.add_argument("--save", "-s", type=str, help="Save plot image to file path")
    parser.add_argument("--stats", action="store_true", help="Compute and print dataset statistics")
    parser.add_argument("--list", "-l", action="store_true", help="List all task IDs in dataset")

    args = parser.parse_args()

    dataset_path = Path(args.dataset) if args.dataset else None

    # Handle dataset stats
    if args.stats:
        target_dir = dataset_path or (PROJECT_ROOT / "data")
        print(f"Loading tasks from {target_dir} to compute statistics...")
        tasks = load_dataset(target_dir, recursive=True)
        if not tasks:
            print("No tasks found.")
            sys.exit(1)
        stats = get_dataset_statistics(tasks)
        print("\n=== ARC Dataset Statistics ===")
        print(json.dumps(stats, indent=2))
        return

    # Handle listing tasks
    if args.list:
        target_dir = dataset_path or (PROJECT_ROOT / "data")
        task_ids = list_task_ids(target_dir, recursive=True)
        print(f"Found {len(task_ids)} tasks in {target_dir}:")
        for tid in task_ids:
            print(f"  {tid}")
        return

    # Determine target task
    task = None
    if args.random:
        target_dir = dataset_path or (PROJECT_ROOT / "data")
        task_ids = list_task_ids(target_dir, recursive=True)
        if not task_ids:
            print(f"No task JSON files found in {target_dir}")
            sys.exit(1)
        import random
        chosen_id = random.choice(task_ids)
        task_file = find_task_file(chosen_id, target_dir)
        task = load_task(task_file)
    elif args.task:
        task_file = find_task_file(args.task, dataset_path)
        task = load_task(task_file)
    else:
        parser.print_help()
        sys.exit(0)

    # Print summary
    print(f"\nTask ID: {task.task_id}")
    print(f"Train pairs: {task.num_train} | Test pairs: {task.num_test}")
    print(f"Colors used: {sorted(list(task.unique_colors))}")
    print(f"Size preserving: {task.is_size_preserving}")
    print(f"Train input shapes: {[p.input.shape for p in task.train]}")
    print(f"Train output shapes: {[p.output.shape for p in task.train]}")
    print(f"Test input shapes: {[p.input.shape for p in task.test]}")

    if args.ascii:
        print("\n" + render_task_ascii(task))

    if args.save:
        save_path = Path(args.save)
        plot_task(task, save_path=save_path)
        print(f"Saved visualization to: {save_path.resolve()}")
    elif not args.ascii:
        # Default behavior: if matplotlib can render interactively, plot and show
        import matplotlib.pyplot as plt
        plot_task(task)
        plt.show()


if __name__ == "__main__":
    main()
