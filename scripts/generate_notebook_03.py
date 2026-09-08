"""Generate the 03_object_reasoning_ablation.ipynb notebook."""

import json
from pathlib import Path

notebook_content = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 03. Research Hypothesis 1: Object-Centric Representation Ablation\n",
                "\n",
                "**ARC Prize 2026 Research Project**  \n",
                "This notebook evaluates whether parsing ARC grids into structured object entities and relational graphs improves exact-match generalization over the grid-level baseline.\n",
                "\n",
                "### Ablation Protocol:\n",
                "- **Experiment A**: Baseline v1 (Grid-level rigid transforms, colors, fractals)\n",
                "- **Experiment B**: Baseline + Object Extraction (Cropping bounding box of salient objects)\n",
                "- **Experiment C**: Baseline + Object Attributes (Filtering & recoloring by area/color/position)\n",
                "- **Experiment D**: Baseline + Object Counting (Cardinality mapping to output grid)\n",
                "- **Experiment E**: Complete Object-Aware Solver (`RuleBasedObjectSolver_v1`)\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%matplotlib inline\n",
                "import json\n",
                "import sys\n",
                "from pathlib import Path\n",
                "import matplotlib.pyplot as plt\n",
                "import numpy as np\n",
                "\n",
                "project_root = Path.cwd().parent if Path.cwd().name == \"notebooks\" else Path.cwd()\n",
                "if str(project_root) not in sys.path:\n",
                "    sys.path.insert(0, str(project_root))\n",
                "\n",
                "from src.data.loader import load_task\n",
                "from src.data.visualizer import plot_task\n",
                "from src.objects.graph import ObjectGraph\n",
                "from src.objects.segmentation import segment_grid, segment_grid_multi\n",
                "\n",
                "print(\"Modules imported successfully!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Load Controlled Ablation Results"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "results_file = project_root / \"results\" / \"object_reasoning_v1.json\"\n",
                "\n",
                "with open(results_file, \"r\", encoding=\"utf-8\") as f:\n",
                "    ablation_results = json.load(f)\n",
                "\n",
                "print(f\"{'Experiment':<42} | {'Solved':<8} | {'Task Acc':<10} | {'Pair Acc':<10} | {'Runtime'}\")\n",
                "print(\"-\" * 80)\n",
                "for name, res in ablation_results.items():\n",
                "    print(\n",
                "        f\"{name:<42} | {res['tasks_solved']:<8} | \"\n",
                "        f\"{res['task_level_accuracy_pct']:>6.2f}%    | \"\n",
                "        f\"{res['test_pair_accuracy_pct']:>6.2f}%    | \"\n",
                "        f\"{res['total_runtime_seconds']:.2f}s\"\n",
                "    )"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Comparative Ablation Accuracy Chart"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "experiments = list(ablation_results.keys())\n",
                "short_labels = [\"Base\", \"Base+Extr\", \"Base+Attr\", \"Base+Count\", \"Full Object\"]\n",
                "task_accs = [ablation_results[k][\"task_level_accuracy_pct\"] for k in experiments]\n",
                "pair_accs = [ablation_results[k][\"test_pair_accuracy_pct\"] for k in experiments]\n",
                "\n",
                "x = np.arange(len(short_labels))\n",
                "width = 0.35\n",
                "\n",
                "fig, ax = plt.subplots(figsize=(10, 5))\n",
                "r1 = ax.bar(x - width/2, task_accs, width, label=\"Task-Level Accuracy (%)\", color=\"#0074D9\", edgecolor=\"black\")\n",
                "r2 = ax.bar(x + width/2, pair_accs, width, label=\"Test-Pair Accuracy (%)\", color=\"#2ECC40\", edgecolor=\"black\")\n",
                "\n",
                "ax.set_ylabel(\"Accuracy (%)\")\n",
                "ax.set_title(\"ARC Hypothesis 1: Ablation Study Comparison\", fontweight=\"bold\")\n",
                "ax.set_xticks(x)\n",
                "ax.set_xticklabels(short_labels)\n",
                "ax.legend()\n",
                "ax.grid(axis=\"y\", linestyle=\"--\", alpha=0.5)\n",
                "\n",
                "for r in [r1, r2]:\n",
                "    for bar in r:\n",
                "        h = bar.get_height()\n",
                "        ax.text(bar.get_x() + bar.get_width()/2., h + 0.05, f\"{h:.2f}%\", ha=\"center\", va=\"bottom\", fontsize=9, fontweight=\"bold\")\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Visualizing Newly Solved Tasks & Discovered Object Graphs"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "base_ids = set(ablation_results[\"Experiment A (Baseline v1)\"][\"solved_task_ids\"])\n",
                "full_ids = set(ablation_results[\"Experiment E (Complete Object Solver v1)\"][\"solved_task_ids\"])\n",
                "new_ids = sorted(list(full_ids - base_ids))\n",
                "\n",
                "print(f\"Newly Solved Tasks by Object Solver: {new_ids}\\n\")\n",
                "\n",
                "for tid in new_ids[:3]:\n",
                "    task_files = list((project_root / \"data\").rglob(f\"{tid}.json\"))\n",
                "    if task_files:\n",
                "        task = load_task(task_files[0])\n",
                "        rule = ablation_results[\"Experiment E (Complete Object Solver v1)\"][\"solved_task_rules\"].get(tid)\n",
                "        print(f\"=== Task {tid} | Discovered Rule: {rule} ===\")\n",
                "        plot_task(task)\n",
                "        plt.show()\n",
                "        \n",
                "        # Inspect object graph of first training input\n",
                "        graph = ObjectGraph.from_grid(task.train[0].input)\n",
                "        print(\"\\n--- Extracted Relational Object Graph (Train Input 0) ---\")\n",
                "        print(graph)\n",
                "        print(\"\\n\")"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = Path(r"C:\Users\DELL\.gemini\antigravity\scratch\arc-reasoning-agent\notebooks\03_object_reasoning_ablation.ipynb")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=2)

print(f"Notebook 03 generated at: {out_path}")
