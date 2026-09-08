"""Generate 05_hierarchical_reasoning_ablation.ipynb notebook."""

import json
from pathlib import Path

notebook_content = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 05. Research Hypothesis 3: Hierarchical Sub-Goal Decomposition\n",
                "\n",
                "**ARC Prize 2026 Research Project**  \n",
                "This notebook evaluates whether property-guided hierarchical sub-goal decomposition solves multi-step ARC transformations ($k \\ge 3$) more efficiently and accurately than blind transformation composition.\n",
                "\n",
                "### Search Strategies Evaluated:\n",
                "- **A. Depth-2 Existing Solver** (`RuleBasedSpatialObjectSolver_v1`)\n",
                "- **B. Blind Depth-3 Search**\n",
                "- **C. Blind Depth-4 Search**\n",
                "- **D. Hierarchical Depth-3 Search**\n",
                "- **E. Hierarchical Depth-4 Search**\n"
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
                "from src.reasoning.state import GridState\n",
                "from src.reasoning.state_difference import StateDifference\n",
                "from src.reasoning.subgoals import propose_subgoals\n",
                "\n",
                "print(\"Hierarchical reasoning modules loaded successfully!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Load Experimental Results"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "results_file = project_root / \"results\" / \"hierarchical_reasoning_v1.json\"\n",
                "\n",
                "with open(results_file, \"r\", encoding=\"utf-8\") as f:\n",
                "    results = json.load(f)\n",
                "\n",
                "eval_split = results[\"eval_split\"]\n",
                "train_split = results[\"train_split\"]\n",
                "targeted_split = results[\"targeted_split\"]\n",
                "\n",
                "print(f\"{'Search Strategy':<38} | {'Eval Acc':<10} | {'Train Acc':<10} | {'Targeted Acc':<12}\")\n",
                "print(\"-\" * 78)\n",
                "for name in eval_split.keys():\n",
                "    ev_acc = eval_split[name]['task_level_accuracy_pct']\n",
                "    tr_acc = train_split[name]['task_level_accuracy_pct']\n",
                "    tg_acc = targeted_split[name]['task_level_accuracy_pct']\n",
                "    print(f\"{name:<38} | {ev_acc:>6.2f}%    | {tr_acc:>6.2f}%    | {tg_acc:>8.2f}%\")"
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

out_path = Path(r"C:\Users\DELL\.gemini\antigravity\scratch\arc-reasoning-agent\notebooks\05_hierarchical_reasoning_ablation.ipynb")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=2)

print(f"Notebook 05 generated at: {out_path}")
