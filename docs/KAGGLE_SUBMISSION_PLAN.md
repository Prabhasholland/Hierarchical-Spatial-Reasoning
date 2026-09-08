# Kaggle ARC-AGI-2 Submission Plan

## 1. Current solver entry point
The current best solver is `RuleBasedHierarchicalSolver_v1` located in `src/solvers/hierarchical_solver.py`.

## 2. Required imports
- `src.data.models`
- `src.solvers.hierarchical_solver`
- Dependencies across `src.objects`, `src.spatial`, `src.reasoning`, `src.topology`, `src.transformations`

## 3. Required dependencies
The current codebase runs entirely on Standard Python + `numpy`.
No external complex libraries (like PyTorch, OpenCV) are needed. We just need `numpy`.

## 4. Dataset assumptions
In Kaggle, the test datasets are usually provided as a single `.json` file containing all tasks (e.g., `arc-agi_test_challenges.json`), OR a directory of `.json` files. We will assume the Kaggle path `/kaggle/input/arc-prize-2024/arc-agi_test_challenges.json` (as standard for recent ARC competitions) or a generic JSON loader if a directory is provided.

## 5. Prediction interface
We need a clean adapter that runs:
```python
def solve_task(task: ARCTask) -> list[dict[str, list[list[int]]]]:
    # calls RuleBasedHierarchicalSolver_v1
    # handles exceptions
    # returns list of {"attempt_1": [...], "attempt_2": [...]}
```

## 6. Files needed for Kaggle
We need a single Kaggle Notebook (`notebooks/06_arc_agi_2_submission.ipynb`) that effectively packages the `/src/` folder (or imports it if we upload the repository as a Kaggle Dataset, which is standard practice). For simplicity and standard Kaggle repo-based submissions, we assume the repo is attached as a Kaggle Dataset, and the notebook adds it to `sys.path`.

## 7. Potential compatibility problems
- Maximum submission runtime (usually 12 hours). We need to enforce a timeout per task if search takes too long.
- Network access is disabled in Kaggle code competitions.
- Hardcoded absolute paths must be replaced with relative/Kaggle paths.

## 8. Recommended Kaggle notebook structure
1. Import `sys` and append repo dataset path.
2. Load JSON from Kaggle competition data.
3. Instantiate `RuleBasedHierarchicalSolver_v1`.
4. Iterate over tasks, run solver.
5. Format predictions with fallbacks.
6. Validate format.
7. Save `/kaggle/working/submission.json`.
