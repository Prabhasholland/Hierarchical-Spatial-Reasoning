import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_dataset
from src.evaluation.harness import evaluate_dataset
from src.solvers.object_solver import RuleBasedObjectSolver_v1
from src.solvers.rule_based import RuleBasedSearchSolver

tasks = load_dataset("data/ARC-AGI-1/data/training")
b = evaluate_dataset(RuleBasedSearchSolver(), tasks)
o = evaluate_dataset(RuleBasedObjectSolver_v1(), tasks)

print(f"Base solved: {b['tasks_solved']}/400 ({b['task_level_accuracy_pct']:.2f}%)")
print(f"Obj solved:  {o['tasks_solved']}/400 ({o['task_level_accuracy_pct']:.2f}%)")
new_ids = sorted(list(set(o['solved_task_ids']) - set(b['solved_task_ids'])))
print(f"Newly solved tasks ({len(new_ids)}): {new_ids}")
for tid in new_ids:
    print(f"  {tid}: {o['solved_task_rules'].get(tid)}")
