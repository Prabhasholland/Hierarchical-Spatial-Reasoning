"""Script to run the full ARC-AGI-2 submission generation pipeline locally."""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_dataset
from src.submission.submission_generator import generate_submission_dict
from src.submission.validation import validate_submission


def main():
    parser = argparse.ArgumentParser(description="Generate ARC-AGI-2 submission.json")
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default=str(PROJECT_ROOT / "data" / "ARC-AGI-2" / "data" / "evaluation"),
        help="Path to the test dataset directory or JSON file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="submission.json",
        help="Path to save the generated submission.json"
    )
    parser.add_argument(
        "--report", 
        type=str, 
        default=str(PROJECT_ROOT / "results" / "submission_run_report.json"),
        help="Path to save the execution report"
    )
    
    args = parser.parse_args()
    data_path = Path(args.data_dir)
    
    print(f"Loading data from: {data_path}")
    if data_path.is_dir():
        tasks = load_dataset(data_path, recursive=True)
    else:
        tasks = load_dataset(data_path.parent, recursive=False)
        # Filter if a specific file was provided but load_dataset loads the whole dir
        if data_path.suffix == '.json':
            tasks = {k: v for k, v in tasks.items() if k == data_path.stem}
            
    print(f"Loaded {len(tasks)} tasks.")
    
    print("Generating submission...")
    submission, report = generate_submission_dict(tasks)
    
    print("Validating submission...")
    try:
        validate_submission(submission, expected_tasks=tasks)
        print("Validation PASSED.")
    except Exception as e:
        print(f"Validation FAILED: {e}")
        sys.exit(1)
        
    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(submission, f)
    print(f"Saved submission to: {out_path.resolve()}")
    
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved execution report to: {report_path.resolve()}")
    
    print(f"Summary: {report['successful_tasks']} successful, {report['failed_tasks']} failed/fallback, {report['runtime_seconds']}s runtime.")

if __name__ == "__main__":
    main()
