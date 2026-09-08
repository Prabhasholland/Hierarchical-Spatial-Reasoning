"""Command line utility to validate a submission.json file."""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.submission.validation import validate_submission


def main():
    parser = argparse.ArgumentParser(description="Validate an ARC-AGI-2 submission.json file")
    parser.add_argument("file", type=str, help="Path to the submission.json file")
    args = parser.parse_args()
    
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
        
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            submission = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format: {e}")
        sys.exit(1)
        
    print(f"Validating {file_path.resolve()}...")
    try:
        validate_submission(submission)
        print("Validation PASSED. The file conforms to ARC-AGI-2 submission structure.")
        print(f"Total tasks in submission: {len(submission)}")
    except Exception as e:
        print(f"Validation FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
