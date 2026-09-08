"""Download ARC-AGI datasets.

Usage:
    python scripts/download_data.py

Downloads ARC-AGI-1 and ARC-AGI-2 public datasets from GitHub
into the data/ directory.
"""

import subprocess
import sys
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data"

DATASETS = {
    "ARC-AGI-1": "https://github.com/fchollet/ARC-AGI.git",
    "ARC-AGI-2": "https://github.com/arcprize/ARC-AGI-2.git",
}


def download_dataset(name: str, url: str) -> None:
    """Clone a dataset repository into the data directory."""
    target = DATA_DIR / name
    if target.exists():
        print(f"[SKIP] {name} already exists at {target}")
        return
    
    print(f"[DOWNLOAD] Cloning {name} from {url}...")
    subprocess.run(
        ["git", "clone", "--depth", "1", url, str(target)],
        check=True,
    )
    print(f"[DONE] {name} downloaded to {target}")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in DATASETS.items():
        try:
            download_dataset(name, url)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to download {name}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
