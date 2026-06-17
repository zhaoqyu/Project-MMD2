#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect saved QEvasion run metrics.")
    parser.add_argument("--run-dir", required=True, help="Run directory created by scripts/train.py.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    metrics_files = sorted((run_dir / "artifacts").rglob("*_metrics.json"))
    if not metrics_files:
        raise FileNotFoundError(f"No metrics files found under {run_dir / 'artifacts'}")
    for path in metrics_files:
        with path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        compact = {
            key: value
            for key, value in metrics.items()
            if key.endswith("f1") or key.endswith("accuracy") or key.endswith("count") or key.endswith("rate")
        }
        print(f"\n{path.relative_to(run_dir)}")
        print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()

