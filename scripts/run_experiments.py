#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multiple QEvasion configs sequentially.")
    parser.add_argument("configs", nargs="+", help="Config YAML files.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for config in args.configs:
        cmd = [sys.executable, str(ROOT / "scripts" / "train.py"), "--config", config]
        if args.dry_run:
            cmd.append("--dry-run")
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

