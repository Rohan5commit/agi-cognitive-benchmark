from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and push the GoalShield Kaggle notebook.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/kaggle/goalshield-executive-functions-benchmark"),
    )
    parser.add_argument("--public", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    build_command = [
        "uv",
        "run",
        "--extra",
        "kaggle",
        "python",
        "scripts/build_kaggle_notebook.py",
        "--output-dir",
        str(args.output_dir),
    ]
    if args.public:
        build_command.append("--public")
    subprocess.run(build_command, check=True)
    subprocess.run(["kaggle", "kernels", "push", "-p", str(args.output_dir)], check=True)


if __name__ == "__main__":
    main()
