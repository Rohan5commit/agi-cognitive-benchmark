from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluating_agi.benchpress import check_novelty


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run BenchPress novelty analysis on model scores.")
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--name", default="GoalShield")
    parser.add_argument("--output", type=Path, default=Path("results/benchpress_report.json"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scores = json.loads(args.scores.read_text(encoding="utf-8"))
    report = check_novelty(scores, name=args.name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
