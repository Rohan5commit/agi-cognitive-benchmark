from __future__ import annotations

import argparse
from pathlib import Path

from agi_cognitive_benchmark.dataset import generate_benchmark_dataset, save_dataset_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate GoalShield benchmark datasets.")
    parser.add_argument("--seed", type=int, default=20260406)
    parser.add_argument("--easy", type=int, default=16)
    parser.add_argument("--medium", type=int, default=16)
    parser.add_argument("--hard", type=int, default=16)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/goalshield_v1_full.jsonl"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = generate_benchmark_dataset(
        seed=args.seed,
        counts={"easy": args.easy, "medium": args.medium, "hard": args.hard},
    )
    save_dataset_records(records, args.output)
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
