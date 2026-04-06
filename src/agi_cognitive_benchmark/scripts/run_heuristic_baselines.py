from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from agi_cognitive_benchmark.baselines import POLICIES, solve_with_policy
from agi_cognitive_benchmark.dataset import load_dataset_records, rehydrate
from agi_cognitive_benchmark.metrics import score_plan_answer
from agi_cognitive_benchmark.models import PlanAnswer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate heuristic baselines on GoalShield.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/goalshield_v1_full.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/heuristic_baselines.csv"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = load_dataset_records(args.dataset)
    rows: list[dict[str, object]] = []
    for record in records:
        scenario, solution = rehydrate(record)
        for policy_name in POLICIES:
            outcome = solve_with_policy(scenario, policy_name)
            if outcome.final_schedule is None or outcome.moved_tasks is None:
                continue
            answer = PlanAnswer(
                applicable_packets=outcome.applicable_packets,
                final_schedule=outcome.final_schedule,
                moved_tasks=outcome.moved_tasks,
                confidence=0,
            )
            scored = score_plan_answer(scenario, solution, answer)
            scored["policy"] = policy_name
            rows.append(scored)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Wrote {len(rows)} scored rows to {args.output}")


if __name__ == "__main__":
    main()
