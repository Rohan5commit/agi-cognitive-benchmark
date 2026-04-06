from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .generator import DIFFICULTY_SPECS, generate_scenario
from .models import Scenario, Solution


def generate_benchmark_dataset(
    seed: int,
    counts: dict[str, int] | None = None,
) -> list[dict[str, object]]:
    requested = counts or {name: 12 for name in DIFFICULTY_SPECS}
    records: list[dict[str, object]] = []
    offset = 0
    for difficulty, total in requested.items():
        families = DIFFICULTY_SPECS[difficulty].families
        for local_index in range(total):
            family = families[local_index % len(families)]
            scenario, solution = generate_scenario(
                seed=seed,
                scenario_index=offset + local_index,
                difficulty=difficulty,
                family=family,
            )
            records.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "difficulty": scenario.difficulty,
                    "family": scenario.family,
                    "prompt": scenario.render_prompt(),
                    "scenario_json": scenario.to_json(),
                    "solution_json": solution.to_json(),
                    "gold_schedule": " ".join(solution.final_schedule),
                    "gold_packets": " ".join(solution.applicable_packets),
                    "moved_tasks": solution.moved_tasks,
                }
            )
        offset += total
    return records


def save_dataset_records(records: Iterable[dict[str, object]], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")


def load_dataset_records(path: str | Path) -> list[dict[str, object]]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_records_dataframe(records: Iterable[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(list(records))


def rehydrate(record: dict[str, object]) -> tuple[Scenario, Solution]:
    return (
        Scenario.from_json(str(record["scenario_json"])),
        Solution.from_json(str(record["solution_json"])),
    )
