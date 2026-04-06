from __future__ import annotations

from typing import Iterable

from .models import PlanAnswer, Scenario, Solution
from .solver import schedule_is_valid


def _normalize_schedule(items: Iterable[str]) -> list[str]:
    return [str(item).strip().upper() for item in items]


def _normalize_packets(items: Iterable[str]) -> list[str]:
    return [str(item).strip().upper() for item in items]


def _packet_f1(predicted: list[str], gold: list[str]) -> tuple[float, float, float]:
    predicted_set = set(predicted)
    gold_set = set(gold)
    if not predicted_set and not gold_set:
        return 1.0, 1.0, 1.0
    if not predicted_set:
        return 0.0, 0.0, 0.0
    true_positive = len(predicted_set & gold_set)
    precision = true_positive / len(predicted_set)
    recall = true_positive / len(gold_set) if gold_set else 1.0
    if precision + recall == 0:
        return precision, recall, 0.0
    return precision, recall, 2 * precision * recall / (precision + recall)


def score_plan_answer(
    scenario: Scenario,
    solution: Solution,
    answer: PlanAnswer | None,
) -> dict[str, float | int | str]:
    if answer is None:
        return {
            "scenario_id": scenario.scenario_id,
            "difficulty": scenario.difficulty,
            "family": scenario.family,
            "schedule_exact": 0.0,
            "schedule_valid": 0.0,
            "position_accuracy": 0.0,
            "packet_precision": 0.0,
            "packet_recall": 0.0,
            "packet_f1": 0.0,
            "moved_exact": 0.0,
            "moved_abs_error": float(solution.moved_tasks),
            "composite": 0.0,
        }

    predicted_schedule = _normalize_schedule(answer.final_schedule)
    predicted_packets = _normalize_packets(answer.applicable_packets)
    gold_schedule = _normalize_schedule(solution.final_schedule)
    gold_packets = _normalize_packets(solution.applicable_packets)

    schedule_exact = float(predicted_schedule == gold_schedule)
    valid = float(
        len(predicted_schedule) == len(scenario.tasks)
        and sorted(predicted_schedule) == sorted(scenario.tasks)
        and schedule_is_valid(predicted_schedule, solution.final_constraints)
    )
    position_accuracy = sum(
        1 for left, right in zip(predicted_schedule, gold_schedule) if left == right
    ) / len(gold_schedule)
    precision, recall, packet_f1 = _packet_f1(predicted_packets, gold_packets)
    moved_exact = float(answer.moved_tasks == solution.moved_tasks)
    moved_abs_error = abs(int(answer.moved_tasks) - solution.moved_tasks)
    composite = round(
        0.55 * schedule_exact
        + 0.2 * position_accuracy
        + 0.15 * packet_f1
        + 0.1 * moved_exact,
        6,
    )
    return {
        "scenario_id": scenario.scenario_id,
        "difficulty": scenario.difficulty,
        "family": scenario.family,
        "schedule_exact": schedule_exact,
        "schedule_valid": valid,
        "position_accuracy": position_accuracy,
        "packet_precision": precision,
        "packet_recall": recall,
        "packet_f1": packet_f1,
        "moved_exact": moved_exact,
        "moved_abs_error": moved_abs_error,
        "composite": composite,
    }
