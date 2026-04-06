from __future__ import annotations

from itertools import permutations
from typing import Callable, Iterable

from .models import Constraint, Packet, Scenario, Solution


def constraint_key(constraint: Constraint) -> tuple[str, str, str]:
    if constraint.kind in {"before", "not_adjacent"}:
        left, right = sorted((constraint.lhs, str(constraint.rhs)))
        return constraint.kind, left, right
    return constraint.kind, constraint.lhs, str(constraint.rhs)


def is_applicable_packet(packet: Packet, active_team: str) -> bool:
    return packet.team == active_team and packet.status == "APPLY"


def merge_constraints(
    base_constraints: Iterable[Constraint],
    packets: Iterable[Packet],
    predicate: Callable[[Packet], bool],
) -> tuple[list[Constraint], list[str]]:
    merged = {constraint_key(item): item for item in base_constraints}
    applied: list[str] = []
    for packet in packets:
        if predicate(packet):
            merged[constraint_key(packet.constraint)] = packet.constraint
            applied.append(packet.packet_id)
    return list(merged.values()), applied


def schedule_is_valid(schedule: list[str], constraints: Iterable[Constraint]) -> bool:
    positions = {task: index for index, task in enumerate(schedule)}
    for constraint in constraints:
        if constraint.kind == "before":
            if positions[constraint.lhs] >= positions[str(constraint.rhs)]:
                return False
        elif constraint.kind == "not_adjacent":
            if abs(positions[constraint.lhs] - positions[str(constraint.rhs)]) == 1:
                return False
        elif constraint.kind == "fixed_position":
            if positions[constraint.lhs] != int(constraint.rhs) - 1:
                return False
    return True


def hamming_distance(left: list[str], right: list[str]) -> int:
    return sum(1 for a, b in zip(left, right) if a != b)


def enumerate_valid_schedules(tasks: list[str], constraints: Iterable[Constraint]) -> list[list[str]]:
    valid = []
    constraints = list(constraints)
    for schedule in permutations(tasks):
        candidate = list(schedule)
        if schedule_is_valid(candidate, constraints):
            valid.append(candidate)
    return valid


def choose_best_schedule(valid_schedules: list[list[str]], baseline: list[str]) -> list[str]:
    ranked = sorted(
        valid_schedules,
        key=lambda schedule: (hamming_distance(schedule, baseline), tuple(schedule)),
    )
    return ranked[0]


def solve_scenario(
    scenario: Scenario,
    predicate: Callable[[Packet], bool] | None = None,
) -> Solution | None:
    packet_filter = predicate or (lambda packet: is_applicable_packet(packet, scenario.active_team))
    final_constraints, applied_packets = merge_constraints(
        scenario.base_constraints,
        scenario.packets,
        packet_filter,
    )
    valid_schedules = enumerate_valid_schedules(scenario.tasks, final_constraints)
    if not valid_schedules:
        return None
    best_schedule = choose_best_schedule(valid_schedules, scenario.baseline_schedule)
    return Solution(
        scenario_id=scenario.scenario_id,
        applicable_packets=applied_packets,
        final_constraints=final_constraints,
        final_schedule=best_schedule,
        moved_tasks=hamming_distance(best_schedule, scenario.baseline_schedule),
        valid_schedule_count=len(valid_schedules),
    )
