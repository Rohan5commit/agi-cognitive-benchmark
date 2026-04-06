from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .models import Packet, Scenario
from .solver import is_applicable_packet, solve_scenario


PacketPolicy = Callable[[Packet, Scenario], bool]


def ignore_updates(packet: Packet, scenario: Scenario) -> bool:
    return False


def apply_all_packets(packet: Packet, scenario: Scenario) -> bool:
    return True


def same_team_any_status(packet: Packet, scenario: Scenario) -> bool:
    return packet.team == scenario.active_team


def latest_visible_key(packet: Packet, scenario: Scenario) -> bool:
    return packet is scenario.packets[-1]


def gold_policy(packet: Packet, scenario: Scenario) -> bool:
    return is_applicable_packet(packet, scenario.active_team)


POLICIES: dict[str, PacketPolicy] = {
    "gold": gold_policy,
    "ignore_updates": ignore_updates,
    "apply_all_packets": apply_all_packets,
    "same_team_any_status": same_team_any_status,
    "latest_visible_key": latest_visible_key,
}


@dataclass(frozen=True)
class PolicyOutcome:
    policy: str
    applicable_packets: list[str]
    final_schedule: list[str] | None
    moved_tasks: int | None


def solve_with_policy(scenario: Scenario, policy_name: str) -> PolicyOutcome:
    solution = solve_scenario(scenario, predicate=lambda packet: POLICIES[policy_name](packet, scenario))
    if solution is None:
        return PolicyOutcome(policy=policy_name, applicable_packets=[], final_schedule=None, moved_tasks=None)
    return PolicyOutcome(
        policy=policy_name,
        applicable_packets=solution.applicable_packets,
        final_schedule=solution.final_schedule,
        moved_tasks=solution.moved_tasks,
    )
