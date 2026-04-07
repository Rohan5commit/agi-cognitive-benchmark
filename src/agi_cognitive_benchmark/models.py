from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any, Literal


ConstraintKind = Literal["before", "not_adjacent", "fixed_position"]
PacketStatus = Literal["APPLY", "ARCHIVED", "DRAFT", "REVOKED"]


@dataclass(frozen=True)
class Constraint:
    kind: ConstraintKind
    lhs: str
    rhs: str | int

    def describe(self) -> str:
        if self.kind == "before":
            return f"{self.lhs} must be before {self.rhs}"
        if self.kind == "not_adjacent":
            return f"{self.lhs} cannot be adjacent to {self.rhs}"
        return f"{self.lhs} must be in position {self.rhs}"

    def compact(self) -> str:
        if self.kind == "before":
            return f"{self.lhs}<{self.rhs}"
        if self.kind == "not_adjacent":
            return f"{self.lhs}!{self.rhs}"
        return f"{self.lhs}@{self.rhs}"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Constraint":
        return cls(
            kind=payload["kind"],
            lhs=payload["lhs"],
            rhs=payload["rhs"],
        )


@dataclass(frozen=True)
class Packet:
    packet_id: str
    team: str
    status: PacketStatus
    constraint: Constraint
    note: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Packet":
        return cls(
            packet_id=payload["packet_id"],
            team=payload["team"],
            status=payload["status"],
            constraint=Constraint.from_dict(payload["constraint"]),
            note=payload["note"],
        )

    def compact(self) -> str:
        return f"{self.packet_id}|{self.team}|{self.status}|{self.constraint.compact()}"


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    seed: int
    difficulty: str
    family: str
    active_team: str
    tasks: list[str]
    baseline_schedule: list[str]
    base_constraints: list[Constraint]
    packets: list[Packet]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Scenario":
        return cls(
            scenario_id=payload["scenario_id"],
            seed=payload["seed"],
            difficulty=payload["difficulty"],
            family=payload["family"],
            active_team=payload["active_team"],
            tasks=list(payload["tasks"]),
            baseline_schedule=list(payload["baseline_schedule"]),
            base_constraints=[Constraint.from_dict(item) for item in payload["base_constraints"]],
            packets=[Packet.from_dict(item) for item in payload["packets"]],
        )

    @classmethod
    def from_json(cls, payload: str) -> "Scenario":
        return cls.from_dict(json.loads(payload))

    def render_prompt(self) -> str:
        tasks = " ".join(self.tasks)
        baseline = " ".join(self.baseline_schedule)
        base_rules = "; ".join(constraint.compact() for constraint in self.base_constraints)
        packets = "; ".join(packet.compact() for packet in self.packets)
        return (
            "GoalShield.\n"
            f"team={self.active_team}\n"
            f"tasks={tasks}\n"
            f"baseline={baseline}\n"
            f"base_rules={base_rules}\n"
            f"packets={packets}\n"
            f"Use only {self.active_team} APPLY packets. Later packet on same rule key overrides earlier.\n"
            "Return the valid final schedule with minimum moved_tasks vs baseline; tie -> lexicographically smallest schedule.\n"
            "JSON only: applicable_packets, final_schedule, moved_tasks, confidence.\n"
            "Rule syntax: A<B before, A!B not-adjacent, A@3 fixed-pos."
        )


@dataclass(frozen=True)
class Solution:
    scenario_id: str
    applicable_packets: list[str]
    final_constraints: list[Constraint]
    final_schedule: list[str]
    moved_tasks: int
    valid_schedule_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Solution":
        return cls(
            scenario_id=payload["scenario_id"],
            applicable_packets=list(payload["applicable_packets"]),
            final_constraints=[Constraint.from_dict(item) for item in payload["final_constraints"]],
            final_schedule=list(payload["final_schedule"]),
            moved_tasks=payload["moved_tasks"],
            valid_schedule_count=payload["valid_schedule_count"],
        )

    @classmethod
    def from_json(cls, payload: str) -> "Solution":
        return cls.from_dict(json.loads(payload))


@dataclass
class PlanAnswer:
    applicable_packets: list[str]
    final_schedule: list[str]
    moved_tasks: int
    confidence: int
