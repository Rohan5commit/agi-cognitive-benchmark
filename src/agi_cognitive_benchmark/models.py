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
        base_lines = "\n".join(
            f"{index}. {constraint.describe()}"
            for index, constraint in enumerate(self.base_constraints, start=1)
        )
        packet_lines = "\n".join(
            (
                f"[{packet.packet_id}] team={packet.team} status={packet.status} :: "
                f"{packet.constraint.describe()} | note={packet.note}"
            )
            for packet in self.packets
        )
        tasks = ", ".join(self.tasks)
        baseline = " ".join(self.baseline_schedule)
        return (
            f"You are auditing a schedule for team {self.active_team}.\n\n"
            "Follow this policy exactly:\n"
            f"- Start from the baseline schedule: {baseline}\n"
            f"- Only packets with team={self.active_team} and status=APPLY are binding.\n"
            "- If two binding packets affect the same rule key, the later packet overrides the earlier one.\n"
            "- Produce the valid final schedule that moves as few tasks as possible from the baseline schedule.\n"
            "- If multiple schedules tie on moved-task count, choose the lexicographically smallest schedule.\n\n"
            f"Tasks: {tasks}\n\n"
            "Base rules:\n"
            f"{base_lines}\n\n"
            "Update packets:\n"
            f"{packet_lines}\n\n"
            "Respond in JSON with these fields:\n"
            "- applicable_packets: list of packet ids that are binding\n"
            "- final_schedule: list of task ids in order\n"
            "- moved_tasks: integer count of positions changed relative to the baseline schedule\n"
            "- confidence: integer from 0 to 100\n"
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
