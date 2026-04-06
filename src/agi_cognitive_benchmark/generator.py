from __future__ import annotations

from dataclasses import dataclass
import random
import string

from .baselines import solve_with_policy
from .models import Constraint, Packet, Scenario
from .solver import solve_scenario


TEAM_NAMES = ("alpha", "beta", "gamma")
PACKET_NOTES = (
    "legacy checklist carried over from an older shift",
    "current operator memo",
    "draft note from another team",
    "audit follow-up packet",
    "temporary workaround from staging",
    "approved line update",
)


@dataclass(frozen=True)
class DifficultySpec:
    name: str
    num_tasks: int
    num_base_constraints: int
    num_distractors: int
    min_move: int
    max_move: int
    families: tuple[str, ...]


DIFFICULTY_SPECS = {
    "easy": DifficultySpec(
        name="easy",
        num_tasks=5,
        num_base_constraints=3,
        num_distractors=2,
        min_move=1,
        max_move=2,
        families=("shield",),
    ),
    "medium": DifficultySpec(
        name="medium",
        num_tasks=6,
        num_base_constraints=4,
        num_distractors=3,
        min_move=2,
        max_move=3,
        families=("shield", "switch"),
    ),
    "hard": DifficultySpec(
        name="hard",
        num_tasks=7,
        num_base_constraints=5,
        num_distractors=4,
        min_move=3,
        max_move=5,
        families=("switch", "repair"),
    ),
}


def candidate_constraints(order: list[str]) -> list[Constraint]:
    constraints: list[Constraint] = []
    for left_index, left in enumerate(order):
        for right_index, right in enumerate(order):
            if left_index < right_index:
                constraints.append(Constraint("before", left, right))
            if abs(left_index - right_index) > 1:
                constraints.append(Constraint("not_adjacent", left, right))
    for position, task in enumerate(order, start=1):
        constraints.append(Constraint("fixed_position", task, position))
    unique: dict[tuple[str, str, str], Constraint] = {}
    for constraint in constraints:
        key = (constraint.kind, constraint.lhs, str(constraint.rhs))
        unique[key] = constraint
    return list(unique.values())


def build_base_constraints(baseline: list[str], count: int, rng: random.Random) -> list[Constraint]:
    pool = candidate_constraints(baseline)
    rng.shuffle(pool)
    selected: list[Constraint] = []
    seen_override_keys: set[tuple[str, str, str]] = set()
    for constraint in pool:
        override_key = override_group_key(constraint)
        if override_key in seen_override_keys:
            continue
        selected.append(constraint)
        seen_override_keys.add(override_key)
        if len(selected) == count:
            return selected
    return selected


def override_group_key(constraint: Constraint) -> tuple[str, str, str]:
    if constraint.kind in {"before", "not_adjacent"}:
        left, right = sorted((constraint.lhs, str(constraint.rhs)))
        return constraint.kind, left, right
    return constraint.kind, constraint.lhs, "*"


def shuffled_target(
    baseline: list[str],
    min_move: int,
    max_move: int,
    rng: random.Random,
) -> list[str]:
    attempts = 0
    while attempts < 500:
        attempts += 1
        target = baseline[:]
        swap_count = rng.randint(2, min(len(target), max_move + 1))
        positions = rng.sample(range(len(target)), k=swap_count)
        subset = [target[index] for index in positions]
        rng.shuffle(subset)
        for index, item in zip(positions, subset, strict=True):
            target[index] = item
        moved = sum(1 for left, right in zip(target, baseline) if left != right)
        if min_move <= moved <= max_move and target != baseline:
            return target
    raise RuntimeError("Failed to generate a target schedule with the requested move range.")


def baseline_violating_constraints(
    baseline: list[str],
    target: list[str],
    base_constraints: list[Constraint],
) -> list[Constraint]:
    baseline_set = {(item.kind, item.lhs, str(item.rhs)) for item in base_constraints}
    violating = []
    for constraint in candidate_constraints(target):
        if (constraint.kind, constraint.lhs, str(constraint.rhs)) in baseline_set:
            continue
        if constraint in base_constraints:
            continue
        positions = {task: index for index, task in enumerate(baseline)}
        if constraint.kind == "before" and positions[constraint.lhs] < positions[str(constraint.rhs)]:
            continue
        if constraint.kind == "not_adjacent" and abs(positions[constraint.lhs] - positions[str(constraint.rhs)]) > 1:
            continue
        if constraint.kind == "fixed_position" and positions[constraint.lhs] == int(constraint.rhs) - 1:
            continue
        violating.append(constraint)
    return violating


def make_packet(packet_id: str, team: str, status: str, constraint: Constraint, rng: random.Random) -> Packet:
    return Packet(
        packet_id=packet_id,
        team=team,
        status=status,
        constraint=constraint,
        note=rng.choice(PACKET_NOTES),
    )


def build_packets(
    family: str,
    active_team: str,
    baseline: list[str],
    target: list[str],
    base_constraints: list[Constraint],
    spec: DifficultySpec,
    rng: random.Random,
) -> list[Packet]:
    violating = baseline_violating_constraints(baseline, target, base_constraints)
    if len(violating) < 2:
        raise RuntimeError("Not enough baseline-violating constraints to generate packets.")

    packets: list[Packet] = []
    packet_counter = 1
    if family == "shield":
        applicable = rng.choice(violating)
        packets.append(make_packet(f"P{packet_counter}", active_team, "APPLY", applicable, rng))
        packet_counter += 1
        distractor_pool = [item for item in violating if item != applicable]
    elif family == "switch":
        switchable = [item for item in violating if item.kind in {"before", "fixed_position"}]
        if not switchable:
            raise RuntimeError("Switch scenarios require a reversible constraint.")
        final_constraint = rng.choice(switchable)
        if final_constraint.kind == "before":
            previous_constraint = Constraint("before", str(final_constraint.rhs), final_constraint.lhs)
        else:
            previous_constraint = Constraint(
                "fixed_position",
                final_constraint.lhs,
                1 if int(final_constraint.rhs) != 1 else len(baseline),
            )
        packets.append(make_packet(f"P{packet_counter}", active_team, "APPLY", previous_constraint, rng))
        packet_counter += 1
        packets.append(make_packet(f"P{packet_counter}", active_team, "APPLY", final_constraint, rng))
        packet_counter += 1
        distractor_pool = [item for item in violating if item != final_constraint]
    else:
        selected = rng.sample(violating, k=2)
        for constraint in selected:
            packets.append(make_packet(f"P{packet_counter}", active_team, "APPLY", constraint, rng))
            packet_counter += 1
        distractor_pool = [item for item in violating if item not in selected]

    while len(packets) < len([packet for packet in packets if packet.team == active_team and packet.status == "APPLY"]) + spec.num_distractors:
        candidate = rng.choice(distractor_pool or violating)
        distractor_team = rng.choice([team for team in TEAM_NAMES if team != active_team])
        distractor_status = rng.choice(["ARCHIVED", "DRAFT", "REVOKED"])
        if rng.random() < 0.5:
            packets.append(make_packet(f"P{packet_counter}", distractor_team, "APPLY", candidate, rng))
        else:
            packets.append(make_packet(f"P{packet_counter}", active_team, distractor_status, candidate, rng))
        packet_counter += 1

    rng.shuffle(packets)
    return [
        Packet(
            packet_id=f"P{index}",
            team=packet.team,
            status=packet.status,
            constraint=packet.constraint,
            note=packet.note,
        )
        for index, packet in enumerate(packets, start=1)
    ]


def scenario_is_interesting(scenario: Scenario) -> bool:
    gold = solve_scenario(scenario)
    if gold is None:
        return False
    if not (DIFFICULTY_SPECS[scenario.difficulty].min_move <= gold.moved_tasks <= DIFFICULTY_SPECS[scenario.difficulty].max_move):
        return False
    if solve_with_policy(scenario, "ignore_updates").final_schedule == gold.final_schedule:
        return False
    if solve_with_policy(scenario, "same_team_any_status").final_schedule == gold.final_schedule:
        return False
    if solve_with_policy(scenario, "apply_all_packets").final_schedule == gold.final_schedule:
        return False
    return True


def generate_scenario(
    seed: int,
    scenario_index: int,
    difficulty: str,
    family: str | None = None,
) -> tuple[Scenario, object]:
    spec = DIFFICULTY_SPECS[difficulty]
    rng = random.Random(seed + scenario_index * 7919)
    attempts = 0
    while attempts < 1000:
        attempts += 1
        selected_family = family or rng.choice(spec.families)
        tasks = list(string.ascii_uppercase[: spec.num_tasks])
        baseline = rng.sample(tasks, k=len(tasks))
        base_constraints = build_base_constraints(baseline, spec.num_base_constraints, rng)
        target = shuffled_target(baseline, spec.min_move, spec.max_move, rng)
        active_team = rng.choice(TEAM_NAMES)
        try:
            packets = build_packets(selected_family, active_team, baseline, target, base_constraints, spec, rng)
        except RuntimeError:
            continue
        scenario = Scenario(
            scenario_id=f"goalshield-{difficulty}-{scenario_index:04d}",
            seed=seed,
            difficulty=difficulty,
            family=selected_family,
            active_team=active_team,
            tasks=tasks,
            baseline_schedule=baseline,
            base_constraints=base_constraints,
            packets=packets,
        )
        solution = solve_scenario(scenario)
        if solution is None:
            continue
        if scenario_is_interesting(scenario):
            return scenario, solution
    raise RuntimeError(f"Failed to generate an interesting {difficulty} scenario after {attempts} attempts.")
