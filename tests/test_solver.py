from agi_cognitive_benchmark.models import Constraint, Packet, Scenario
from agi_cognitive_benchmark.solver import solve_scenario


def test_solver_respects_latest_applicable_packet_override() -> None:
    scenario = Scenario(
        scenario_id="unit-switch",
        seed=1,
        difficulty="medium",
        family="switch",
        active_team="alpha",
        tasks=["A", "B", "C", "D"],
        baseline_schedule=["A", "B", "C", "D"],
        base_constraints=[
            Constraint("before", "A", "D"),
            Constraint("before", "B", "C"),
        ],
        packets=[
            Packet("P1", "alpha", "APPLY", Constraint("before", "D", "A"), "first override"),
            Packet("P2", "beta", "APPLY", Constraint("fixed_position", "C", 1), "wrong team"),
            Packet("P3", "alpha", "APPLY", Constraint("before", "A", "D"), "latest wins"),
            Packet("P4", "alpha", "DRAFT", Constraint("fixed_position", "B", 4), "inactive"),
        ],
    )
    solution = solve_scenario(scenario)
    assert solution is not None
    assert solution.applicable_packets == ["P1", "P3"]
    assert solution.final_schedule == ["A", "B", "C", "D"]
    assert solution.moved_tasks == 0


def test_solver_uses_minimal_repair_objective() -> None:
    scenario = Scenario(
        scenario_id="unit-repair",
        seed=2,
        difficulty="hard",
        family="repair",
        active_team="alpha",
        tasks=["A", "B", "C", "D", "E"],
        baseline_schedule=["A", "B", "C", "D", "E"],
        base_constraints=[
            Constraint("before", "A", "E"),
            Constraint("not_adjacent", "B", "E"),
        ],
        packets=[
            Packet("P1", "alpha", "APPLY", Constraint("fixed_position", "C", 5), "move C late"),
            Packet("P2", "alpha", "APPLY", Constraint("before", "D", "B"), "move D earlier"),
            Packet("P3", "alpha", "DRAFT", Constraint("fixed_position", "A", 4), "ignore draft"),
        ],
    )
    solution = solve_scenario(scenario)
    assert solution is not None
    assert solution.final_schedule == ["A", "E", "D", "B", "C"]
    assert solution.moved_tasks == 4
