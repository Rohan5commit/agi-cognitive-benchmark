from agi_cognitive_benchmark.dataset import DATASET_COLUMNS, build_records_dataframe
from agi_cognitive_benchmark.baselines import solve_with_policy
from agi_cognitive_benchmark.generator import DIFFICULTY_SPECS, generate_scenario


def test_generator_hits_requested_family() -> None:
    scenario, solution = generate_scenario(
        seed=20260406,
        scenario_index=7,
        difficulty="medium",
        family="switch",
    )
    assert scenario.family == "switch"
    assert solution is not None
    spec = DIFFICULTY_SPECS["medium"]
    assert spec.min_move <= solution.moved_tasks <= spec.max_move


def test_generator_produces_nontrivial_cases() -> None:
    scenario, solution = generate_scenario(
        seed=20260406,
        scenario_index=19,
        difficulty="hard",
        family="repair",
    )
    assert solution is not None
    assert solve_with_policy(scenario, "ignore_updates").final_schedule != solution.final_schedule
    assert solve_with_policy(scenario, "apply_all_packets").final_schedule != solution.final_schedule


def test_build_records_dataframe_preserves_schema_for_empty_datasets() -> None:
    frame = build_records_dataframe([])
    assert frame.empty
    assert frame.columns.tolist() == DATASET_COLUMNS
