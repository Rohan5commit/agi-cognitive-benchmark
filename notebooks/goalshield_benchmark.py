# %% [markdown]
# # GoalShield
#
# GoalShield is an executive-functions benchmark for interrupted plan repair.
# It measures whether a model can:
#
# 1. preserve a baseline goal state,
# 2. inhibit distractor packets,
# 3. apply the right rule updates, and
# 4. minimally repair a schedule instead of over-replanning.

# %%
!pip install -q https://github.com/Kaggle/kaggle-benchmarks/archive/refs/heads/ci.zip
!pip install -q https://github.com/Rohan5commit/agi-cognitive-benchmark/archive/refs/heads/main.zip

# %%
from dataclasses import dataclass
from io import StringIO
import json

import pandas as pd

import kaggle_benchmarks as kbench

from agi_cognitive_benchmark.dataset import build_records_dataframe, generate_benchmark_dataset
from agi_cognitive_benchmark.metrics import score_plan_answer
from agi_cognitive_benchmark.models import PlanAnswer, Scenario, Solution

# %%
BENCHMARK_RECORDS = generate_benchmark_dataset(
    seed=20260406,
    counts={"easy": 18, "medium": 18, "hard": 18},
)
BENCHMARK_DF = build_records_dataframe(BENCHMARK_RECORDS)
EVAL_DF = BENCHMARK_DF[["prompt", "scenario_json", "solution_json"]].copy()
BENCHMARK_DF.head(3)

# %%
SYSTEM_PROMPT = """
You are solving a schedule-repair task.
Always follow the explicit packet policy in the user prompt.
Return only the JSON object required by the schema.
"""


@kbench.task(name="goalshield_item", store_task=False)
def goalshield_item(llm, prompt: str, scenario_json: str, solution_json: str) -> dict:
    scenario = Scenario.from_json(scenario_json)
    solution = Solution.from_json(solution_json)
    with kbench.chats.new(f"goalshield-{scenario.scenario_id}"):
        kbench.system.send(SYSTEM_PROMPT)
        answer = llm.prompt(prompt, schema=PlanAnswer)
    return score_plan_answer(scenario, solution, answer)


@kbench.task(name="goalshield_benchmark")
def goalshield_benchmark(llm, evaluation_df_json: str) -> tuple[float, float]:
    evaluation_df = pd.read_json(StringIO(evaluation_df_json), orient="records")
    with kbench.client.enable_cache():
        runs = goalshield_item.evaluate(
            llm=[llm],
            evaluation_data=evaluation_df,
            max_attempts=1,
            retry_delay=15,
            timeout=240,
            n_jobs=4,
            remove_run_files=True,
            stop_condition=lambda collected_runs: len(collected_runs) == evaluation_df.shape[0],
        )
    eval_df = runs.as_dataframe()
    return (
        float(eval_df.result.str.get("composite").mean()),
        float(eval_df.result.str.get("composite").std(ddof=0)),
    )

# %%
benchmark_payload = EVAL_DF.to_json(orient="records")
leaderboard_run = goalshield_benchmark.run(kbench.llm, benchmark_payload)
leaderboard_run

# %%
preferred_models = [
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
    "anthropic/claude-sonnet-4",
    "anthropic/claude-opus-4",
    "openai/gpt-4.1",
    "openai/gpt-4.5",
    "openai/o3-mini-high",
    "meta/llama-4-maverick",
]
available_model_names = [name for name in preferred_models if name in kbench.llms]
evaluation_models = [kbench.llms[name] for name in available_model_names] or [kbench.llm]
evaluation_models

# %%
with kbench.client.enable_cache():
    model_runs = goalshield_item.evaluate(
        llm=evaluation_models,
        evaluation_data=EVAL_DF,
        max_attempts=1,
        retry_delay=15,
        timeout=240,
        n_jobs=4,
        remove_run_files=True,
        stop_condition=lambda collected_runs: len(collected_runs) == len(evaluation_models) * EVAL_DF.shape[0],
    )
model_eval_df = model_runs.as_dataframe()
model_summary = (
    model_eval_df.groupby("llm_name")
    .agg(
        composite=("result", lambda series: float(series.str.get("composite").mean())),
        schedule_exact=("result", lambda series: float(series.str.get("schedule_exact").mean())),
        packet_f1=("result", lambda series: float(series.str.get("packet_f1").mean())),
    )
    .sort_values("composite", ascending=False)
)
model_summary

# %%
model_summary.to_csv("/kaggle/working/goalshield_model_summary.csv")
BENCHMARK_DF.to_csv("/kaggle/working/goalshield_dataset.csv", index=False)
print("Saved /kaggle/working/goalshield_model_summary.csv")
print("Saved /kaggle/working/goalshield_dataset.csv")

# %%
BENCHPRESS_MAP = {
    "google/gemini-2.5-flash": "gemini-2.5-flash",
    "google/gemini-2.5-pro": "gemini-2.5-pro",
    "google/gemini-3-flash": "gemini-3-flash",
    "google/gemini-3-pro": "gemini-3-pro",
    "anthropic/claude-sonnet-4": "claude-sonnet-4",
    "anthropic/claude-opus-4": "claude-opus-4",
    "openai/gpt-4.1": "gpt-4.1",
    "openai/gpt-4.5": "gpt-4.5",
    "openai/o3-mini-high": "o3-mini-high",
    "meta/llama-4-maverick": "llama-4-maverick",
    "meta/llama-4-scout": "llama-4-scout",
}

benchpress_scores = {
    BENCHPRESS_MAP[name]: score * 100.0
    for name, score in model_summary["composite"].items()
    if name in BENCHPRESS_MAP
}
with open("/kaggle/working/goalshield_benchpress_input.json", "w", encoding="utf-8") as handle:
    json.dump(benchpress_scores, handle, indent=2)
print("Saved /kaggle/working/goalshield_benchpress_input.json")

# %%
%choose goalshield_benchmark
