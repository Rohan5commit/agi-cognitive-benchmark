# %% [markdown]
# # GoalShield
#
# GoalShield is an executive-functions benchmark for interrupted plan repair.
# This Kaggle notebook is configured as a competition run, not a smoke test:
#
# 1. build three solver-backed dataset slices,
# 2. spend the hosted Benchmarks quota on a broad frontier-model sweep,
# 3. export per-difficulty and per-family artifacts for the writeup, and
# 4. keep a live progress log in `/kaggle/working/goalshield_progress.json`.

# %%
!pip install -q https://github.com/Kaggle/kaggle-benchmarks/archive/refs/heads/ci.zip
!pip install -q https://github.com/Rohan5commit/agi-cognitive-benchmark/archive/refs/heads/main.zip

# %%
from io import StringIO
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import pandas as pd

import kaggle_benchmarks as kbench

from agi_cognitive_benchmark.dataset import build_records_dataframe, generate_benchmark_dataset
from agi_cognitive_benchmark.metrics import score_plan_answer
from agi_cognitive_benchmark.models import PlanAnswer, Scenario, Solution

WORKDIR = Path("/kaggle/working")
PROGRESS_PATH = WORKDIR / "goalshield_progress.json"
RUN_PLAN_PATH = WORKDIR / "goalshield_run_plan.json"
RUNTIME_INFO_PATH = WORKDIR / "goalshield_runtime_info.json"
AVAILABLE_MODELS_PATH = WORKDIR / "goalshield_available_models.json"
FAILURES_PATH = WORKDIR / "goalshield_failures.json"

MODEL_PRIORITY = [
    "google/gemini-3-pro",
    "google/gemini-3-flash",
    "google/gemini-2.5-pro",
    "google/gemini-2.5-flash",
    "anthropic/claude-opus-4",
    "anthropic/claude-sonnet-4",
    "openai/gpt-4.5",
    "openai/gpt-4.1",
    "openai/o3",
    "openai/o3-mini-high",
    "meta/llama-4-maverick",
    "meta/llama-4-scout",
    "meta/llama-3.1-405b",
    "meta/llama-3.1-70b",
    "xai/grok-3-beta",
    "deepseek/deepseek-r1",
    "qwen/qwen3-32b",
    "mistral/magistral-medium",
]
EXCLUDE_MODEL_TERMS = (
    "embedding",
    "vision",
    "image",
    "audio",
    "speech",
    "transcribe",
    "tts",
    "rerank",
    "moderation",
    "safety",
    "instruct-embed",
)
DATASET_SPECS = {
    "primary": {
        "seed": 20260406,
        "counts": {"easy": 24, "medium": 24, "hard": 24},
        "description": "Main benchmark slice used for the Kaggle benchmark entity.",
    },
    "probe": {
        "seed": 20260407,
        "counts": {"easy": 10, "medium": 10, "hard": 10},
        "description": "Breadth sweep slice for additional model coverage.",
    },
    "holdout": {
        "seed": 20260408,
        "counts": {"easy": 8, "medium": 8, "hard": 8},
        "description": "Robustness slice for top primary models.",
    },
}
BENCHPRESS_MAP = {
    "google/gemini-2.5-flash": "gemini-2.5-flash",
    "google/gemini-2.5-pro": "gemini-2.5-pro",
    "google/gemini-3-flash": "gemini-3-flash",
    "google/gemini-3-pro": "gemini-3-pro",
    "anthropic/claude-sonnet-4": "claude-sonnet-4",
    "anthropic/claude-opus-4": "claude-opus-4",
    "openai/gpt-4.1": "gpt-4.1",
    "openai/gpt-4.5": "gpt-4.5",
    "openai/o3": "o3",
    "openai/o3-mini-high": "o3-mini-high",
    "meta/llama-4-maverick": "llama-4-maverick",
    "meta/llama-4-scout": "llama-4-scout",
}
PROGRESS_LOG: list[dict[str, object]] = []
FAILURES: list[dict[str, object]] = []


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def append_progress(
    *,
    stage: str,
    status: str,
    detail: str,
    completed_units: int,
    total_units: int,
    extra: dict[str, object] | None = None,
) -> None:
    entry = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage": stage,
        "status": status,
        "detail": detail,
        "completed_units": completed_units,
        "total_units": total_units,
    }
    if extra:
        entry["extra"] = extra
    PROGRESS_LOG.append(entry)
    write_json(PROGRESS_PATH, PROGRESS_LOG)
    print(f"[{completed_units}/{total_units}] {stage} {status}: {detail}")


def detect_runtime_info() -> dict[str, object]:
    info: dict[str, object] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "default_model": getattr(kbench.llm, "name", str(kbench.llm)),
    }
    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        info["torch_error"] = f"{type(exc).__name__}: {exc}"
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            info["nvidia_smi"] = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except FileNotFoundError:
        info["nvidia_smi"] = "not_installed"
    return info


def is_candidate_text_model(model_name: str) -> bool:
    lowered = model_name.lower()
    return "/" in model_name and not any(term in lowered for term in EXCLUDE_MODEL_TERMS)


def ordered_candidate_models(available_names: list[str]) -> list[str]:
    filtered = [name for name in available_names if is_candidate_text_model(name)]
    ordered: list[str] = []
    seen: set[str] = set()
    for name in MODEL_PRIORITY:
        if name in filtered and name not in seen:
            ordered.append(name)
            seen.add(name)

    provider_caps = {
        "google": 3,
        "anthropic": 2,
        "openai": 4,
        "meta": 3,
        "xai": 1,
        "deepseek": 2,
        "qwen": 2,
        "mistral": 2,
    }
    provider_counts: dict[str, int] = {}
    for name in sorted(filtered):
        if name in seen:
            continue
        provider = name.split("/", maxsplit=1)[0]
        if provider_counts.get(provider, 0) >= provider_caps.get(provider, 1):
            continue
        ordered.append(name)
        seen.add(name)
        provider_counts[provider] = provider_counts.get(provider, 0) + 1
    return ordered


def normalize_runs_dataframe(runs, dataset_df: pd.DataFrame, dataset_name: str, sweep_name: str) -> pd.DataFrame:
    run_df = runs.as_dataframe().copy()
    run_df["model_name"] = run_df["llm"].map(lambda value: getattr(value, "name", str(value)))
    run_df = run_df.drop(columns=["llm"], errors="ignore")
    result_df = pd.json_normalize(run_df["result"])
    result_df.columns = [f"metric_{column}" for column in result_df.columns]
    merged = pd.concat(
        [
            run_df.drop(columns=["result"], errors="ignore").reset_index(drop=True),
            result_df.reset_index(drop=True),
        ],
        axis=1,
    )
    metadata_df = dataset_df[
        [
            "scenario_id",
            "difficulty",
            "family",
            "gold_schedule",
            "gold_packets",
            "moved_tasks",
        ]
    ].rename(columns={"moved_tasks": "gold_moved_tasks"})
    merged = merged.merge(
        metadata_df,
        left_on="metric_scenario_id",
        right_on="scenario_id",
        how="left",
        suffixes=("", "_dataset"),
    )
    merged["dataset_name"] = dataset_name
    merged["sweep_name"] = sweep_name
    return merged


def summarize_results(eval_df: pd.DataFrame) -> pd.DataFrame:
    if eval_df.empty:
        return pd.DataFrame(
            columns=[
                "model_name",
                "scenarios",
                "composite",
                "schedule_exact",
                "schedule_valid",
                "position_accuracy",
                "packet_precision",
                "packet_recall",
                "packet_f1",
                "moved_exact",
                "moved_abs_error",
            ]
        )
    return (
        eval_df.groupby("model_name")
        .agg(
            scenarios=("metric_scenario_id", "count"),
            composite=("metric_composite", "mean"),
            schedule_exact=("metric_schedule_exact", "mean"),
            schedule_valid=("metric_schedule_valid", "mean"),
            position_accuracy=("metric_position_accuracy", "mean"),
            packet_precision=("metric_packet_precision", "mean"),
            packet_recall=("metric_packet_recall", "mean"),
            packet_f1=("metric_packet_f1", "mean"),
            moved_exact=("metric_moved_exact", "mean"),
            moved_abs_error=("metric_moved_abs_error", "mean"),
        )
        .reset_index()
        .sort_values(["composite", "schedule_exact", "packet_f1"], ascending=False)
    )


def summarize_by_slice(eval_df: pd.DataFrame, slice_column: str) -> pd.DataFrame:
    if eval_df.empty:
        return pd.DataFrame(
            columns=[
                "model_name",
                slice_column,
                "scenarios",
                "composite",
                "schedule_exact",
                "packet_f1",
                "moved_exact",
            ]
        )
    return (
        eval_df.groupby(["model_name", slice_column])
        .agg(
            scenarios=("metric_scenario_id", "count"),
            composite=("metric_composite", "mean"),
            schedule_exact=("metric_schedule_exact", "mean"),
            packet_f1=("metric_packet_f1", "mean"),
            moved_exact=("metric_moved_exact", "mean"),
        )
        .reset_index()
        .sort_values(["model_name", slice_column])
    )


def build_error_profile(eval_df: pd.DataFrame) -> pd.DataFrame:
    if eval_df.empty:
        return pd.DataFrame(
            columns=[
                "model_name",
                "invalid_schedule_rate",
                "packet_miss_rate",
                "overrepair_rate",
            ]
        )
    profile_df = eval_df.copy()
    profile_df["invalid_schedule_rate"] = 1.0 - profile_df["metric_schedule_valid"]
    profile_df["packet_miss_rate"] = 1.0 - profile_df["metric_packet_recall"]
    profile_df["overrepair_rate"] = (profile_df["metric_moved_abs_error"] > 0).astype(float)
    return (
        profile_df.groupby("model_name")
        .agg(
            invalid_schedule_rate=("invalid_schedule_rate", "mean"),
            packet_miss_rate=("packet_miss_rate", "mean"),
            overrepair_rate=("overrepair_rate", "mean"),
        )
        .reset_index()
        .sort_values(["overrepair_rate", "packet_miss_rate"], ascending=False)
    )


def persist_summary_bundle(prefix: str, eval_df: pd.DataFrame) -> None:
    eval_df.to_csv(WORKDIR / f"{prefix}_scenario_results.csv", index=False)
    summarize_results(eval_df).to_csv(WORKDIR / f"{prefix}_model_summary.csv", index=False)
    summarize_by_slice(eval_df, "difficulty").to_csv(
        WORKDIR / f"{prefix}_difficulty_summary.csv",
        index=False,
    )
    summarize_by_slice(eval_df, "family").to_csv(
        WORKDIR / f"{prefix}_family_summary.csv",
        index=False,
    )
    build_error_profile(eval_df).to_csv(
        WORKDIR / f"{prefix}_error_profile.csv",
        index=False,
    )


def evaluate_models(
    *,
    stage_name: str,
    sweep_name: str,
    model_names: list[str],
    evaluation_df: pd.DataFrame,
    dataset_df: pd.DataFrame,
    dataset_name: str,
    progress_state: dict[str, int],
) -> pd.DataFrame:
    collected_frames: list[pd.DataFrame] = []
    total_models = len(model_names)
    for index, model_name in enumerate(model_names, start=1):
        append_progress(
            stage=stage_name,
            status="running",
            detail=f"{dataset_name}: {model_name} ({index}/{total_models})",
            completed_units=progress_state["completed"],
            total_units=progress_state["total"],
            extra={"dataset_name": dataset_name, "model_name": model_name},
        )
        try:
            with kbench.client.enable_cache():
                runs = goalshield_item.evaluate(
                    llm=[kbench.llms[model_name]],
                    evaluation_data=evaluation_df,
                    max_attempts=1,
                    retry_delay=15,
                    timeout=240,
                    n_jobs=4,
                    remove_run_files=True,
                    stop_condition=lambda collected_runs: len(collected_runs) == evaluation_df.shape[0],
                )
            normalized_df = normalize_runs_dataframe(runs, dataset_df, dataset_name, sweep_name)
            collected_frames.append(normalized_df)
            combined_df = pd.concat(collected_frames, ignore_index=True)
            persist_summary_bundle(f"goalshield_{sweep_name}", combined_df)
            progress_state["completed"] += 1
            append_progress(
                stage=stage_name,
                status="completed",
                detail=f"{dataset_name}: {model_name}",
                completed_units=progress_state["completed"],
                total_units=progress_state["total"],
                extra={
                    "dataset_name": dataset_name,
                    "model_name": model_name,
                    "scenario_count": int(normalized_df.shape[0]),
                },
            )
        except Exception as exc:
            failure = {
                "stage": stage_name,
                "dataset_name": dataset_name,
                "model_name": model_name,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            FAILURES.append(failure)
            write_json(FAILURES_PATH, FAILURES)
            progress_state["completed"] += 1
            append_progress(
                stage=stage_name,
                status="failed",
                detail=f"{dataset_name}: {model_name}",
                completed_units=progress_state["completed"],
                total_units=progress_state["total"],
                extra=failure,
            )
            print(f"Skipping {model_name}: {type(exc).__name__}: {exc}")
    if collected_frames:
        return pd.concat(collected_frames, ignore_index=True)
    return pd.DataFrame()


AVAILABLE_MODEL_NAMES = sorted(getattr(kbench, "llms", {}).keys())
DEFAULT_MODEL_NAME = getattr(kbench.llm, "name", str(kbench.llm))
CANDIDATE_MODEL_NAMES = ordered_candidate_models(AVAILABLE_MODEL_NAMES)
if DEFAULT_MODEL_NAME in AVAILABLE_MODEL_NAMES and DEFAULT_MODEL_NAME not in CANDIDATE_MODEL_NAMES:
    CANDIDATE_MODEL_NAMES = [DEFAULT_MODEL_NAME] + CANDIDATE_MODEL_NAMES

PRIMARY_MODEL_NAMES = CANDIDATE_MODEL_NAMES[:8]
PROBE_MODEL_NAMES = CANDIDATE_MODEL_NAMES[8:16]
ROBUSTNESS_TOP_K = min(3, len(PRIMARY_MODEL_NAMES))
ROBUSTNESS_REQUESTED_MODEL_NAMES = PRIMARY_MODEL_NAMES[:ROBUSTNESS_TOP_K]

TOTAL_PROGRESS_UNITS = (
    1
    + len(DATASET_SPECS)
    + 1
    + len(PRIMARY_MODEL_NAMES)
    + len(PROBE_MODEL_NAMES)
    + ROBUSTNESS_TOP_K
    + 1
)
PROGRESS_STATE = {"completed": 0, "total": TOTAL_PROGRESS_UNITS}

RUNTIME_INFO = detect_runtime_info()
write_json(RUNTIME_INFO_PATH, RUNTIME_INFO)
write_json(
    AVAILABLE_MODELS_PATH,
    {
        "default_model": DEFAULT_MODEL_NAME,
        "available_models": AVAILABLE_MODEL_NAMES,
        "candidate_models": CANDIDATE_MODEL_NAMES,
        "primary_models": PRIMARY_MODEL_NAMES,
        "probe_models": PROBE_MODEL_NAMES,
    },
)
append_progress(
    stage="runtime",
    status="running",
    detail="Captured runtime and model inventory",
    completed_units=PROGRESS_STATE["completed"],
    total_units=PROGRESS_STATE["total"],
    extra={"available_model_count": len(AVAILABLE_MODEL_NAMES)},
)
PROGRESS_STATE["completed"] += 1
append_progress(
    stage="runtime",
    status="completed",
    detail="Captured runtime and model inventory",
    completed_units=PROGRESS_STATE["completed"],
    total_units=PROGRESS_STATE["total"],
    extra=RUNTIME_INFO,
)

RUN_PLAN = {
    "default_model": DEFAULT_MODEL_NAME,
    "dataset_specs": DATASET_SPECS,
    "primary_models": PRIMARY_MODEL_NAMES,
    "probe_models": PROBE_MODEL_NAMES,
    "robustness_requested_models": ROBUSTNESS_REQUESTED_MODEL_NAMES,
    "total_progress_units": TOTAL_PROGRESS_UNITS,
}
write_json(RUN_PLAN_PATH, RUN_PLAN)

# %%
append_progress(
    stage="datasets",
    status="running",
    detail="Building benchmark datasets",
    completed_units=PROGRESS_STATE["completed"],
    total_units=PROGRESS_STATE["total"],
)
DATASET_RECORDS: dict[str, list[dict[str, object]]] = {}
DATASET_FRAMES: dict[str, pd.DataFrame] = {}
EVAL_FRAMES: dict[str, pd.DataFrame] = {}
for dataset_name, spec in DATASET_SPECS.items():
    dataset_records = generate_benchmark_dataset(seed=spec["seed"], counts=spec["counts"])
    dataset_df = build_records_dataframe(dataset_records)
    DATASET_RECORDS[dataset_name] = dataset_records
    DATASET_FRAMES[dataset_name] = dataset_df
    EVAL_FRAMES[dataset_name] = dataset_df[["prompt", "scenario_json", "solution_json"]].copy()
    dataset_df.to_csv(WORKDIR / f"goalshield_dataset_{dataset_name}.csv", index=False)
    PROGRESS_STATE["completed"] += 1
    append_progress(
        stage="datasets",
        status="completed",
        detail=f"Built {dataset_name} dataset",
        completed_units=PROGRESS_STATE["completed"],
        total_units=PROGRESS_STATE["total"],
        extra={
            "dataset_name": dataset_name,
            "scenario_count": int(dataset_df.shape[0]),
            "difficulty_counts": dataset_df["difficulty"].value_counts().to_dict(),
        },
    )
DATASET_FRAMES["primary"].head(3)

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
benchmark_payload = EVAL_FRAMES["primary"].to_json(orient="records")
append_progress(
    stage="benchmark",
    status="running",
    detail=f"Running main benchmark with default model {DEFAULT_MODEL_NAME}",
    completed_units=PROGRESS_STATE["completed"],
    total_units=PROGRESS_STATE["total"],
)
leaderboard_run = goalshield_benchmark.run(kbench.llm, benchmark_payload)
(WORKDIR / "goalshield_leaderboard_run.txt").write_text(str(leaderboard_run), encoding="utf-8")
PROGRESS_STATE["completed"] += 1
append_progress(
    stage="benchmark",
    status="completed",
    detail=f"Completed main benchmark with default model {DEFAULT_MODEL_NAME}",
    completed_units=PROGRESS_STATE["completed"],
    total_units=PROGRESS_STATE["total"],
)
leaderboard_run

# %%
PRIMARY_EVAL_DF = evaluate_models(
    stage_name="primary_sweep",
    sweep_name="primary",
    model_names=PRIMARY_MODEL_NAMES,
    evaluation_df=EVAL_FRAMES["primary"],
    dataset_df=DATASET_FRAMES["primary"],
    dataset_name="primary",
    progress_state=PROGRESS_STATE,
)
PRIMARY_SUMMARY = summarize_results(PRIMARY_EVAL_DF)
PRIMARY_SUMMARY

# %%
PROBE_EVAL_DF = evaluate_models(
    stage_name="probe_sweep",
    sweep_name="probe",
    model_names=PROBE_MODEL_NAMES,
    evaluation_df=EVAL_FRAMES["probe"],
    dataset_df=DATASET_FRAMES["probe"],
    dataset_name="probe",
    progress_state=PROGRESS_STATE,
)
PROBE_SUMMARY = summarize_results(PROBE_EVAL_DF)
PROBE_SUMMARY

# %%
robustness_top_models = PRIMARY_SUMMARY.head(ROBUSTNESS_TOP_K)["model_name"].tolist()
ROBUSTNESS_MODEL_NAMES = (
    robustness_top_models
    + [name for name in ROBUSTNESS_REQUESTED_MODEL_NAMES if name not in robustness_top_models]
)[: len(ROBUSTNESS_REQUESTED_MODEL_NAMES)]
ROBUSTNESS_EVAL_DF = evaluate_models(
    stage_name="robustness_sweep",
    sweep_name="robustness",
    model_names=ROBUSTNESS_MODEL_NAMES,
    evaluation_df=EVAL_FRAMES["holdout"],
    dataset_df=DATASET_FRAMES["holdout"],
    dataset_name="holdout",
    progress_state=PROGRESS_STATE,
)
ROBUSTNESS_SUMMARY = summarize_results(ROBUSTNESS_EVAL_DF)
ROBUSTNESS_SUMMARY

# %%
ALL_EVAL_DF = pd.concat(
    [frame for frame in [PRIMARY_EVAL_DF, PROBE_EVAL_DF, ROBUSTNESS_EVAL_DF] if not frame.empty],
    ignore_index=True,
) if any(not frame.empty for frame in [PRIMARY_EVAL_DF, PROBE_EVAL_DF, ROBUSTNESS_EVAL_DF]) else pd.DataFrame()
if not ALL_EVAL_DF.empty:
    ALL_EVAL_DF.to_csv(WORKDIR / "goalshield_all_scenario_results.csv", index=False)

PRIMARY_SUMMARY.to_csv(WORKDIR / "goalshield_model_summary.csv", index=False)
summarize_by_slice(PRIMARY_EVAL_DF, "difficulty").to_csv(
    WORKDIR / "goalshield_difficulty_summary.csv",
    index=False,
)
summarize_by_slice(PRIMARY_EVAL_DF, "family").to_csv(
    WORKDIR / "goalshield_family_summary.csv",
    index=False,
)
build_error_profile(PRIMARY_EVAL_DF).to_csv(
    WORKDIR / "goalshield_error_profile.csv",
    index=False,
)

benchpress_scores = {
    BENCHPRESS_MAP[name]: score * 100.0
    for name, score in PRIMARY_SUMMARY.set_index("model_name")["composite"].to_dict().items()
    if name in BENCHPRESS_MAP
}
with open(WORKDIR / "goalshield_benchpress_input.json", "w", encoding="utf-8") as handle:
    json.dump(benchpress_scores, handle, indent=2)

summary_lines = [
    "# GoalShield Kaggle Run Summary",
    "",
    f"- Default benchmark model: `{DEFAULT_MODEL_NAME}`",
    f"- Primary models requested: `{len(PRIMARY_MODEL_NAMES)}`",
    f"- Probe models requested: `{len(PROBE_MODEL_NAMES)}`",
    f"- Holdout models requested: `{len(ROBUSTNESS_MODEL_NAMES)}`",
    f"- Failures captured: `{len(FAILURES)}`",
]
if not PRIMARY_SUMMARY.empty:
    top_row = PRIMARY_SUMMARY.iloc[0]
    summary_lines.append(
        f"- Best primary composite so far: `{top_row['model_name']}` = `{top_row['composite']:.3f}`"
    )
(WORKDIR / "goalshield_submission_summary.md").write_text(
    "\n".join(summary_lines),
    encoding="utf-8",
)

PROGRESS_STATE["completed"] += 1
append_progress(
    stage="exports",
    status="completed",
    detail="Wrote summaries, slices, and writeup artifacts",
    completed_units=PROGRESS_STATE["completed"],
    total_units=PROGRESS_STATE["total"],
    extra={
        "primary_models_completed": int(PRIMARY_SUMMARY.shape[0]),
        "probe_models_completed": int(PROBE_SUMMARY.shape[0]),
        "robustness_models_completed": int(ROBUSTNESS_SUMMARY.shape[0]),
    },
)
print("Saved /kaggle/working/goalshield_model_summary.csv")
print("Saved /kaggle/working/goalshield_difficulty_summary.csv")
print("Saved /kaggle/working/goalshield_family_summary.csv")
print("Saved /kaggle/working/goalshield_error_profile.csv")
print("Saved /kaggle/working/goalshield_benchpress_input.json")
print("Saved /kaggle/working/goalshield_progress.json")
PRIMARY_SUMMARY

# %%
append_progress(
    stage="novelty",
    status="running",
    detail="Attempting optional BenchPress novelty analysis",
    completed_units=PROGRESS_STATE["completed"],
    total_units=PROGRESS_STATE["total"],
)
novelty_status: dict[str, object] = {"benchpress_model_count": len(benchpress_scores)}
if len(benchpress_scores) >= 3:
    install_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "https://github.com/kafkasl/evaluating-agi/archive/refs/heads/main.zip",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    novelty_status["install_returncode"] = install_result.returncode
    if install_result.stderr.strip():
        novelty_status["install_stderr_tail"] = install_result.stderr.strip().splitlines()[-10:]
    if install_result.returncode == 0:
        try:
            from evaluating_agi.benchpress import check_novelty

            novelty_report = check_novelty(benchpress_scores, name="GoalShield")
            write_json(WORKDIR / "goalshield_benchpress_report.json", novelty_report)
            novelty_status["report_written"] = True
        except Exception as exc:
            novelty_status["report_error"] = f"{type(exc).__name__}: {exc}"
    else:
        novelty_status["report_error"] = "optional novelty package install failed"
else:
    novelty_status["report_error"] = "need at least 3 mapped models for BenchPress novelty"

write_json(WORKDIR / "goalshield_novelty_status.json", novelty_status)
PROGRESS_STATE["completed"] += 1
append_progress(
    stage="novelty",
    status="completed",
    detail="Finished optional novelty analysis step",
    completed_units=PROGRESS_STATE["completed"],
    total_units=PROGRESS_STATE["total"],
    extra=novelty_status,
)
pd.DataFrame(PROGRESS_LOG).tail(10)

# %%
%choose goalshield_benchmark
