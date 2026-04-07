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
import re
import subprocess
import sys
import time

import pandas as pd

import kaggle_benchmarks as kbench

from agi_cognitive_benchmark.dataset import build_records_dataframe, generate_benchmark_dataset
from agi_cognitive_benchmark.metrics import score_plan_answer
from agi_cognitive_benchmark.models import Scenario, Solution
from agi_cognitive_benchmark.parsing import parse_plan_answer_response

WORKDIR = Path("/kaggle/working")
PROGRESS_PATH = WORKDIR / "goalshield_progress.json"
RUN_PLAN_PATH = WORKDIR / "goalshield_run_plan.json"
RUNTIME_INFO_PATH = WORKDIR / "goalshield_runtime_info.json"
AVAILABLE_MODELS_PATH = WORKDIR / "goalshield_available_models.json"
FAILURES_PATH = WORKDIR / "goalshield_failures.json"
PARTIALS_PATH = WORKDIR / "goalshield_partial_models.json"

RUN_PROFILE = {
    "name": "adaptive_submission",
    "primary_model_count": 4,
    "healthcheck_model_count": 10,
    "probe_model_count": 0,
    "robustness_top_k": 1,
    "benchmark_model_name": "google/gemini-2.0-flash",
    "benchmark_n_jobs": 2,
    "evaluation_n_jobs": 2,
    "benchmark_timeout": 180,
    "evaluation_timeout": 240,
    "retry_n_jobs": 1,
    "retry_timeout": 420,
    "max_completion_retries": 6,
    "evaluation_batch_size": 6,
    "retry_batch_size": 3,
}

MODEL_PRIORITY = [
    "google/gemini-2.0-flash",
    "google/gemma-3-27b",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "deepseek-ai/deepseek-v3.2",
    "google/gemini-2.0-flash-lite",
    "google/gemma-3-12b",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
    "google/gemini-3-flash-preview",
    "google/gemini-3.1-flash-lite-preview",
    "google/gemini-3.1-pro-preview",
    "anthropic/claude-haiku-4-5@20251001",
    "anthropic/claude-sonnet-4@20250514",
    "anthropic/claude-opus-4-1@20250805",
    "google/gemma-4-31b",
    "qwen/qwen3-next-80b-a3b-instruct",
    "qwen/qwen3-coder-480b-a35b-instruct",
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
        "counts": {"easy": 18, "medium": 18, "hard": 18},
        "description": "Main benchmark slice used for the Kaggle benchmark entity.",
    },
    "probe": {
        "seed": 20260407,
        "counts": {"easy": 8, "medium": 8, "hard": 8},
        "description": "Breadth sweep slice for extra model coverage.",
    },
    "holdout": {
        "seed": 20260408,
        "counts": {"easy": 6, "medium": 6, "hard": 6},
        "description": "Robustness slice for top primary models.",
    },
}
BENCHPRESS_MAP = {
    "google/gemini-2.0-flash": "gemini-2.0-flash",
    "google/gemma-3-27b": "gemma-3-27b",
    "openai/gpt-oss-20b": "gpt-oss-20b",
    "openai/gpt-oss-120b": "gpt-oss-120b",
    "deepseek-ai/deepseek-v3.2": "deepseek-v3.2",
    "google/gemini-2.5-flash": "gemini-2.5-flash",
    "google/gemini-2.5-pro": "gemini-2.5-pro",
    "google/gemini-3-flash-preview": "gemini-3-flash",
    "anthropic/claude-haiku-4-5@20251001": "claude-haiku-4.5",
    "anthropic/claude-sonnet-4@20250514": "claude-sonnet-4",
    "anthropic/claude-opus-4-1@20250805": "claude-opus-4",
}
PROGRESS_LOG: list[dict[str, object]] = []
FAILURES: list[dict[str, object]] = []
PARTIAL_MODELS: list[dict[str, object]] = []


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
    run_records = []
    for run in getattr(runs, "runs", []):
        record = dict(getattr(run, "params", {}))
        record["run_id"] = getattr(run, "id", "")
        record["result"] = getattr(run, "result", None)
        param_id = getattr(run, "param_id", None)
        if param_id is not None:
            record["id"] = param_id
        error_message = getattr(run, "error_message", None)
        if error_message:
            record["error_message"] = error_message
        run_records.append(record)
    if not run_records:
        return pd.DataFrame()

    run_df = pd.DataFrame(run_records).copy()
    run_df["model_name"] = run_df["llm"].map(lambda value: getattr(value, "name", str(value)))
    run_df = run_df.drop(columns=["llm"], errors="ignore")
    result_df = pd.json_normalize(
        run_df["result"].map(lambda value: value if isinstance(value, dict) else {})
    )
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
    if "metric_scenario_id" not in merged.columns:
        return pd.DataFrame()
    return merged[merged["metric_scenario_id"].notna()].reset_index(drop=True)


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


def summarize_response_health(eval_df: pd.DataFrame) -> pd.DataFrame:
    if eval_df.empty:
        return pd.DataFrame(
            columns=[
                "model_name",
                "scenarios",
                "parsed",
                "unparsed",
                "error",
                "parsed_rate",
                "mean_composite",
                "mean_response_chars",
                "dominant_error_type",
            ]
        )

    health_df = eval_df.copy()
    health_df["parsed"] = health_df["metric_response_status"].eq("parsed").astype(int)
    health_df["unparsed"] = health_df["metric_response_status"].eq("unparsed").astype(int)
    health_df["error"] = health_df["metric_response_status"].eq("error").astype(int)

    summary = (
        health_df.groupby("model_name")
        .agg(
            scenarios=("metric_scenario_id", "count"),
            parsed=("parsed", "sum"),
            unparsed=("unparsed", "sum"),
            error=("error", "sum"),
            parsed_rate=("parsed", "mean"),
            mean_composite=("metric_composite", "mean"),
            mean_response_chars=("metric_response_chars", "mean"),
        )
        .reset_index()
    )
    dominant_error = (
        health_df[health_df["metric_response_error_type"].astype(str) != ""]
        .groupby("model_name")["metric_response_error_type"]
        .agg(lambda values: values.value_counts().index[0])
        .to_dict()
    )
    summary["dominant_error_type"] = summary["model_name"].map(dominant_error).fillna("")
    return summary.sort_values(
        ["parsed", "error", "mean_composite", "mean_response_chars"],
        ascending=[False, True, False, False],
    ).reset_index(drop=True)


def select_primary_models(
    health_summary: pd.DataFrame,
    healthcheck_model_names: list[str],
    fallback_names: list[str],
) -> list[str]:
    priority = {name: index for index, name in enumerate(fallback_names)}
    health_records = {
        row["model_name"]: row
        for row in health_summary.to_dict(orient="records")
    }
    denied_models = {
        name
        for name, row in health_records.items()
        if row.get("dominant_error_type") == "PermissionDeniedError"
    }

    ranked = sorted(
        healthcheck_model_names,
        key=lambda name: (
            -int(health_records.get(name, {}).get("parsed", 0)),
            int(health_records.get(name, {}).get("error", 999)),
            -int(name in BENCHPRESS_MAP),
            -float(health_records.get(name, {}).get("mean_composite", 0.0) or 0.0),
            priority.get(name, 999),
        ),
    )

    selected = [
        name
        for name in ranked
        if int(health_records.get(name, {}).get("parsed", 0)) > 0
    ]
    for name in ranked + fallback_names:
        if name in denied_models:
            continue
        if name in AVAILABLE_MODEL_NAMES and name not in selected:
            selected.append(name)
        if len(selected) >= RUN_PROFILE["primary_model_count"]:
            break
    return selected


def combine_unique_results(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    frames = [frame for frame in [existing_df, new_df] if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["metric_scenario_id"], keep="last")
        .reset_index(drop=True)
    )


def chunk_dataframe(frame: pd.DataFrame, chunk_size: int) -> list[pd.DataFrame]:
    if frame.empty:
        return []
    chunk_size = max(1, int(chunk_size))
    return [frame.iloc[start : start + chunk_size].copy() for start in range(0, len(frame), chunk_size)]


def batch_size_for_round(round_index: int) -> int:
    if round_index == 0:
        return RUN_PROFILE["evaluation_batch_size"]
    if round_index == 1:
        return RUN_PROFILE["retry_batch_size"]
    return 1


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
    partial_frames: list[pd.DataFrame] = []
    total_models = len(model_names)
    expected_count = int(dataset_df.shape[0])
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
            normalized_df = pd.DataFrame()
            remaining_dataset_df = dataset_df.copy()
            max_rounds = RUN_PROFILE["max_completion_retries"] + 1
            for round_index in range(max_rounds):
                previous_count = (
                    int(normalized_df["metric_scenario_id"].nunique())
                    if not normalized_df.empty
                    else 0
                )
                current_eval_df = remaining_dataset_df[["prompt", "scenario_json", "solution_json"]]
                if current_eval_df.empty:
                    break
                timeout = (
                    RUN_PROFILE["evaluation_timeout"]
                    if round_index == 0
                    else RUN_PROFILE["retry_timeout"]
                )
                n_jobs = (
                    RUN_PROFILE["evaluation_n_jobs"]
                    if round_index == 0
                    else RUN_PROFILE["retry_n_jobs"]
                )
                chunk_size = batch_size_for_round(round_index)
                batch_frames: list[pd.DataFrame] = []
                for batch_index, batch_df in enumerate(chunk_dataframe(current_eval_df, chunk_size), start=1):
                    append_progress(
                        stage=stage_name,
                        status="running",
                        detail=f"{dataset_name}: {model_name} batch {batch_index}",
                        completed_units=progress_state["completed"],
                        total_units=progress_state["total"],
                        extra={
                            "dataset_name": dataset_name,
                            "model_name": model_name,
                            "round_index": round_index,
                            "batch_index": batch_index,
                            "batch_size": int(batch_df.shape[0]),
                        },
                    )
                    with kbench.client.enable_cache():
                        runs = goalshield_item.evaluate(
                            llm=[kbench.llms[model_name]],
                            evaluation_data=batch_df,
                            max_attempts=1,
                            retry_delay=15,
                            timeout=timeout,
                            n_jobs=min(n_jobs, max(1, int(batch_df.shape[0]))),
                            remove_run_files=True,
                            stop_condition=lambda collected_runs: len(collected_runs) == batch_df.shape[0],
                        )
                    attempt_df = normalize_runs_dataframe(runs, dataset_df, dataset_name, sweep_name)
                    if not attempt_df.empty:
                        batch_frames.append(attempt_df)
                if batch_frames:
                    normalized_df = combine_unique_results(
                        normalized_df,
                        pd.concat(batch_frames, ignore_index=True),
                    )
                completed_ids = set(normalized_df["metric_scenario_id"]) if not normalized_df.empty else set()
                remaining_dataset_df = dataset_df[~dataset_df["scenario_id"].isin(completed_ids)].copy()
                if remaining_dataset_df.empty:
                    break
                current_count = len(completed_ids)
                if current_count == previous_count and chunk_size == 1:
                    break
                if round_index < max_rounds - 1:
                    append_progress(
                        stage=stage_name,
                        status="retrying",
                        detail=f"{dataset_name}: {model_name} retry {round_index + 1}",
                        completed_units=progress_state["completed"],
                        total_units=progress_state["total"],
                        extra={
                            "dataset_name": dataset_name,
                            "model_name": model_name,
                            "completed_scenarios": len(completed_ids),
                            "expected_scenarios": expected_count,
                            "remaining_scenarios": int(remaining_dataset_df.shape[0]),
                        },
                    )

            final_count = (
                int(normalized_df["metric_scenario_id"].nunique())
                if not normalized_df.empty
                else 0
            )
            if final_count == expected_count:
                collected_frames.append(normalized_df)
                combined_df = pd.concat(collected_frames, ignore_index=True)
                persist_summary_bundle(f"goalshield_{sweep_name}", combined_df)
                completion_status = "completed"
                completion_extra = {
                    "dataset_name": dataset_name,
                    "model_name": model_name,
                    "scenario_count": final_count,
                }
            else:
                partial_info = {
                    "stage": stage_name,
                    "dataset_name": dataset_name,
                    "model_name": model_name,
                    "expected_scenarios": expected_count,
                    "actual_scenarios": final_count,
                }
                PARTIAL_MODELS.append(partial_info)
                write_json(PARTIALS_PATH, PARTIAL_MODELS)
                if not normalized_df.empty:
                    partial_frames.append(normalized_df.assign(expected_scenarios=expected_count))
                    pd.concat(partial_frames, ignore_index=True).to_csv(
                        WORKDIR / f"goalshield_{sweep_name}_partial_scenario_results.csv",
                        index=False,
                    )
                completion_status = "partial"
                completion_extra = partial_info
            progress_state["completed"] += 1
            append_progress(
                stage=stage_name,
                status=completion_status,
                detail=f"{dataset_name}: {model_name}",
                completed_units=progress_state["completed"],
                total_units=progress_state["total"],
                extra=completion_extra,
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
BENCHMARK_MODEL_NAME = (
    RUN_PROFILE["benchmark_model_name"]
    if RUN_PROFILE["benchmark_model_name"] in AVAILABLE_MODEL_NAMES
    else DEFAULT_MODEL_NAME
)
CANDIDATE_MODEL_NAMES = ordered_candidate_models(AVAILABLE_MODEL_NAMES)
if DEFAULT_MODEL_NAME in AVAILABLE_MODEL_NAMES and DEFAULT_MODEL_NAME not in CANDIDATE_MODEL_NAMES:
    CANDIDATE_MODEL_NAMES = [DEFAULT_MODEL_NAME] + CANDIDATE_MODEL_NAMES

HEALTHCHECK_MODEL_NAMES = [
    name for name in CANDIDATE_MODEL_NAMES if name in BENCHPRESS_MAP
][: RUN_PROFILE["healthcheck_model_count"]]
for name in CANDIDATE_MODEL_NAMES:
    if len(HEALTHCHECK_MODEL_NAMES) >= RUN_PROFILE["healthcheck_model_count"]:
        break
    if name not in HEALTHCHECK_MODEL_NAMES:
        HEALTHCHECK_MODEL_NAMES.append(name)

PRIMARY_MODEL_NAMES = HEALTHCHECK_MODEL_NAMES[: RUN_PROFILE["primary_model_count"]]
PROBE_MODEL_NAMES = CANDIDATE_MODEL_NAMES[
    RUN_PROFILE["primary_model_count"] : RUN_PROFILE["primary_model_count"] + RUN_PROFILE["probe_model_count"]
]
ROBUSTNESS_TOP_K = min(RUN_PROFILE["robustness_top_k"], len(PRIMARY_MODEL_NAMES))
ROBUSTNESS_REQUESTED_MODEL_NAMES = PRIMARY_MODEL_NAMES[:ROBUSTNESS_TOP_K]
HEALTHCHECK_SCENARIOS = 3
PRIMARY_SCENARIOS = sum(DATASET_SPECS["primary"]["counts"].values())
PROBE_SCENARIOS = sum(DATASET_SPECS["probe"]["counts"].values())
HOLDOUT_SCENARIOS = sum(DATASET_SPECS["holdout"]["counts"].values())
ESTIMATED_TOTAL_CALLS = (
    PRIMARY_SCENARIOS
    + HEALTHCHECK_SCENARIOS * len(HEALTHCHECK_MODEL_NAMES)
    + PRIMARY_SCENARIOS * RUN_PROFILE["primary_model_count"]
    + PROBE_SCENARIOS * len(PROBE_MODEL_NAMES)
    + HOLDOUT_SCENARIOS * RUN_PROFILE["robustness_top_k"]
)

TOTAL_PROGRESS_UNITS = (
    1
    + len(DATASET_SPECS)
    + len(HEALTHCHECK_MODEL_NAMES)
    + 1
    + RUN_PROFILE["primary_model_count"]
    + len(PROBE_MODEL_NAMES)
    + RUN_PROFILE["robustness_top_k"]
    + 1
    + 1
)
PROGRESS_STATE = {"completed": 0, "total": TOTAL_PROGRESS_UNITS}

RUNTIME_INFO = detect_runtime_info()
write_json(RUNTIME_INFO_PATH, RUNTIME_INFO)
write_json(
    AVAILABLE_MODELS_PATH,
    {
        "default_model": DEFAULT_MODEL_NAME,
        "benchmark_model": BENCHMARK_MODEL_NAME,
        "available_models": AVAILABLE_MODEL_NAMES,
        "candidate_models": CANDIDATE_MODEL_NAMES,
        "healthcheck_models": HEALTHCHECK_MODEL_NAMES,
        "run_profile": RUN_PROFILE,
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
    extra={
        "available_model_count": len(AVAILABLE_MODEL_NAMES),
        "run_profile": RUN_PROFILE["name"],
        "estimated_total_calls": ESTIMATED_TOTAL_CALLS,
    },
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
    "benchmark_model": BENCHMARK_MODEL_NAME,
    "run_profile": RUN_PROFILE,
    "dataset_specs": DATASET_SPECS,
    "healthcheck_models": HEALTHCHECK_MODEL_NAMES,
    "primary_models": PRIMARY_MODEL_NAMES,
    "probe_models": PROBE_MODEL_NAMES,
    "robustness_requested_models": ROBUSTNESS_REQUESTED_MODEL_NAMES,
    "estimated_total_calls": ESTIMATED_TOTAL_CALLS,
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
HEALTHCHECK_DATASET_DF = (
    DATASET_FRAMES["primary"]
    .sort_values(["difficulty", "scenario_id"])
    .groupby("difficulty", group_keys=False)
    .head(1)
    .reset_index(drop=True)
)
EVAL_FRAMES["healthcheck"] = HEALTHCHECK_DATASET_DF[["prompt", "scenario_json", "solution_json"]].copy()
HEALTHCHECK_DATASET_DF.to_csv(WORKDIR / "goalshield_dataset_healthcheck.csv", index=False)
DATASET_FRAMES["primary"].head(3)

# %%
SYSTEM_PROMPT = """
You are solving a schedule-repair task.
Always follow the explicit packet policy in the user prompt.
Return only a JSON object with keys applicable_packets, final_schedule, moved_tasks, confidence.
"""


@kbench.task(name="goalshield_item", store_task=False)
def goalshield_item(llm, prompt: str, scenario_json: str, solution_json: str) -> dict:
    scenario = Scenario.from_json(scenario_json)
    solution = Solution.from_json(solution_json)
    raw_answer = ""
    response_status = "parsed"
    response_error_type = ""
    response_error_message = ""
    with kbench.chats.new(f"goalshield-{scenario.scenario_id}"):
        kbench.system.send(SYSTEM_PROMPT)
        try:
            raw_answer = llm.prompt(prompt)
            answer = parse_plan_answer_response(raw_answer)
            if answer is None:
                response_status = "unparsed"
        except Exception as exc:
            answer = None
            response_status = "error"
            response_error_type = type(exc).__name__
            response_error_message = str(exc)
            raw_answer = str(exc)
    result = score_plan_answer(scenario, solution, answer)
    result["response_status"] = response_status
    result["response_error_type"] = response_error_type
    result["response_error_message"] = response_error_message
    result["response_chars"] = len(raw_answer)
    result["response_has_json"] = int(bool(re.search(r"```(?:json)?|\\{", raw_answer, re.IGNORECASE)))
    return result


@kbench.task(name="goalshield_benchmark")
def goalshield_benchmark(llm, evaluation_df_json: str) -> tuple[float, float]:
    evaluation_df = pd.read_json(StringIO(evaluation_df_json), orient="records")
    with kbench.client.enable_cache():
        runs = goalshield_item.evaluate(
            llm=[llm],
            evaluation_data=evaluation_df,
            max_attempts=1,
            retry_delay=15,
            timeout=RUN_PROFILE["benchmark_timeout"],
            n_jobs=RUN_PROFILE["benchmark_n_jobs"],
            remove_run_files=True,
            stop_condition=lambda collected_runs: len(collected_runs) == evaluation_df.shape[0],
        )
    eval_df = runs.as_dataframe()
    return (
        float(eval_df.result.str.get("composite").mean()),
        float(eval_df.result.str.get("composite").std(ddof=0)),
    )

# %%
HEALTHCHECK_EVAL_DF = evaluate_models(
    stage_name="healthcheck",
    sweep_name="healthcheck",
    model_names=HEALTHCHECK_MODEL_NAMES,
    evaluation_df=EVAL_FRAMES["healthcheck"],
    dataset_df=HEALTHCHECK_DATASET_DF,
    dataset_name="healthcheck",
    progress_state=PROGRESS_STATE,
)
HEALTHCHECK_SUMMARY = summarize_response_health(HEALTHCHECK_EVAL_DF)
HEALTHCHECK_SUMMARY.to_csv(WORKDIR / "goalshield_healthcheck_summary.csv", index=False)
PRIMARY_MODEL_NAMES = select_primary_models(
    HEALTHCHECK_SUMMARY,
    HEALTHCHECK_MODEL_NAMES,
    CANDIDATE_MODEL_NAMES,
)
ROBUSTNESS_TOP_K = min(RUN_PROFILE["robustness_top_k"], len(PRIMARY_MODEL_NAMES))
ROBUSTNESS_REQUESTED_MODEL_NAMES = PRIMARY_MODEL_NAMES[:ROBUSTNESS_TOP_K]
write_json(
    AVAILABLE_MODELS_PATH,
    {
        "default_model": DEFAULT_MODEL_NAME,
        "benchmark_model": BENCHMARK_MODEL_NAME,
        "available_models": AVAILABLE_MODEL_NAMES,
        "candidate_models": CANDIDATE_MODEL_NAMES,
        "healthcheck_models": HEALTHCHECK_MODEL_NAMES,
        "healthcheck_selected_primary_models": PRIMARY_MODEL_NAMES,
        "run_profile": RUN_PROFILE,
        "primary_models": PRIMARY_MODEL_NAMES,
        "probe_models": PROBE_MODEL_NAMES,
    },
)
RUN_PLAN.update(
    {
        "healthcheck_models": HEALTHCHECK_MODEL_NAMES,
        "selected_primary_models": PRIMARY_MODEL_NAMES,
        "robustness_requested_models": ROBUSTNESS_REQUESTED_MODEL_NAMES,
    }
)
write_json(RUN_PLAN_PATH, RUN_PLAN)
HEALTHCHECK_SUMMARY

# %%
benchmark_payload = EVAL_FRAMES["primary"].to_json(orient="records")
append_progress(
    stage="benchmark",
    status="running",
    detail=f"Running main benchmark with benchmark model {BENCHMARK_MODEL_NAME}",
    completed_units=PROGRESS_STATE["completed"],
    total_units=PROGRESS_STATE["total"],
)
leaderboard_run = goalshield_benchmark.run(kbench.llms[BENCHMARK_MODEL_NAME], benchmark_payload)
(WORKDIR / "goalshield_leaderboard_run.txt").write_text(str(leaderboard_run), encoding="utf-8")
PROGRESS_STATE["completed"] += 1
append_progress(
    stage="benchmark",
    status="completed",
    detail=f"Completed main benchmark with benchmark model {BENCHMARK_MODEL_NAME}",
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
PRIMARY_HEALTH_SUMMARY = summarize_response_health(PRIMARY_EVAL_DF)
PRIMARY_HEALTH_SUMMARY.to_csv(WORKDIR / "goalshield_primary_health_summary.csv", index=False)
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

benchpress_eligible_models = set(
    PRIMARY_HEALTH_SUMMARY.loc[PRIMARY_HEALTH_SUMMARY["parsed"] > 0, "model_name"].tolist()
)
benchpress_scores = {
    BENCHPRESS_MAP[name]: score * 100.0
    for name, score in PRIMARY_SUMMARY.set_index("model_name")["composite"].to_dict().items()
    if name in BENCHPRESS_MAP and name in benchpress_eligible_models
}
with open(WORKDIR / "goalshield_benchpress_input.json", "w", encoding="utf-8") as handle:
    json.dump(benchpress_scores, handle, indent=2)

summary_lines = [
    "# GoalShield Kaggle Run Summary",
    "",
    f"- Run profile: `{RUN_PROFILE['name']}`",
    f"- Default benchmark model: `{DEFAULT_MODEL_NAME}`",
    f"- Benchmark run model: `{BENCHMARK_MODEL_NAME}`",
    f"- Healthcheck models requested: `{len(HEALTHCHECK_MODEL_NAMES)}`",
    f"- Primary models requested: `{len(PRIMARY_MODEL_NAMES)}`",
    f"- Probe models requested: `{len(PROBE_MODEL_NAMES)}`",
    f"- Holdout models requested: `{len(ROBUSTNESS_MODEL_NAMES)}`",
    f"- Estimated total model calls: `{ESTIMATED_TOTAL_CALLS}`",
    f"- Failures captured: `{len(FAILURES)}`",
    f"- Partial models excluded: `{len(PARTIAL_MODELS)}`",
    f"- Selected primary models: `{', '.join(PRIMARY_MODEL_NAMES)}`",
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
        "partial_models_excluded": int(len(PARTIAL_MODELS)),
    },
)
print("Saved /kaggle/working/goalshield_model_summary.csv")
print("Saved /kaggle/working/goalshield_difficulty_summary.csv")
print("Saved /kaggle/working/goalshield_family_summary.csv")
print("Saved /kaggle/working/goalshield_error_profile.csv")
print("Saved /kaggle/working/goalshield_benchpress_input.json")
print("Saved /kaggle/working/goalshield_partial_models.json")
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
    deps_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "scipy",
            "scikit-learn",
            "openpyxl",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    novelty_status["deps_returncode"] = deps_result.returncode
    if deps_result.stderr.strip():
        novelty_status["deps_stderr_tail"] = deps_result.stderr.strip().splitlines()[-10:]
    install_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-deps",
            "https://github.com/kafkasl/evaluating-agi/archive/refs/heads/main.zip",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    novelty_status["install_returncode"] = install_result.returncode
    if install_result.stderr.strip():
        novelty_status["install_stderr_tail"] = install_result.stderr.strip().splitlines()[-10:]
    if deps_result.returncode == 0 and install_result.returncode == 0:
        try:
            from evaluating_agi.benchpress import check_novelty

            novelty_report = check_novelty(benchpress_scores, name="GoalShield")
            write_json(WORKDIR / "goalshield_benchpress_report.json", novelty_report)
            novelty_status["report_written"] = True
        except Exception as exc:
            novelty_status["report_error"] = f"{type(exc).__name__}: {exc}"
    elif deps_result.returncode != 0:
        novelty_status["report_error"] = "optional novelty dependencies install failed"
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
