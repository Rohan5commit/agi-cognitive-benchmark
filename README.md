# GoalShield

GoalShield is a Kaggle Benchmark for the April 16, 2026 `kaggle-measuring-agi` hackathon. It targets the **Executive Functions** track with one sharp question:

**Can a model preserve a goal state, inhibit distractor updates, and minimally repair a plan when the rule stream changes mid-task?**

The benchmark is built around one domain only: **schedule repair under interruption**. Each task starts from a baseline plan, injects a stream of update packets with mixed authority, and asks the model to output the final schedule that:

1. applies only the binding packets,
2. satisfies all active constraints, and
3. moves as few tasks as possible from the baseline plan.

That combination isolates planning, inhibitory control, working memory, and cognitive flexibility without drifting into generic reasoning trivia.

## Core Insight

Most planning-style benchmarks tell you whether a model can produce *a* valid answer in a static setting. GoalShield is designed to expose **executive drift** instead:

- models that over-trust the latest visible packet,
- models that follow same-team drafts they should suppress,
- models that can solve constraints but cannot preserve the minimum-edit objective,
- models that over-replan rather than repair.

The intended novelty claim is that **executive control failure is not just "bad planning"; it is the interaction between rule selection, goal maintenance, and revision economy**.

## Task Families

GoalShield v1 uses three solver-backed task families.

| Family | What changes | Primary failure mode |
| --- | --- | --- |
| `shield` | One real update appears among distractor packets | Salience-driven distraction |
| `switch` | Two applicable packets share a rule key; the later one overrides the earlier one | Perseveration on stale rules |
| `repair` | Multiple binding updates force a small but nonzero repair | Over-replanning vs. minimal repair |

Every scenario is exactly scored by brute-force search over all schedules. The gold answer is deterministic because ties are broken lexicographically after minimizing moved-task count.

## Dataset Summary

The committed `goalshield_v1_full.jsonl` dataset contains **54 tasks**:

| Difficulty | Families | Count |
| --- | --- | --- |
| `easy` | `shield` | 18 |
| `medium` | `shield`, `switch` | 18 |
| `hard` | `switch`, `repair` | 18 |

Moved-task distribution in v1:

| Difficulty | Typical repair cost |
| --- | --- |
| `easy` | 2 moved positions |
| `medium` | usually 2, sometimes 3 |
| `hard` | usually 3, sometimes 4 |

The main leaderboard metric is a **composite score**:

- `55%` exact final schedule match
- `20%` position-wise schedule accuracy
- `15%` packet-selection F1
- `10%` exact moved-task count

This keeps the benchmark sensitive even when a model is close but not exact.

## Baseline Signal

The repo includes deterministic heuristic baselines in [results/heuristic_summary_overall.csv](results/heuristic_summary_overall.csv).

| Policy | Schedule Exact | Packet F1 | Composite |
| --- | ---: | ---: | ---: |
| `gold` | 1.000 | 1.000 | 1.000 |
| `latest_visible_key` | 0.438 | 0.375 | 0.504 |
| `same_team_any_status` | 0.000 | 0.646 | 0.205 |
| `apply_all_packets` | 0.000 | 0.504 | 0.175 |
| `ignore_updates` | 0.000 | 0.000 | 0.120 |

The important pattern is that the strongest naive baseline is **not** the one that ignores updates. It is the one that follows the most recent visible packet. That is exactly the executive-control signature GoalShield is meant to discriminate.

## Latest Kaggle Evidence

The latest successful Kaggle Benchmarks run is archived in [results/kaggle_v20](results/kaggle_v20). This was a **quota-constrained novelty-harvest run**: a one-scenario mapped-model comparison used to validate BenchPress compatibility and extract a clean cross-model signal without burning more hosted quota on failed sweeps.

Primary mapped-model comparison from that run:

| Model | Composite | Note |
| --- | ---: | --- |
| `google/gemini-2.5-pro` | 0.90 | perfect schedule and packet selection on the sampled item |
| `google/gemini-3-flash-preview` | 0.90 | matched `gemini-2.5-pro` on the sampled item |
| `google/gemini-2.5-flash` | 0.85 | perfect schedule, but weaker packet-selection signal |

The same run also wrote a valid BenchPress novelty report in [results/kaggle_v20/goalshield_benchpress_report.json](results/kaggle_v20/goalshield_benchpress_report.json) from:

```json
{
  "gemini-2.5-pro": 90.0,
  "gemini-3-flash": 90.0,
  "gemini-2.5-flash": 85.0
}
```

BenchPress summary:

- `median_error = 3.42`
- `gemini-2.5-pro` scored `+3.42` above the BenchPress estimate
- `gemini-3-flash` was close to the BenchPress estimate (`+0.16`)
- `gemini-2.5-flash` scored `-5.00` below the BenchPress estimate

Interpretation: even on a compact executive-control task, closely related frontier models separate on **rule filtering and minimal-repair discipline**, not just generic reasoning strength. This run is directional rather than final leaderboard evidence, but it confirms GoalShield can produce a clean, mapped multi-model signal inside Kaggle.

## Repository Layout

| Path | Purpose |
| --- | --- |
| [src/agi_cognitive_benchmark](src/agi_cognitive_benchmark) | scenario schema, generator, solver, scoring, baselines |
| [data/goalshield_v1_pilot.jsonl](data/goalshield_v1_pilot.jsonl) | small smoke-test dataset |
| [data/goalshield_v1_full.jsonl](data/goalshield_v1_full.jsonl) | full v1 benchmark dataset |
| [notebooks/goalshield_benchmark.py](notebooks/goalshield_benchmark.py) | Kaggle notebook source in `py:percent` format |
| [scripts/build_kaggle_notebook.py](scripts/build_kaggle_notebook.py) | converts the notebook source to a Kaggle-pushable `.ipynb` |
| [scripts/publish_kaggle_notebook.py](scripts/publish_kaggle_notebook.py) | builds and pushes the Kaggle notebook |
| [writeup/draft.md](writeup/draft.md) | competition writeup draft |
| [results/heuristic_baselines_full.csv](results/heuristic_baselines_full.csv) | per-task heuristic outputs |

## Local Workflow

Generate the datasets:

```bash
uv run goalshield-generate --easy 18 --medium 18 --hard 18 --output data/goalshield_v1_full.jsonl
```

Run the heuristic baselines:

```bash
uv run goalshield-baselines --dataset data/goalshield_v1_full.jsonl --output results/heuristic_baselines_full.csv
```

Run tests:

```bash
uv run --extra dev pytest -q
```

## Kaggle Workflow

Build the Kaggle artifact:

```bash
uv run --extra kaggle python scripts/build_kaggle_notebook.py
```

Push the notebook to Kaggle:

```bash
python scripts/publish_kaggle_notebook.py
```

The notebook:

- installs `kaggle-benchmarks`,
- installs this repo directly from GitHub,
- requests a Kaggle GPU runtime while spending the competition quota through hosted `kaggle-benchmarks` model calls,
- materializes profile-specific dataset slices (`primary`, `probe`, `holdout`),
- runs the main `goalshield_benchmark` task on the primary slice,
- supports both broad frontier sweeps and quota-constrained novelty-harvest profiles,
- writes live progress updates to `/kaggle/working/goalshield_progress.json`,
- exports per-model, per-difficulty, per-family, and error-profile CSVs to `/kaggle/working`,
- attempts BenchPress novelty analysis when at least 3 mapped models are available.

## References

- Kaggle competition brief: https://www.kaggle.com/competitions/kaggle-measuring-agi
- Kaggle Benchmarks SDK: https://github.com/Kaggle/kaggle-benchmarks/tree/ci
- DeepMind cognitive framework paper: https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/measuring-progress-toward-agi/measuring-progress-toward-agi-a-cognitive-framework.pdf
- BenchPress scaffold: https://github.com/kafkasl/evaluating-agi
