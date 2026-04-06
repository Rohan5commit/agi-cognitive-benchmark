from __future__ import annotations

import argparse
import json
from pathlib import Path

import jupytext


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the Kaggle notebook artifact.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("notebooks/goalshield_benchmark.py"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/kaggle/goalshield-executive-functions-benchmark"),
    )
    parser.add_argument("--username", default="rohansan1")
    parser.add_argument("--slug", default="goalshield-executive-functions-benchmark")
    parser.add_argument("--title", default="GoalShield Executive Functions Benchmark")
    parser.add_argument("--competition", default="kaggle-measuring-agi")
    parser.add_argument("--public", action="store_true")
    return parser


def convert_py_percent_to_ipynb(source: Path, destination: Path) -> None:
    notebook = jupytext.read(source, fmt="py:percent")
    notebook.metadata.setdefault(
        "kernelspec",
        {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
    )
    notebook.metadata.setdefault(
        "language_info",
        {
            "name": "python",
            "version": "3.11",
        },
    )
    jupytext.write(notebook, destination)


def write_metadata(
    output_dir: Path,
    username: str,
    slug: str,
    title: str,
    competition: str,
    is_private: bool,
) -> None:
    metadata = {
        "id": f"{username}/{slug}",
        "title": title,
        "code_file": "benchmark.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": is_private,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [competition] if competition else [],
        "kernel_sources": [],
        "model_sources": [],
        "keywords": ["personal-benchmark"],
    }
    (output_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    convert_py_percent_to_ipynb(args.source, args.output_dir / "benchmark.ipynb")
    write_metadata(
        output_dir=args.output_dir,
        username=args.username,
        slug=args.slug,
        title=args.title,
        competition=args.competition,
        is_private=not args.public,
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
