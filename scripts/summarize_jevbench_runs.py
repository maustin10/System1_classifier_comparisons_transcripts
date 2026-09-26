#!/usr/bin/env python3
"""Create a compact comparison from public JevBench cohort results."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

COHORTS = ("original", "easy", "hard")


def summarize(jevbench: Path, tasks: Path, results: Path) -> dict:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "jevbench.cli",
            "summarize",
            "--tasks",
            str(tasks),
            "--results",
            str(results),
        ],
        cwd=jevbench,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def parse_system(value: str) -> tuple[str, Path]:
    try:
        name, directory = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("system must be NAME=DIRECTORY") from error
    return name, Path(directory).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jevbench-dir", type=Path, required=True)
    parser.add_argument("--system", action="append", type=parse_system, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    jevbench = args.jevbench_dir.resolve()
    output: dict[str, object] = {
        "protocol": "JevBench public cohorts",
        "jevbench_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=jevbench, text=True
        ).strip(),
        "scope": "231 public decisions only; not an official sealed JevBench submission",
        "systems": {},
    }
    for system_name, directory in args.system:
        cohorts: dict[str, dict[str, object]] = {}
        total_correct = total_planned = 0
        for cohort in COHORTS:
            result_file = directory / f"{cohort}.jsonl"
            if not result_file.is_file():
                cohorts[cohort] = {"status": "not_run"}
                continue
            report = summarize(
                jevbench,
                jevbench / "datasets" / "public" / f"{cohort}.jsonl",
                result_file,
            )
            planned = int(report["n_planned"])
            correct = int(report["n_correct"])
            total_planned += planned
            total_correct += correct
            cohorts[cohort] = {
                "status": "complete" if report["complete"] else "partial",
                "correct": correct,
                "decisions": planned,
                "accuracy": report["accuracy"],
                "brier_mean": report["brier_mean"],
                "ece": report["ece"]["ece"],
                "p50_latency_s": report["latency"]["p50_s"],
                "p95_latency_s": report["latency"]["p95_s"],
                "schema_validity": report["schema_validity_strict"],
                "operational_success": report["operational_success"],
            }
        output["systems"][system_name] = {
            "cohorts": cohorts,
            "public_total": {
                "correct": total_correct,
                "decisions": total_planned,
                "accuracy": total_correct / total_planned if total_planned else None,
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
