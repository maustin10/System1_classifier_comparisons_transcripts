#!/usr/bin/env python3
"""Tune per-attribute GLiClass thresholds on validation, then lock test output."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data" / "benchmark_1000"


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def metrics(truth: list[list[int]], prediction: list[list[int]]) -> dict[str, float | int]:
    pairs = [(y, p) for yr, pr in zip(truth, prediction) for y, p in zip(yr, pr)]
    tp = sum(y == 1 and p == 1 for y, p in pairs)
    fp = sum(y == 0 and p == 1 for y, p in pairs)
    fn = sum(y == 1 and p == 0 for y, p in pairs)
    tn = sum(y == 0 and p == 0 for y, p in pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    exact = sum(yr == pr for yr, pr in zip(truth, prediction)) / len(truth)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": (tp + tn) / len(pairs),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match_rate": exact,
    }


def best_threshold(truth: list[int], probabilities: list[float]) -> float:
    best: tuple[float, float, float, float] | None = None
    selected = 0.5
    for step in range(1, 100):
        threshold = step / 100
        prediction = [int(value >= threshold) for value in probabilities]
        current = metrics([truth], [prediction])
        candidate = (
            float(current["f1"]),
            float(current["precision"]),
            -abs(threshold - 0.5),
            threshold,
        )
        if best is None or candidate > best:
            best = candidate
            selected = threshold
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    split = read(WORK / "split.json")
    attributes = split["metadata"]["attributes"]
    validation_rows = [row for row in split["rows"] if row["split"] == "validation"]
    test_rows = [row for row in split["rows"] if row["split"] == "test"]
    validation = read(WORK / f"{args.name}_validation_results.json")
    test = read(WORK / f"{args.name}_results.json")

    expected_validation_ids = [str(row["id"]) for row in validation_rows]
    expected_test_ids = [str(row["id"]) for row in test_rows]
    if list(validation["rows"]) != expected_validation_ids:
        raise RuntimeError("Validation result IDs differ from the locked split")
    if list(test["rows"]) != expected_test_ids:
        raise RuntimeError("Test result IDs differ from the locked split")

    y_validation = [[int(row["truth"][a]) for a in attributes] for row in validation_rows]
    y_test = [[int(row["truth"][a]) for a in attributes] for row in test_rows]
    p_validation = [
        [validation["rows"][str(row["id"])]["probabilities"][a]["present"] for a in attributes]
        for row in validation_rows
    ]
    p_test = [
        [test["rows"][str(row["id"])]["probabilities"][a]["present"] for a in attributes]
        for row in test_rows
    ]
    thresholds = [
        best_threshold(
            [row[col] for row in y_validation],
            [row[col] for row in p_validation],
        )
        for col in range(len(attributes))
    ]
    default_validation = [[int(p >= 0.5) for p in row] for row in p_validation]
    tuned_validation = [
        [int(p >= thresholds[col]) for col, p in enumerate(row)] for row in p_validation
    ]
    default_test = [[int(p >= 0.5) for p in row] for row in p_test]
    tuned_test = [
        [int(p >= thresholds[col]) for col, p in enumerate(row)] for row in p_test
    ]

    output_rows: dict[str, Any] = {}
    for row_index, row in enumerate(test_rows):
        output_rows[str(row["id"])] = {
            "prediction": {
                attribute: tuned_test[row_index][col]
                for col, attribute in enumerate(attributes)
            },
            "probabilities": test["rows"][str(row["id"])]["probabilities"],
        }

    result = {
        "metadata": {
            "method": "Unchanged GLiClass probabilities with per-attribute F1 thresholds selected on validation only",
            "model": test["metadata"]["model"],
            "training_rows": 0,
            "threshold_tuning_rows": len(validation_rows),
            "test_rows": len(test_rows),
            "threshold_grid": "0.01 through 0.99 in 0.01 increments",
            "thresholds": dict(zip(attributes, thresholds, strict=True)),
            "threshold_summary": {
                "min": min(thresholds),
                "median": sorted(thresholds)[len(thresholds) // 2],
                "mean": sum(thresholds) / len(thresholds),
                "max": max(thresholds),
            },
            "validation_default_0_5": metrics(y_validation, default_validation),
            "validation_tuned": metrics(y_validation, tuned_validation),
            "test_default_0_5": metrics(y_test, default_test),
            "test_tuned": metrics(y_test, tuned_test),
            "inference_elapsed_seconds": test["metadata"]["elapsed_seconds"],
            "timing_scope": "Uses existing GLiClass test probabilities; threshold application time is negligible.",
        },
        "rows": output_rows,
    }
    write(WORK / f"{args.name}_opt_threshold_results.json", result)
    print(json.dumps(result["metadata"], indent=2))


if __name__ == "__main__":
    main()
