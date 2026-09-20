#!/usr/bin/env python3
"""Validate and score available systems on the locked synthetic test split."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data" / "benchmark_1000"


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def binary_metrics(pairs: list[tuple[int, int]]) -> dict[str, float | int]:
    tp = sum(y == 1 and p == 1 for y, p in pairs)
    fp = sum(y == 0 and p == 1 for y, p in pairs)
    fn = sum(y == 1 and p == 0 for y, p in pairs)
    tn = sum(y == 0 and p == 0 for y, p in pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": (tp + tn) / len(pairs) if pairs else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main() -> None:
    split = read(WORK / "split.json")
    attributes = split["metadata"]["attributes"]
    test_rows = [row for row in split["rows"] if row["split"] == "test"]
    sources = {
        "modernbert_zero_shot": read(WORK / "modernbert_zero_shot_results.json"),
        "modernbert_trained": read(WORK / "modernbert_trained_results.json"),
        "jev": read(WORK / "jev_results.json"),
        "sol": read(WORK / "sol_results.json"),
        "luna": read(WORK / "luna_results.json"),
    }
    deberta_path = WORK / "deberta_v3_large_zeroshot_v2_c_results.json"
    if deberta_path.exists():
        sources["deberta_v3_large_zero_shot_c"] = read(deberta_path)
    optimized_path = WORK / "modernbert_zero_shot_opt_threshold_results.json"
    if optimized_path.exists():
        sources["modernbert_zero_shot_opt_threshold"] = read(optimized_path)
    expected_ids = [str(row["id"]) for row in test_rows]
    for model, source in sources.items():
        if list(source["rows"]) != expected_ids:
            raise RuntimeError(f"{model} test ID sequence differs from the locked split")
        for row_id in expected_ids:
            prediction = source["rows"][row_id]["prediction"]
            if list(prediction) != attributes:
                raise RuntimeError(f"{model} row {row_id} has the wrong attribute schema")
            if set(prediction.values()) - {0, 1}:
                raise RuntimeError(f"{model} row {row_id} has non-binary output")

    trained_timing = read(WORK / "modernbert_trained_timing.json")
    elapsed = {
        "modernbert_zero_shot": sources["modernbert_zero_shot"]["metadata"]["elapsed_seconds"],
        "modernbert_trained": trained_timing["elapsed_seconds"],
        "jev": sources["jev"]["metadata"]["elapsed_seconds_current_run"],
        "sol": sources["sol"].get("elapsed_seconds"),
        # The Luna rerun measured cached JSON regeneration/validation (0.01s),
        # not model inference.  Exclude it rather than publishing a false speed.
        "luna": None,
    }
    if "deberta_v3_large_zero_shot_c" in sources:
        elapsed["deberta_v3_large_zero_shot_c"] = sources[
            "deberta_v3_large_zero_shot_c"
        ]["metadata"]["elapsed_seconds"]
    if "modernbert_zero_shot_opt_threshold" in sources:
        elapsed["modernbert_zero_shot_opt_threshold"] = sources[
            "modernbert_zero_shot_opt_threshold"
        ]["metadata"]["inference_elapsed_seconds"]
    timing_scope = {
        "modernbert_zero_shot": sources["modernbert_zero_shot"]["metadata"]["timing_scope"],
        "modernbert_trained": trained_timing["timing_scope"],
        "jev": sources["jev"]["metadata"]["timing_scope"],
        "sol": sources["sol"].get("timing_scope", "n.a."),
        "luna": "Codex-hosted model inference timing was not reliably exposed; cached-output validation timing is excluded.",
    }
    if "deberta_v3_large_zero_shot_c" in sources:
        timing_scope["deberta_v3_large_zero_shot_c"] = sources[
            "deberta_v3_large_zero_shot_c"
        ]["metadata"]["timing_scope"]
    if "modernbert_zero_shot_opt_threshold" in sources:
        timing_scope["modernbert_zero_shot_opt_threshold"] = sources[
            "modernbert_zero_shot_opt_threshold"
        ]["metadata"]["timing_scope"]

    report: dict[str, Any] = {
        "metadata": {
            "test_rows": len(test_rows),
            "attributes": len(attributes),
            "decisions": len(test_rows) * len(attributes),
            "split_seed": split["metadata"]["seed"],
            "test_ids": [row["id"] for row in test_rows],
        },
        "models": {},
        "rows": {},
    }
    for model, source in sources.items():
        all_pairs: list[tuple[int, int]] = []
        row_metrics: list[dict[str, float | int]] = []
        per_attribute: dict[str, Any] = {}
        exact = 0
        for row in test_rows:
            prediction = source["rows"][str(row["id"])]["prediction"]
            pairs = [(int(row["truth"][a]), int(prediction[a])) for a in attributes]
            all_pairs.extend(pairs)
            current = binary_metrics(pairs)
            row_metrics.append(current)
            exact += int(current["accuracy"] == 1.0)
        for attribute in attributes:
            per_attribute[attribute] = binary_metrics([
                (
                    int(row["truth"][attribute]),
                    int(source["rows"][str(row["id"])]["prediction"][attribute]),
                )
                for row in test_rows
            ])
        macro_precision = sum(float(v["precision"]) for v in per_attribute.values()) / len(attributes)
        macro_recall = sum(float(v["recall"]) for v in per_attribute.values()) / len(attributes)
        macro_f1 = sum(float(v["f1"]) for v in per_attribute.values()) / len(attributes)
        seconds = elapsed[model]
        report["models"][model] = {
            "micro": binary_metrics(all_pairs),
            "macro_attribute_precision": macro_precision,
            "macro_attribute_recall": macro_recall,
            "macro_attribute_f1": macro_f1,
            "macro_row_precision": sum(float(v["precision"]) for v in row_metrics) / len(row_metrics),
            "macro_row_recall": sum(float(v["recall"]) for v in row_metrics) / len(row_metrics),
            "exact_match_conversations": exact,
            "exact_match_rate": exact / len(test_rows),
            "elapsed_seconds": seconds,
            "decisions_per_second": (len(all_pairs) / seconds) if seconds else None,
            "milliseconds_per_decision": (seconds * 1000 / len(all_pairs)) if seconds else None,
            "timing_scope": timing_scope[model],
            "attributes": per_attribute,
        }

    for row in test_rows:
        row_id = str(row["id"])
        report["rows"][row_id] = {}
        for model, source in sources.items():
            prediction = source["rows"][row_id]["prediction"]
            report["rows"][row_id][model] = binary_metrics([
                (int(row["truth"][a]), int(prediction[a])) for a in attributes
            ])

    (WORK / "comparison_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        model: {
            "accuracy": values["micro"]["accuracy"],
            "precision": values["micro"]["precision"],
            "recall": values["micro"]["recall"],
            "f1": values["micro"]["f1"],
            "exact_match": values["exact_match_rate"],
            "elapsed_seconds": values["elapsed_seconds"],
        }
        for model, values in report["models"].items()
    }, indent=2))


if __name__ == "__main__":
    main()
