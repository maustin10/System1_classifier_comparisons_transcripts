#!/usr/bin/env python3
"""Benchmark the GLiClass-backed AskATT System1 Choice and Noul adapters."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from askatt_system1_api.engine import GLiClassEngine  # noqa: E402
from askatt_system1_api.service import AskATTSystem1Service  # noqa: E402
from run_classifiers_shared import ATTRIBUTE_DESCRIPTIONS  # noqa: E402

WORK = ROOT / "data" / "benchmark_1000"


def metrics(truth: list[list[int]], prediction: list[list[int]]) -> dict[str, float | int]:
    pairs = [(actual, predicted) for row_a, row_p in zip(truth, prediction, strict=True) for actual, predicted in zip(row_a, row_p, strict=True)]
    tp = sum(actual == 1 and predicted == 1 for actual, predicted in pairs)
    fp = sum(actual == 0 and predicted == 1 for actual, predicted in pairs)
    fn = sum(actual == 1 and predicted == 0 for actual, predicted in pairs)
    tn = sum(actual == 0 and predicted == 0 for actual, predicted in pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    exact = sum(actual == predicted for actual, predicted in zip(truth, prediction, strict=True)) / len(truth)
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
        "errors": fp + fn,
    }


def questions(kind: str) -> dict[str, dict[str, Any]]:
    result = {}
    for attribute, description in ATTRIBUTE_DESCRIPTIONS.items():
        # GLiClass treats the positive criterion as its runtime label. Keep that
        # criterion concise; verbose yes/no prose materially degrades a
        # uni-encoder even though it can help an instruction-following model.
        true_rule = description
        false_rule = f"The conversation does not contain evidence that {description}."
        instructions = f"Is this attribute explicitly present in the conversation: {description}?"
        if kind == "noul":
            result[attribute] = {
                "type": "noul",
                "instructions": instructions,
                "criteria": {"true": true_rule, "false": false_rule},
            }
        else:
            result[attribute] = {
                "type": "choice",
                "instructions": instructions,
                "criteria": {
                    "absent": false_rule,
                    "present": true_rule,
                },
            }
    return result


def run_variant(
    service: AskATTSystem1Service,
    rows: list[dict[str, Any]],
    kind: str,
    attributes: list[str],
) -> dict[str, Any]:
    request_questions = questions(kind)
    predictions: list[list[int]] = []
    probabilities: dict[str, dict[str, float]] = {}
    total_inference = 0.0
    total_input_tokens = 0
    total_forward_passes = 0
    truncated_rows = 0
    sample_response = None
    wall_started = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        response, metadata = service.evaluate(
            {
                "model": "jev-latest",
                "state": row["conversation"],
                "questions": request_questions,
            }
        )
        current = []
        row_probabilities = {}
        for attribute in attributes:
            answer = response["answers"][attribute]
            if kind == "noul":
                probability = float(answer["noul"])
                current.append(int(probability >= 0.5))
            else:
                probability = float(answer["probabilities"]["present"])
                current.append(int(answer["choice"] == "present"))
            row_probabilities[attribute] = probability
        predictions.append(current)
        probabilities[str(row["id"])] = row_probabilities
        total_inference += float(metadata["inference_seconds"])
        total_input_tokens += int(metadata["input_tokens"])
        total_forward_passes += int(metadata["forward_passes"])
        truncated_rows += int(bool(metadata["truncated"]))
        sample_response = sample_response or response
        if index % 10 == 0 or index == len(rows):
            print(f"AskATT {kind} {index}/{len(rows)}", flush=True)
    wall_seconds = time.perf_counter() - wall_started
    truth = [[int(row["truth"][attribute]) for attribute in attributes] for row in rows]
    quality = metrics(truth, predictions)
    return {
        "question_type": kind,
        "rows": len(rows),
        "questions_per_row": len(attributes),
        "decisions": len(rows) * len(attributes),
        "quality": quality,
        "timing": {
            "wall_seconds": wall_seconds,
            "model_inference_seconds": total_inference,
            "mean_wall_ms_per_state": 1000 * wall_seconds / len(rows),
            "mean_model_ms_per_state": 1000 * total_inference / len(rows),
            "states_per_second_wall": len(rows) / wall_seconds,
            "decisions_per_second_wall": len(rows) * len(attributes) / wall_seconds,
            "scope": "Local CPU sequential requests after one warm-up; HTTP/network time excluded.",
        },
        "usage": {
            "input_tokens_total": total_input_tokens,
            "input_tokens_mean_per_state": total_input_tokens / len(rows),
            "forward_passes_total": total_forward_passes,
            "forward_passes_mean_per_state": total_forward_passes / len(rows),
            "truncated_rows": truncated_rows,
        },
        "sample_response": sample_response,
        "probabilities": probabilities,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "models" / "gliclass-modern-large-v3.0",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "askatt_system1_api_benchmark.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = json.loads((WORK / "split.json").read_text())
    attributes = list(split["metadata"]["attributes"])
    rows = [row for row in split["rows"] if row["split"] == "test"]
    if args.limit:
        rows = rows[: args.limit]

    load_started = time.perf_counter()
    engine = GLiClassEngine(args.model_dir, device=args.device, max_labels=64)
    service = AskATTSystem1Service(engine, labels_per_pass=64)
    device = engine.device
    model_load_seconds = time.perf_counter() - load_started

    # Warm both compiled shapes before timing.
    service.evaluate({"model": "jev-latest", "state": rows[0]["conversation"], "questions": questions("noul")})
    service.evaluate({"model": "jev-latest", "state": rows[0]["conversation"], "questions": questions("choice")})

    variants = {
        kind: run_variant(service, rows, kind, attributes)
        for kind in ("noul", "choice")
    }
    output = {
        "metadata": {
            "api": "Askatt_system1_api",
            "version": "0.1.0",
            "model": engine.model_name,
            "model_path": str(args.model_dir),
            "device": device,
            "model_load_seconds": model_load_seconds,
            "max_length": engine.max_length,
            "labels_per_pass": service.labels_per_pass,
            "split": "locked test",
            "rows": len(rows),
            "attributes": len(attributes),
            "runtime": {
                "python": platform.python_version(),
                "machine": platform.machine(),
            },
            "compatibility": {
                "endpoint": "/v1/systemone",
                "types": ["noul", "choice"],
                "choice_confidence": "normalized entropy of the AskATT probability distribution; not TypeSafe confidence",
                "choice_probability_mapping": "binary complements for present/absent; otherwise softmax over log-odds of GLiClass option scores",
            },
        },
        "variants": variants,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    summary = {
        kind: {"quality": result["quality"], "timing": result["timing"], "usage": result["usage"]}
        for kind, result in variants.items()
    }
    print(json.dumps({"output": str(args.output), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
