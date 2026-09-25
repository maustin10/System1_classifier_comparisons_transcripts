#!/usr/bin/env python3
"""Benchmark AskATT System1 multiclass Choice behavior with the real model."""
from __future__ import annotations

import argparse
import copy
import json
import platform
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from askatt_system1_api.engine import GLiClassEngine  # noqa: E402
from askatt_system1_api.service import AskATTSystem1Service  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "models" / "gliclass-modern-large-v3.0",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--cases",
        type=Path,
        default=ROOT / "data" / "askatt_choice_benchmark.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "askatt_choice_benchmark.json",
    )
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def reverse_options(questions: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = copy.deepcopy(questions)
    for question in result.values():
        question["criteria"] = dict(reversed(list(question["criteria"].items())))
    return result


def quality_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    correct = sum(record["correct"] for record in records)
    by_question: dict[str, list[bool]] = defaultdict(list)
    by_difficulty: dict[str, list[bool]] = defaultdict(list)
    confidence = []
    wrong_high_confidence = 0
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        by_question[record["question_id"]].append(record["correct"])
        by_difficulty[record["difficulty"]].append(record["correct"])
        confidence.append(float(record["confidence"]))
        wrong_high_confidence += int(
            not record["correct"] and float(record["confidence"]) >= 0.5
        )
        confusion[f"{record['question_id']}::{record['expected']}"][record["predicted"]] += 1
    return {
        "decisions": total,
        "correct": correct,
        "errors": total - correct,
        "accuracy": correct / total if total else 0.0,
        "mean_confidence": statistics.fmean(confidence) if confidence else 0.0,
        "wrong_at_confidence_ge_0_5": wrong_high_confidence,
        "wrong_high_confidence_rate": wrong_high_confidence / (total - correct)
        if total > correct
        else 0.0,
        "accuracy_by_question": {
            key: sum(values) / len(values) for key, values in sorted(by_question.items())
        },
        "accuracy_by_difficulty": {
            key: sum(values) / len(values) for key, values in sorted(by_difficulty.items())
        },
        "confusion": {key: dict(value) for key, value in sorted(confusion.items())},
    }


def run_semantic_variant(
    service: AskATTSystem1Service,
    cases: list[dict[str, Any]],
    questions: dict[str, dict[str, Any]],
    *,
    order: str,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    case_predictions: dict[str, dict[str, str]] = {}
    total_inference = 0.0
    total_input_tokens = 0
    total_forward_passes = 0
    total_compiled_labels = 0
    truncated = 0
    wall_started = time.perf_counter()
    for index, case in enumerate(cases, start=1):
        response, metadata = service.evaluate(
            {
                "model": "jev-latest",
                "state": case["state"],
                "questions": questions,
            }
        )
        predictions: dict[str, str] = {}
        for question_id, expected in case["expected"].items():
            answer = response["answers"][question_id]
            predicted = str(answer["choice"])
            probabilities = answer["probabilities"]
            expected_options = set(questions[question_id]["criteria"])
            if set(probabilities) != expected_options:
                raise ValueError(f"Probability keys do not match options for {question_id}")
            if predicted not in expected_options:
                raise ValueError(f"Invalid choice for {question_id}: {predicted}")
            if abs(sum(probabilities.values()) - 1.0) > 1e-6:
                raise ValueError(f"Probabilities do not sum to one for {question_id}")
            predictions[question_id] = predicted
            records.append(
                {
                    "case_id": case["id"],
                    "difficulty": case["difficulty"],
                    "question_id": question_id,
                    "expected": expected,
                    "predicted": predicted,
                    "correct": predicted == expected,
                    "confidence": float(answer["confidence"]),
                    "probabilities": probabilities,
                    "probability_sum": sum(probabilities.values()),
                }
            )
        case_predictions[case["id"]] = predictions
        total_inference += float(metadata["inference_seconds"])
        total_input_tokens += int(metadata["input_tokens"])
        total_forward_passes += int(metadata["forward_passes"])
        total_compiled_labels += int(metadata["compiled_labels"])
        truncated += int(bool(metadata["truncated"]))
        print(f"AskATT Choice {order} {index}/{len(cases)}", flush=True)
    wall_seconds = time.perf_counter() - wall_started
    exact_cases = sum(
        all(case_predictions[case["id"]][key] == expected for key, expected in case["expected"].items())
        for case in cases
    )
    return {
        "option_order": order,
        "states": len(cases),
        "questions_per_state": len(questions),
        "quality": quality_summary(records),
        "exact_state_match_rate": exact_cases / len(cases),
        "timing": {
            "wall_seconds": wall_seconds,
            "model_inference_seconds": total_inference,
            "mean_wall_ms_per_state": 1000 * wall_seconds / len(cases),
            "states_per_second_wall": len(cases) / wall_seconds,
            "decisions_per_second_wall": len(records) / wall_seconds,
            "scope": "Local CPU sequential service calls after warm-up; HTTP/network time excluded.",
        },
        "usage": {
            "input_tokens_total": total_input_tokens,
            "input_tokens_mean_per_state": total_input_tokens / len(cases),
            "forward_passes_total": total_forward_passes,
            "forward_passes_mean_per_state": total_forward_passes / len(cases),
            "compiled_labels_mean_per_state": total_compiled_labels / len(cases),
            "truncated_states": truncated,
        },
        "predictions": case_predictions,
        "records": records,
    }


def cardinality_questions(option_count: int, target_index: int) -> tuple[dict[str, Any], str]:
    criteria: dict[str, str] = {}
    for index in range(option_count):
        option = f"route_{index + 1:03d}"
        token = f"DESTINATION-{index + 1:03d}"
        criteria[option] = f"The approved routing destination is {token}"
    target = f"route_{target_index + 1:03d}"
    state = (
        "A routing controller has selected exactly one destination. "
        f"The approved routing destination is DESTINATION-{target_index + 1:03d}. "
        "Choose that destination and ignore every other candidate."
    )
    return {
        "model": "jev-latest",
        "state": state,
        "questions": {
            "destination": {
                "type": "choice",
                "instructions": "Which routing destination was explicitly approved?",
                "criteria": criteria,
            }
        },
    }, target


def run_cardinality(service: AskATTSystem1Service) -> dict[str, Any]:
    records = []
    for option_count in (2, 4, 8, 16, 32, 65):
        target_indexes = sorted({0, option_count // 2, option_count - 1})
        for target_index in target_indexes:
            payload, expected = cardinality_questions(option_count, target_index)
            response, metadata = service.evaluate(payload)
            answer = response["answers"]["destination"]
            records.append(
                {
                    "option_count": option_count,
                    "target_position": target_index,
                    "expected": expected,
                    "predicted": answer["choice"],
                    "correct": answer["choice"] == expected,
                    "confidence": answer["confidence"],
                    "forward_passes": metadata["forward_passes"],
                    "compiled_labels": metadata["compiled_labels"],
                    "input_tokens": metadata["input_tokens"],
                    "inference_seconds": metadata["inference_seconds"],
                    "truncated": metadata["truncated"],
                }
            )
            print(
                f"AskATT Choice cardinality {option_count} options target {target_index + 1}",
                flush=True,
            )
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[int(record["option_count"])].append(record)
    return {
        "cases": len(records),
        "accuracy": sum(record["correct"] for record in records) / len(records),
        "by_option_count": {
            str(option_count): {
                "cases": len(group),
                "accuracy": sum(record["correct"] for record in group) / len(group),
                "mean_inference_ms": 1000 * statistics.fmean(record["inference_seconds"] for record in group),
                "mean_forward_passes": statistics.fmean(record["forward_passes"] for record in group),
                "mean_input_tokens": statistics.fmean(record["input_tokens"] for record in group),
            }
            for option_count, group in sorted(grouped.items())
        },
        "records": records,
        "note": "Synthetic exact-match stress test for option cardinality and the 64-label pass boundary; not a semantic benchmark.",
    }


def main() -> None:
    args = parse_args()
    benchmark = json.loads(args.cases.read_text())
    cases = benchmark["cases"][: args.limit] if args.limit else benchmark["cases"]
    questions = benchmark["questions"]

    load_started = time.perf_counter()
    engine = GLiClassEngine(args.model_dir, device=args.device, max_labels=64)
    service = AskATTSystem1Service(engine, labels_per_pass=64)
    device = engine.device
    model_load_seconds = time.perf_counter() - load_started

    service.evaluate(
        {"model": "jev-latest", "state": cases[0]["state"], "questions": questions}
    )
    normal = run_semantic_variant(service, cases, questions, order="declared")
    reversed_result = run_semantic_variant(
        service, cases, reverse_options(questions), order="reversed"
    )

    flips = []
    for case in cases:
        case_id = case["id"]
        for question_id in case["expected"]:
            before = normal["predictions"][case_id][question_id]
            after = reversed_result["predictions"][case_id][question_id]
            if before != after:
                flips.append(
                    {
                        "case_id": case_id,
                        "question_id": question_id,
                        "declared": before,
                        "reversed": after,
                    }
                )
    total_semantic_decisions = len(cases) * len(questions)
    cardinality = run_cardinality(service)
    output = {
        "metadata": {
            "api": "Askatt_system1_api",
            "benchmark": benchmark["metadata"],
            "model": engine.model_name,
            "model_path": str(args.model_dir),
            "device": device,
            "model_load_seconds": model_load_seconds,
            "max_length": engine.max_length,
            "labels_per_pass": service.labels_per_pass,
            "choice_instruction_mode": service.choice_instruction_mode,
            "runtime": {"python": platform.python_version(), "machine": platform.machine()},
        },
        "semantic_choice": {
            "declared_order": normal,
            "reversed_order": reversed_result,
            "option_order_stability": {
                "decisions": total_semantic_decisions,
                "flips": len(flips),
                "flip_rate": len(flips) / total_semantic_decisions,
                "details": flips,
            },
        },
        "cardinality_stress": cardinality,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "declared_accuracy": normal["quality"]["accuracy"],
                "reversed_accuracy": reversed_result["quality"]["accuracy"],
                "option_order_flip_rate": output["semantic_choice"]["option_order_stability"]["flip_rate"],
                "cardinality_accuracy": cardinality["accuracy"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
