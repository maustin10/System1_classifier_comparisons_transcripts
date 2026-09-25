#!/usr/bin/env python3
"""Run TypeSafe.ai JEV on the shared semantic Choice benchmark."""
from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from typesafe_decision_poc import (  # noqa: E402
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    TypeSafeError,
    _load_dotenv,
    _resolve_api_key,
    call_api,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        type=Path,
        default=ROOT / "data" / "askatt_choice_benchmark.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "jev_semantic_choice_benchmark.json",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "results" / ".jev_semantic_choice_api_cache.json",
    )
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def reverse_options(questions: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = copy.deepcopy(questions)
    for question in result.values():
        question["criteria"] = dict(reversed(list(question["criteria"].items())))
    return result


def quality_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_question: dict[str, list[bool]] = defaultdict(list)
    by_difficulty: dict[str, list[bool]] = defaultdict(list)
    confidence = []
    top_probability = []
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        by_question[record["question_id"]].append(record["correct"])
        by_difficulty[record["difficulty"]].append(record["correct"])
        if record["confidence"] is not None:
            confidence.append(float(record["confidence"]))
        top_probability.append(float(record["top_probability"]))
        confusion[f"{record['question_id']}::{record['expected']}"][record["predicted"]] += 1
    correct = sum(record["correct"] for record in records)
    return {
        "decisions": len(records),
        "correct": correct,
        "errors": len(records) - correct,
        "accuracy": correct / len(records),
        "mean_confidence": statistics.fmean(confidence) if confidence else None,
        "mean_top_probability": statistics.fmean(top_probability),
        "accuracy_by_question": {
            key: sum(values) / len(values) for key, values in sorted(by_question.items())
        },
        "accuracy_by_difficulty": {
            key: sum(values) / len(values) for key, values in sorted(by_difficulty.items())
        },
        "confusion": {key: dict(value) for key, value in sorted(confusion.items())},
    }


def validate_answer(
    answer: dict[str, Any], question_id: str, question: dict[str, Any]
) -> tuple[str, dict[str, float]]:
    predicted = str(answer.get("choice"))
    probabilities = {
        str(key): float(value)
        for key, value in (answer.get("probabilities") or {}).items()
    }
    expected_options = set(question["criteria"])
    if predicted not in expected_options:
        raise TypeSafeError(f"Invalid JEV choice for {question_id}: {predicted!r}")
    if set(probabilities) != expected_options:
        raise TypeSafeError(f"JEV probability keys do not match {question_id} options")
    probability_sum = sum(probabilities.values())
    if abs(probability_sum - 1.0) > 0.02:
        raise TypeSafeError(
            f"JEV probabilities for {question_id} sum to {probability_sum:.6f}"
        )
    return predicted, probabilities


def run_variant(
    cases: list[dict[str, Any]],
    questions: dict[str, dict[str, Any]],
    *,
    order: str,
    api_key: str,
    endpoint: str,
    model: str,
    cache: dict[str, Any],
    cache_path: Path,
    refresh: bool,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    predictions: dict[str, dict[str, str]] = {}
    call_seconds = []
    input_tokens = 0
    output_tokens = 0
    live_calls = 0
    started = time.perf_counter()
    for index, case in enumerate(cases, start=1):
        cache_key = f"{order}:{case['id']}"
        if refresh or cache_key not in cache:
            payload = {
                "model": model,
                "state": case["state"],
                "questions": questions,
            }
            call_started = time.perf_counter()
            response = call_api(payload, api_key, endpoint=endpoint, timeout=120.0, max_retries=5)
            elapsed = time.perf_counter() - call_started
            cache[cache_key] = {"response": response, "elapsed_seconds": elapsed}
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False) + "\n")
            live_calls += 1
        cached = cache[cache_key]
        response = cached["response"]
        call_seconds.append(float(cached.get("elapsed_seconds", 0.0)))
        usage = response.get("usage") or {}
        input_tokens += int(usage.get("input_tokens", 0) or 0)
        output_tokens += int(usage.get("output_tokens", 0) or 0)
        answers = response.get("answers") or {}
        current = {}
        for question_id, expected in case["expected"].items():
            answer = answers.get(question_id)
            if not isinstance(answer, dict):
                raise TypeSafeError(f"Missing JEV answer for {case['id']} / {question_id}")
            predicted, probabilities = validate_answer(
                answer, question_id, questions[question_id]
            )
            current[question_id] = predicted
            records.append(
                {
                    "case_id": case["id"],
                    "difficulty": case["difficulty"],
                    "question_id": question_id,
                    "expected": expected,
                    "predicted": predicted,
                    "correct": predicted == expected,
                    "confidence": answer.get("confidence"),
                    "top_probability": probabilities[predicted],
                    "probabilities": probabilities,
                    "probability_sum": sum(probabilities.values()),
                }
            )
        predictions[case["id"]] = current
        print(f"JEV Choice {order} {index}/{len(cases)}", flush=True)
    wall_seconds = time.perf_counter() - started
    exact = sum(
        all(predictions[case["id"]][key] == value for key, value in case["expected"].items())
        for case in cases
    )
    return {
        "option_order": order,
        "states": len(cases),
        "questions_per_state": len(questions),
        "quality": quality_summary(records),
        "exact_state_match_rate": exact / len(cases),
        "timing": {
            "wall_seconds_current_run": wall_seconds,
            "live_calls_current_run": live_calls,
            "cached_call_seconds_sum": sum(call_seconds),
            "mean_api_call_seconds": statistics.fmean(call_seconds),
            "scope": "Hosted API call wall time including network and response parsing; cached timings are retained when calls are reused.",
        },
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
        "predictions": predictions,
        "records": records,
    }


def paired_counts(
    cases: list[dict[str, Any]],
    first: dict[str, dict[str, str]],
    second: dict[str, dict[str, str]],
) -> dict[str, int | float]:
    counts = Counter()
    for case in cases:
        for question_id, expected in case["expected"].items():
            a = first[case["id"]][question_id] == expected
            b = second[case["id"]][question_id] == expected
            counts[
                "both_correct" if a and b else
                "first_only_correct" if a else
                "second_only_correct" if b else
                "both_wrong"
            ] += 1
    return {**counts, "decisions": len(cases) * len(cases[0]["expected"])}


def main() -> None:
    args = parse_args()
    if args.env_file:
        _load_dotenv(args.env_file)
    else:
        for candidate in (
            ROOT / ".env",
            Path.home() / "Finops" / "zero-shot-poc" / ".env",
        ):
            if candidate.is_file():
                _load_dotenv(candidate)
                break
    api_key = _resolve_api_key()
    benchmark = json.loads(args.cases.read_text())
    cache = json.loads(args.cache.read_text()) if args.cache.exists() else {}
    declared = run_variant(
        benchmark["cases"], benchmark["questions"], order="declared",
        api_key=api_key, endpoint=args.endpoint, model=args.model,
        cache=cache, cache_path=args.cache, refresh=args.refresh,
    )
    reversed_result = run_variant(
        benchmark["cases"], reverse_options(benchmark["questions"]), order="reversed",
        api_key=api_key, endpoint=args.endpoint, model=args.model,
        cache=cache, cache_path=args.cache, refresh=args.refresh,
    )
    flips = []
    for case in benchmark["cases"]:
        for question_id in benchmark["questions"]:
            before = declared["predictions"][case["id"]][question_id]
            after = reversed_result["predictions"][case["id"]][question_id]
            if before != after:
                flips.append({
                    "case_id": case["id"], "question_id": question_id,
                    "declared": before, "reversed": after,
                })

    askatt = json.loads((ROOT / "results" / "askatt_choice_benchmark.json").read_text())["semantic_choice"]["declared_order"]
    modernbert = json.loads((ROOT / "results" / "modernbert_choice_benchmark.json").read_text())["modernbert"]["declared_order"]
    output = {
        "metadata": {
            "model": args.model,
            "endpoint": args.endpoint,
            "benchmark": benchmark["metadata"],
            "note": "API key is never written to cache or results.",
        },
        "jev": {
            "declared_order": declared,
            "reversed_order": reversed_result,
            "option_order_stability": {
                "decisions": declared["quality"]["decisions"],
                "flips": len(flips),
                "flip_rate": len(flips) / declared["quality"]["decisions"],
                "details": flips,
            },
        },
        "references": {
            "askatt": {
                "model": "knowledgator/gliclass-modern-large-v3.0",
                "quality": askatt["quality"],
                "exact_state_match_rate": askatt["exact_state_match_rate"],
                "paired_with_jev": paired_counts(
                    benchmark["cases"], declared["predictions"], askatt["predictions"]
                ),
            },
            "modernbert": {
                "model": "MoritzLaurer/ModernBERT-large-zeroshot-v2.0",
                "quality": modernbert["quality"],
                "exact_state_match_rate": modernbert["exact_state_match_rate"],
                "paired_with_jev": paired_counts(
                    benchmark["cases"], declared["predictions"], modernbert["predictions"]
                ),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "jev_accuracy": declared["quality"]["accuracy"],
        "askatt_accuracy": askatt["quality"]["accuracy"],
        "modernbert_accuracy": modernbert["quality"]["accuracy"],
        "jev_option_order_flip_rate": output["jev"]["option_order_stability"]["flip_rate"],
    }, indent=2))


if __name__ == "__main__":
    main()
