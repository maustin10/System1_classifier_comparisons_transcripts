#!/usr/bin/env python3
"""Compare zero-shot ModernBERT NLI with AskATT on the Choice benchmark."""
from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zero_shot_decision_poc import Comparison, TransformersNliScorer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "models" / "modernbert-zeroshot",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--cases",
        type=Path,
        default=ROOT / "data" / "askatt_choice_benchmark.json",
    )
    parser.add_argument(
        "--askatt-results",
        type=Path,
        default=ROOT / "results" / "askatt_choice_benchmark.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "modernbert_choice_benchmark.json",
    )
    return parser.parse_args()


def softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def quality_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_question: dict[str, list[bool]] = defaultdict(list)
    by_difficulty: dict[str, list[bool]] = defaultdict(list)
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        by_question[record["question_id"]].append(record["correct"])
        by_difficulty[record["difficulty"]].append(record["correct"])
        confusion[f"{record['question_id']}::{record['expected']}"][record["predicted"]] += 1
    correct = sum(record["correct"] for record in records)
    return {
        "decisions": len(records),
        "correct": correct,
        "errors": len(records) - correct,
        "accuracy": correct / len(records),
        "mean_top_probability": statistics.fmean(record["top_probability"] for record in records),
        "accuracy_by_question": {
            key: sum(values) / len(values) for key, values in sorted(by_question.items())
        },
        "accuracy_by_difficulty": {
            key: sum(values) / len(values) for key, values in sorted(by_difficulty.items())
        },
        "confusion": {key: dict(value) for key, value in sorted(confusion.items())},
    }


def build_comparisons(
    cases: list[dict[str, Any]],
    questions: dict[str, dict[str, Any]],
    *,
    reverse: bool,
) -> tuple[list[Comparison], list[tuple[str, str, str]]]:
    comparisons: list[Comparison] = []
    mapping: list[tuple[str, str, str]] = []
    for case in cases:
        for question_id, question in questions.items():
            premise = (
                f"State evidence:\n{case['state']}\n\n"
                f"Decision question:\n{question['instructions']}"
            )
            options = list(question["criteria"].items())
            if reverse:
                options.reverse()
            for option_id, description in options:
                comparisons.append(
                    Comparison(
                        decision_id=f"{case['id']}::{question_id}",
                        candidate_id=option_id,
                        premise=premise,
                        hypothesis=(
                            f"The correct option is '{option_id}'. "
                            f"Definition: {description}"
                        ),
                    )
                )
                mapping.append((case["id"], question_id, option_id))
    return comparisons, mapping


def run_variant(
    scorer: TransformersNliScorer,
    cases: list[dict[str, Any]],
    questions: dict[str, dict[str, Any]],
    *,
    reverse: bool,
) -> dict[str, Any]:
    comparisons, mapping = build_comparisons(cases, questions, reverse=reverse)
    started = time.perf_counter()
    evidence = scorer.score(comparisons)
    elapsed = time.perf_counter() - started

    grouped: dict[tuple[str, str], list[tuple[str, float]]] = defaultdict(list)
    for (case_id, question_id, option_id), item in zip(mapping, evidence, strict=True):
        grouped[(case_id, question_id)].append((option_id, item.entailment_logit))

    by_case = {case["id"]: case for case in cases}
    predictions: dict[str, dict[str, str]] = defaultdict(dict)
    records: list[dict[str, Any]] = []
    for (case_id, question_id), rows in grouped.items():
        probabilities = softmax([logit for _, logit in rows])
        distribution = {
            option_id: probability
            for (option_id, _), probability in zip(rows, probabilities, strict=True)
        }
        predicted = max(distribution, key=distribution.get)
        expected = by_case[case_id]["expected"][question_id]
        predictions[case_id][question_id] = predicted
        records.append(
            {
                "case_id": case_id,
                "difficulty": by_case[case_id]["difficulty"],
                "question_id": question_id,
                "expected": expected,
                "predicted": predicted,
                "correct": predicted == expected,
                "top_probability": distribution[predicted],
                "probabilities": distribution,
            }
        )

    exact = sum(
        all(predictions[case["id"]][key] == value for key, value in case["expected"].items())
        for case in cases
    )
    return {
        "option_order": "reversed" if reverse else "declared",
        "states": len(cases),
        "decisions": len(records),
        "nli_pairs": len(comparisons),
        "quality": quality_summary(records),
        "exact_state_match_rate": exact / len(cases),
        "timing": {
            "elapsed_seconds": elapsed,
            "pairs_per_second": len(comparisons) / elapsed,
            "decisions_per_second": len(records) / elapsed,
            "scope": "Local CPU NLI scoring after model load; HTTP/network time excluded.",
        },
        "predictions": dict(predictions),
        "records": records,
    }


def paired_comparison(
    cases: list[dict[str, Any]],
    modernbert: dict[str, Any],
    askatt: dict[str, Any],
) -> dict[str, Any]:
    counts = Counter()
    disagreements = []
    for case in cases:
        case_id = case["id"]
        for question_id, expected in case["expected"].items():
            mb = modernbert["predictions"][case_id][question_id]
            aa = askatt["predictions"][case_id][question_id]
            mb_correct = mb == expected
            aa_correct = aa == expected
            if mb_correct and aa_correct:
                counts["both_correct"] += 1
            elif mb_correct:
                counts["modernbert_only_correct"] += 1
            elif aa_correct:
                counts["askatt_only_correct"] += 1
            else:
                counts["both_wrong"] += 1
            if mb != aa:
                disagreements.append(
                    {
                        "case_id": case_id,
                        "question_id": question_id,
                        "expected": expected,
                        "modernbert": mb,
                        "askatt": aa,
                    }
                )
    total = len(cases) * len(cases[0]["expected"])
    return {
        **counts,
        "decisions": total,
        "prediction_disagreements": len(disagreements),
        "prediction_disagreement_rate": len(disagreements) / total,
        "details": disagreements,
    }


def main() -> None:
    args = parse_args()
    benchmark = json.loads(args.cases.read_text())
    askatt = json.loads(args.askatt_results.read_text())["semantic_choice"]

    load_started = time.perf_counter()
    scorer = TransformersNliScorer(
        str(args.model_dir), device=args.device, batch_size=args.batch_size
    )
    model_load_seconds = time.perf_counter() - load_started
    # One warm-up pair keeps model initialization out of the timed comparisons.
    warmup, _ = build_comparisons(
        benchmark["cases"][:1], benchmark["questions"], reverse=False
    )
    scorer.score(warmup[:1])

    declared = run_variant(
        scorer, benchmark["cases"], benchmark["questions"], reverse=False
    )
    reversed_result = run_variant(
        scorer, benchmark["cases"], benchmark["questions"], reverse=True
    )
    flips = []
    for case in benchmark["cases"]:
        for question_id in benchmark["questions"]:
            before = declared["predictions"][case["id"]][question_id]
            after = reversed_result["predictions"][case["id"]][question_id]
            if before != after:
                flips.append(
                    {
                        "case_id": case["id"],
                        "question_id": question_id,
                        "declared": before,
                        "reversed": after,
                    }
                )

    output = {
        "metadata": {
            "model": str(args.model_dir),
            "model_family": "MoritzLaurer/ModernBERT-large-zeroshot-v2.0",
            "method": "Pairwise zero-shot NLI; one state/question/option pair per comparison; softmax over option entailment logits.",
            "model_load_seconds": model_load_seconds,
            "device": args.device,
            "batch_size": args.batch_size,
            "runtime": {"python": platform.python_version(), "machine": platform.machine()},
        },
        "modernbert": {
            "declared_order": declared,
            "reversed_order": reversed_result,
            "option_order_stability": {
                "decisions": declared["decisions"],
                "flips": len(flips),
                "flip_rate": len(flips) / declared["decisions"],
                "details": flips,
            },
        },
        "askatt_reference": {
            "model": "knowledgator/gliclass-modern-large-v3.0",
            "declared_order": {
                "quality": askatt["declared_order"]["quality"],
                "exact_state_match_rate": askatt["declared_order"]["exact_state_match_rate"],
            },
            "reversed_order": {
                "quality": askatt["reversed_order"]["quality"],
                "exact_state_match_rate": askatt["reversed_order"]["exact_state_match_rate"],
            },
            "option_order_stability": askatt["option_order_stability"],
        },
        "paired_declared_order": paired_comparison(
            benchmark["cases"], declared, askatt["declared_order"]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "modernbert_accuracy": declared["quality"]["accuracy"],
                "askatt_accuracy": askatt["declared_order"]["quality"]["accuracy"],
                "modernbert_option_order_flip_rate": output["modernbert"]["option_order_stability"]["flip_rate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
