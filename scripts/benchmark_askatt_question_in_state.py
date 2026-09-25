#!/usr/bin/env python3
"""Test preserving Choice instructions by scoring each question with its state."""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from askatt_system1_api.engine import GLiClassEngine  # noqa: E402


def softmax_logits(scores: list[float]) -> list[float]:
    logits = [
        math.log(min(1 - 1e-6, max(1e-6, score)) / min(1 - 1e-6, max(1e-6, 1 - score)))
        for score in scores
    ]
    peak = max(logits)
    weights = [math.exp(value - peak) for value in logits]
    total = sum(weights)
    return [weight / total for weight in weights]


def main() -> None:
    benchmark = json.loads((ROOT / "data" / "askatt_choice_benchmark.json").read_text())
    engine = GLiClassEngine(ROOT / "models" / "gliclass-modern-large-v3.0", device="cpu")
    records = []
    predictions: dict[str, dict[str, str]] = defaultdict(dict)
    total_tokens = 0
    elapsed = 0.0
    for case in benchmark["cases"]:
        for question_id, question in benchmark["questions"].items():
            options = list(question["criteria"])
            labels = [question["criteria"][option] for option in options]
            scoped_state = f"{case['state']}\n\nClassification question: {question['instructions']}"
            result = engine.score(scoped_state, labels)
            probabilities = softmax_logits(result.scores)
            winner = max(range(len(options)), key=probabilities.__getitem__)
            predicted = options[winner]
            expected = case["expected"][question_id]
            predictions[case["id"]][question_id] = predicted
            total_tokens += result.input_tokens
            elapsed += result.elapsed_seconds
            records.append(
                {
                    "case_id": case["id"],
                    "question_id": question_id,
                    "difficulty": case["difficulty"],
                    "expected": expected,
                    "predicted": predicted,
                    "correct": predicted == expected,
                }
            )

    by_question: dict[str, list[bool]] = defaultdict(list)
    for record in records:
        by_question[record["question_id"]].append(record["correct"])
    exact = sum(
        all(predictions[case["id"]][key] == value for key, value in case["expected"].items())
        for case in benchmark["cases"]
    )
    output = {
        "metadata": {
            "model": engine.model_name,
            "strategy": "Append each instruction to the state and score that question's option definitions in a separate pass.",
            "states": len(benchmark["cases"]),
            "forward_passes_per_state": len(benchmark["questions"]),
        },
        "accuracy": sum(record["correct"] for record in records) / len(records),
        "correct": sum(record["correct"] for record in records),
        "decisions": len(records),
        "exact_state_match_rate": exact / len(benchmark["cases"]),
        "accuracy_by_question": {
            key: sum(values) / len(values) for key, values in sorted(by_question.items())
        },
        "input_tokens_mean_per_state": total_tokens / len(benchmark["cases"]),
        "model_inference_seconds": elapsed,
        "records": records,
    }
    path = ROOT / "results" / "askatt_choice_question_in_state.json"
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
