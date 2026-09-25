#!/usr/bin/env python3
"""Sweep concise label encodings for AskATT semantic Choice questions."""
from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from askatt_system1_api.engine import GLiClassEngine  # noqa: E402


Format = Callable[[str, str, str], str]


QUESTION_SCOPES = {
    "Which department should own the caller's primary request?": "owning department",
    "What is the operational urgency of the primary request?": "operational urgency level",
    "What is the caller's expressed sentiment?": "caller sentiment",
    "What is the best immediate next action for the primary request?": "immediate next action",
}


FORMATS: dict[str, Format] = {
    "description_only_baseline": lambda question, option, description: description,
    "semantic_scope_definition": lambda question, option, description: (
        f"{QUESTION_SCOPES.get(question, question)}: {description}"
    ),
    "semantic_scope_option_definition": lambda question, option, description: (
        f"{QUESTION_SCOPES.get(question, question)} — {option.replace('_', ' ')}: {description}"
    ),
    "question_colon_definition": lambda question, option, description: f"{question}: {description}",
    "question_newline_definition": lambda question, option, description: f"{question}\n{description}",
    "definition_question_suffix": lambda question, option, description: f"{description}. Question: {question}",
    "question_option_definition_compact": lambda question, option, description: f"{question} {option}: {description}",
    "question_answer_definition": lambda question, option, description: f"{question} Answer: {description}",
    "definition_question_parenthetical": lambda question, option, description: f"{description} ({question})",
}


def softmax_logits(scores: list[float]) -> list[float]:
    logits = [math.log(min(1 - 1e-6, max(1e-6, score)) / min(1 - 1e-6, max(1e-6, 1 - score))) for score in scores]
    peak = max(logits)
    weights = [math.exp(value - peak) for value in logits]
    total = sum(weights)
    return [weight / total for weight in weights]


def main() -> None:
    benchmark = json.loads((ROOT / "data" / "askatt_choice_benchmark.json").read_text())
    engine = GLiClassEngine(ROOT / "models" / "gliclass-modern-large-v3.0", device="cpu")
    output = {}
    for format_name, render in FORMATS.items():
        records = []
        elapsed = 0.0
        total_tokens = 0
        predictions: dict[str, dict[str, str]] = defaultdict(dict)
        for case in benchmark["cases"]:
            compiled = []
            groups = []
            for question_id, question in benchmark["questions"].items():
                start = len(compiled)
                options = list(question["criteria"])
                compiled.extend(
                    render(question["instructions"], option, question["criteria"][option])
                    for option in options
                )
                groups.append((question_id, options, start, len(compiled)))
            result = engine.score(case["state"], compiled)
            elapsed += result.elapsed_seconds
            total_tokens += result.input_tokens
            for question_id, options, start, end in groups:
                probabilities = softmax_logits(result.scores[start:end])
                winner = max(range(len(options)), key=probabilities.__getitem__)
                predicted = options[winner]
                expected = case["expected"][question_id]
                predictions[case["id"]][question_id] = predicted
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
        by_difficulty: dict[str, list[bool]] = defaultdict(list)
        for record in records:
            by_question[record["question_id"]].append(record["correct"])
            by_difficulty[record["difficulty"]].append(record["correct"])
        exact = sum(
            all(predictions[case["id"]][key] == value for key, value in case["expected"].items())
            for case in benchmark["cases"]
        )
        output[format_name] = {
            "accuracy": sum(record["correct"] for record in records) / len(records),
            "correct": sum(record["correct"] for record in records),
            "decisions": len(records),
            "exact_state_match_rate": exact / len(benchmark["cases"]),
            "accuracy_by_question": {
                key: sum(values) / len(values) for key, values in sorted(by_question.items())
            },
            "accuracy_by_difficulty": {
                key: sum(values) / len(values) for key, values in sorted(by_difficulty.items())
            },
            "input_tokens_mean_per_state": total_tokens / len(benchmark["cases"]),
            "model_inference_seconds": elapsed,
            "records": records,
        }
        print(format_name, output[format_name]["accuracy"], flush=True)
    ranked = sorted(output, key=lambda key: output[key]["accuracy"], reverse=True)
    result = {
        "metadata": {
            "model": engine.model_name,
            "purpose": "Select a question-aware label encoding before retraining.",
            "states": len(benchmark["cases"]),
            "decisions_per_format": len(benchmark["cases"]) * len(benchmark["questions"]),
        },
        "ranking": ranked,
        "formats": output,
    }
    path = ROOT / "results" / "askatt_choice_label_format_sweep.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(path), "ranking": ranked}, indent=2))


if __name__ == "__main__":
    main()
