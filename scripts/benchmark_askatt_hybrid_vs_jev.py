#!/usr/bin/env python3
"""Benchmark the hybrid AskATT base/LoRA policy against matched JEV results."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OPEN_SYSTEM = ROOT.parent / "OpenSystem1-classifier"
TRAINING = ROOT.parent / "Training_System1"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from askatt_system1_api.engine import GLiClassEngine  # noqa: E402
from askatt_system1_api.service import AskATTSystem1Service  # noqa: E402
from jev_noul_benchmark import best_threshold, metrics, payload  # noqa: E402


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def semantic_benchmark(service: AskATTSystem1Service) -> dict[str, Any]:
    benchmark = read(ROOT / "data" / "askatt_choice_benchmark.json")
    records = []
    timings = []
    profiles: Counter[str] = Counter()
    for index, case in enumerate(benchmark["cases"], start=1):
        started = time.perf_counter()
        response, metadata = service.evaluate({
            "model": "askatt-system1-hybrid",
            "state": case["state"],
            "questions": benchmark["questions"],
        })
        timings.append(time.perf_counter() - started)
        profiles.update(metadata["question_routing_profiles"].values())
        for question_id, expected in case["expected"].items():
            answer = response["answers"][question_id]
            records.append({
                "case_id": case["id"],
                "difficulty": case["difficulty"],
                "question_id": question_id,
                "expected": expected,
                "predicted": answer["choice"],
                "correct": answer["choice"] == expected,
                "probabilities": answer["probabilities"],
            })
        print(f"Hybrid semantic {index}/{len(benchmark['cases'])}", flush=True)
    by_question: dict[str, list[bool]] = defaultdict(list)
    for record in records:
        by_question[record["question_id"]].append(record["correct"])
    return {
        "decisions": len(records),
        "correct": sum(record["correct"] for record in records),
        "accuracy": sum(record["correct"] for record in records) / len(records),
        "accuracy_by_question": {
            key: sum(values) / len(values) for key, values in sorted(by_question.items())
        },
        "routing_profiles": dict(profiles),
        "wall_seconds": sum(timings),
        "records": records,
    }


def bfcl_benchmark(
    service: AskATTSystem1Service,
    *,
    label: str = "Hybrid",
    routing_profile: str | None = None,
) -> dict[str, Any]:
    benchmark = read(OPEN_SYSTEM / "data" / "bfcl-routing-subset.json")
    records = []
    profiles: Counter[str] = Counter()
    started = time.perf_counter()
    for index, case in enumerate(benchmark["cases"], start=1):
        question = {
            "type": "choice",
            "instructions": case["question"],
            "criteria": case["options"],
        }
        if routing_profile is not None:
            question["routing_profile"] = routing_profile
        response, metadata = service.evaluate({
            "model": "askatt-system1-hybrid",
            "state": case["state"],
            "questions": {case["id"]: question},
        })
        profiles.update(metadata["question_routing_profiles"].values())
        predicted = response["answers"][case["id"]]["choice"]
        records.append({
            "case_id": case["id"],
            "category": case["category"],
            "expected": case["expected"],
            "predicted": predicted,
            "correct": predicted == case["expected"],
            "probabilities": response["answers"][case["id"]]["probabilities"],
        })
        print(f"{label} BFCL {index}/{len(benchmark['cases'])}", flush=True)
    positives = [record for record in records if record["category"] == "multiple"]
    negatives = [record for record in records if record["category"] == "irrelevance"]
    return {
        "cases": len(records),
        "accuracy": sum(record["correct"] for record in records) / len(records),
        "selection_accuracy": sum(record["correct"] for record in positives) / len(positives),
        "no_tool_recall": sum(record["correct"] for record in negatives) / len(negatives),
        "routing_profiles": dict(profiles),
        "wall_seconds": time.perf_counter() - started,
        "records": records,
    }


def transcript_benchmark(service: AskATTSystem1Service) -> dict[str, Any]:
    split = read(ROOT / "data" / "benchmark_1000" / "split.json")
    attributes = split["metadata"]["attributes"]
    probabilities: dict[str, dict[str, dict[str, float]]] = {}
    total_wall = 0.0
    profile_counts: Counter[str] = Counter()
    for part in ("validation", "test"):
        rows = [row for row in split["rows"] if row["split"] == part]
        part_scores: dict[str, dict[str, float]] = {}
        for index, row in enumerate(rows, start=1):
            started = time.perf_counter()
            response, metadata = service.evaluate(payload(row))
            total_wall += time.perf_counter() - started
            profile_counts.update(metadata["question_routing_profiles"].values())
            part_scores[str(row["id"])] = {
                attribute: float(response["answers"][attribute]["noul"])
                for attribute in attributes
            }
            print(f"Hybrid transcript {part} {index}/{len(rows)}", flush=True)
        probabilities[part] = part_scores

    validation_rows = [row for row in split["rows"] if row["split"] == "validation"]
    test_rows = [row for row in split["rows"] if row["split"] == "test"]
    y_validation = [[int(row["truth"][a]) for a in attributes] for row in validation_rows]
    y_test = [[int(row["truth"][a]) for a in attributes] for row in test_rows]
    p_validation = [[probabilities["validation"][str(row["id"])][a] for a in attributes] for row in validation_rows]
    p_test = [[probabilities["test"][str(row["id"])][a] for a in attributes] for row in test_rows]
    thresholds = [
        best_threshold(
            [row[column] for row in y_validation],
            [row[column] for row in p_validation],
        )
        for column in range(len(attributes))
    ]
    default_test = [[int(value >= 0.5) for value in row] for row in p_test]
    tuned_test = [
        [int(value >= thresholds[column]) for column, value in enumerate(row)]
        for row in p_test
    ]
    return {
        "rows": {"validation": len(validation_rows), "test": len(test_rows)},
        "default_test": metrics(y_test, default_test),
        "validation_tuned_test": metrics(y_test, tuned_test),
        "thresholds": dict(zip(attributes, thresholds, strict=True)),
        "routing_profiles": dict(profile_counts),
        "wall_seconds": total_wall,
        "probabilities": probabilities,
    }


def matched_jev_results() -> dict[str, Any]:
    semantic_path = ROOT / "results" / "jev_semantic_choice_refresh_2026-09-26.json"
    if not semantic_path.exists():
        semantic_path = ROOT / "results" / "jev_semantic_choice_benchmark.json"
    bfcl_path = ROOT / "results" / "jev_bfcl_refresh_2026-09-26.json"
    if not bfcl_path.exists():
        bfcl_path = OPEN_SYSTEM / "results" / "bfcl-routing-typesafe-results.json"
    semantic = read(semantic_path)
    bfcl = read(bfcl_path)
    transcript = read(ROOT / "results" / "jev_noul_comparison_1000.json")
    return {
        "semantic_choice": semantic["jev"]["declared_order"]["quality"],
        "bfcl_routing": bfcl["summary"],
        "transcript_noul": {
            "default_test": transcript["jev_noul_default_0_5"]["test"],
            "validation_tuned_test": transcript["jev_noul_validation_tuned"]["test"],
        },
        "source_artifacts": {
            "semantic_choice": str(semantic_path),
            "bfcl_routing": str(bfcl_path),
            "transcript_noul": str(ROOT / "results" / "jev_noul_comparison_1000.json"),
        },
    }


def base_only_summary(
    hybrid: dict[str, Any],
    bfcl: dict[str, Any],
) -> dict[str, Any]:
    """Record the base baseline without duplicating large decision payloads."""
    semantic = hybrid["semantic_choice"]
    transcript = hybrid["transcript_noul"]
    return {
        "semantic_choice": {
            "same_as": "askatt_hybrid.semantic_choice",
            "reason": "The hybrid policy routes every semantic Choice question to base.",
            "decisions": semantic["decisions"],
            "correct": semantic["correct"],
            "accuracy": semantic["accuracy"],
        },
        "transcript_noul": {
            "same_as": "askatt_hybrid.transcript_noul",
            "reason": "The hybrid policy routes every transcript Noul question to base.",
            "rows": transcript["rows"],
            "default_test": transcript["default_test"],
            "validation_tuned_test": transcript["validation_tuned_test"],
        },
        "bfcl_routing": bfcl,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "models" / "gliclass-modern-large-v3.0",
    )
    parser.add_argument(
        "--lora-checkpoint",
        type=Path,
        default=TRAINING / "checkpoints" / "pilot_2000_lora_r8_last8.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "askatt_hybrid_vs_jev.json",
    )
    parser.add_argument(
        "--only-bfcl",
        action="store_true",
        help="Rerun only BFCL and preserve semantic/transcript results in --output.",
    )
    parser.add_argument(
        "--only-transcript",
        action="store_true",
        help="Rerun only transcripts and preserve choice/routing results in --output.",
    )
    parser.add_argument(
        "--only-baseline-bfcl",
        action="store_true",
        help="Add a base-only BFCL comparison to an existing output artifact.",
    )
    args = parser.parse_args()
    base = GLiClassEngine(args.model_dir, device="cpu", max_labels=64)
    if args.only_baseline_bfcl:
        if not args.output.exists():
            raise FileNotFoundError(
                "--only-baseline-bfcl requires an existing output artifact"
            )
        result = read(args.output)
        baseline_service = AskATTSystem1Service(base, labels_per_pass=64)
        baseline_bfcl = bfcl_benchmark(
            baseline_service,
            label="Base-only",
            routing_profile="base",
        )
        result["askatt_baseline"] = base_only_summary(
            result["askatt_hybrid"], baseline_bfcl
        )
        result["jev"] = matched_jev_results()
        write(args.output, result)
        print(json.dumps({"output": str(args.output)}, indent=2))
        return
    tool = GLiClassEngine(
        args.model_dir,
        device="cpu",
        max_labels=64,
        checkpoint=args.lora_checkpoint,
    )
    service = AskATTSystem1Service(base, tool_engine=tool, labels_per_pass=64)
    if args.only_bfcl:
        if not args.output.exists():
            raise FileNotFoundError("--only-bfcl requires an existing output artifact")
        result = read(args.output)
        result["askatt_hybrid"]["bfcl_routing"] = bfcl_benchmark(service)
        baseline_service = AskATTSystem1Service(base, labels_per_pass=64)
        baseline_bfcl = bfcl_benchmark(
            baseline_service,
            label="Base-only",
            routing_profile="base",
        )
        result["askatt_baseline"] = base_only_summary(
            result["askatt_hybrid"], baseline_bfcl
        )
        result["jev"] = matched_jev_results()
        write(args.output, result)
        print(json.dumps({"output": str(args.output)}, indent=2))
        return
    if args.only_transcript:
        if not args.output.exists():
            raise FileNotFoundError("--only-transcript requires an existing output artifact")
        result = read(args.output)
        result["askatt_hybrid"]["transcript_noul"] = transcript_benchmark(service)
        result["jev"] = matched_jev_results()
        write(args.output, result)
        print(json.dumps({"output": str(args.output)}, indent=2))
        return
    result = {
        "metadata": {
            "policy": "base for general semantic/transcript questions; LoRA for tool routing",
            "base_model": str(args.model_dir),
            "tool_lora_checkpoint": str(args.lora_checkpoint),
            "threshold_policy": "transcript thresholds selected on validation only; choices use argmax",
            "noul_boundary_mode": "positive (paired mode tested separately and rejected)",
            "jev_refresh": {
                "semantic_choice": "live 2026-09-26",
                "bfcl_routing": "live 2026-09-26",
                "transcript_noul": "existing locked 300-call live run",
            },
        },
        "askatt_hybrid": {},
        "jev": matched_jev_results(),
    }
    result["askatt_hybrid"]["semantic_choice"] = semantic_benchmark(service)
    result["askatt_hybrid"]["transcript_noul"] = transcript_benchmark(service)
    result["askatt_hybrid"]["bfcl_routing"] = bfcl_benchmark(service)
    baseline_service = AskATTSystem1Service(base, labels_per_pass=64)
    baseline_bfcl = bfcl_benchmark(
        baseline_service,
        label="Base-only",
        routing_profile="base",
    )
    result["askatt_baseline"] = base_only_summary(
        result["askatt_hybrid"], baseline_bfcl
    )
    write(args.output, result)
    print(json.dumps({"output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
