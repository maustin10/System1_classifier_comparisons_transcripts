#!/usr/bin/env python3
"""Prepare, train, and score the 1,000-transcript synthetic benchmark.

The supervised ModernBERT system is deliberately a frozen encoder plus a
trained 27-label logistic head.  This is a reproducible linear-probe fine-tune:
the ModernBERT representation is reused, while every classification weight and
threshold is learned only from the training and validation partitions.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data" / "benchmark_1000"
INPUT = ROOT / "data" / "synth_input.json"
MODEL_PATH = ROOT / "models" / "modernbert-zeroshot"
TYPESAFE_ENV = ROOT / ".env"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_classifiers_shared import (  # noqa: E402
    ATTRIBUTE_DESCRIPTIONS,
    build_jev_payload,
)
from typesafe_decision_poc import (  # noqa: E402
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    _load_dotenv,
    _resolve_api_key,
    call_api,
)
from zero_shot_decision_poc import Comparison, TransformersNliScorer  # noqa: E402


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def prepare() -> None:
    rows = read_json(INPUT)["rows"]
    if len(rows) != 1000:
        raise RuntimeError(f"Expected 1,000 rows, found {len(rows)}")
    attributes = list(rows[0]["truth"])
    if attributes != list(ATTRIBUTE_DESCRIPTIONS):
        raise RuntimeError("Attribute schema differs from the benchmark schema")

    # Rows 3..1000 cycle through six scenario families.  Take exactly 25 from
    # each family for validation and test; keep the two approved examples in
    # training.  The fixed RNG makes the split reproducible.
    rng = np.random.default_rng(20260919)
    buckets: list[list[int]] = [[] for _ in range(6)]
    for row in rows:
        row_id = int(row["id"])
        if row_id >= 3:
            buckets[(row_id - 3) % 6].append(row_id)
    train_ids = {1, 2}
    validation_ids: set[int] = set()
    test_ids: set[int] = set()
    for bucket in buckets:
        shuffled = np.array(bucket, dtype=np.int64)
        rng.shuffle(shuffled)
        test_ids.update(int(x) for x in shuffled[:25])
        validation_ids.update(int(x) for x in shuffled[25:50])
        train_ids.update(int(x) for x in shuffled[50:])
    if (len(train_ids), len(validation_ids), len(test_ids)) != (700, 150, 150):
        raise RuntimeError("Split sizes are not 700/150/150")

    split_for = {
        **{row_id: "train" for row_id in train_ids},
        **{row_id: "validation" for row_id in validation_ids},
        **{row_id: "test" for row_id in test_ids},
    }
    split_rows = [
        {**row, "split": split_for[int(row["id"])]}
        for row in rows
    ]
    split = {
        "metadata": {
            "seed": 20260919,
            "method": "Scenario-family-stratified deterministic split",
            "train_count": 700,
            "validation_count": 150,
            "test_count": 150,
            "attributes": attributes,
        },
        "rows": split_rows,
    }
    write_json(WORK / "split.json", split)
    blind_test = {
        "attributes": attributes,
        "rows": [
            {"id": row["id"], "conversation": row["conversation"]}
            for row in split_rows
            if row["split"] == "test"
        ],
    }
    write_json(WORK / "blind_test.json", blind_test)

    support: dict[str, dict[str, int]] = {}
    for attribute in attributes:
        support[attribute] = {}
        for part in ("train", "validation", "test"):
            support[attribute][part] = sum(
                int(row["truth"][attribute])
                for row in split_rows
                if row["split"] == part
            )
    write_json(WORK / "split_support.json", support)
    print(json.dumps(split["metadata"], indent=2))


def _load_encoder():
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        torch_dtype="auto",
    )
    model.to("cpu")
    model.eval()
    return torch, tokenizer, model


def extract_embeddings(batch_size: int = 4, max_length: int = 1024) -> None:
    split = read_json(WORK / "split.json")
    rows = split["rows"]
    torch, tokenizer, model = _load_encoder()
    vectors: list[np.ndarray] = []
    started = time.perf_counter()
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        encoded = tokenizer(
            [row["conversation"] for row in chunk],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.inference_mode():
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        vectors.append(pooled.float().cpu().numpy())
        if start == 0 or (start // batch_size + 1) % 25 == 0:
            print(f"embedded {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    embeddings = np.concatenate(vectors, axis=0)
    elapsed = time.perf_counter() - started
    np.savez_compressed(
        WORK / "modernbert_embeddings.npz",
        ids=np.array([row["id"] for row in rows], dtype=np.int64),
        embeddings=embeddings,
    )
    write_json(
        WORK / "embedding_metadata.json",
        {
            "model": str(MODEL_PATH),
            "pooling": "attention-mask-aware mean pooling of final hidden state",
            "max_length": max_length,
            "batch_size": batch_size,
            "rows": len(rows),
            "embedding_size": int(embeddings.shape[1]),
            "elapsed_seconds": elapsed,
        },
    )
    print(json.dumps({"shape": list(embeddings.shape), "elapsed_seconds": elapsed}, indent=2))


def _best_f1_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    best = (-1.0, -1.0, 0.5)
    for threshold in np.linspace(0.05, 0.95, 181):
        prediction = probabilities >= threshold
        tp = int(np.sum((y_true == 1) & prediction))
        fp = int(np.sum((y_true == 0) & prediction))
        fn = int(np.sum((y_true == 1) & ~prediction))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        candidate = (f1, -abs(float(threshold) - 0.5), float(threshold))
        if candidate > best:
            best = candidate
    return best[2]


def train_head() -> None:
    from sklearn.linear_model import LogisticRegression

    split = read_json(WORK / "split.json")
    rows = split["rows"]
    attributes = split["metadata"]["attributes"]
    data = np.load(WORK / "modernbert_embeddings.npz")
    ids = data["ids"]
    embeddings = data["embeddings"]
    by_id = {int(row_id): embeddings[index] for index, row_id in enumerate(ids)}

    def matrix(part: str) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        selected = [row for row in rows if row["split"] == part]
        x = np.stack([by_id[int(row["id"])] for row in selected])
        y = np.array(
            [[int(row["truth"][attribute]) for attribute in attributes] for row in selected],
            dtype=np.int64,
        )
        return x, y, selected

    x_train, y_train, _ = matrix("train")
    x_validation, y_validation, _ = matrix("validation")
    x_test, _, test_rows = matrix("test")
    coefficients = []
    intercepts = []
    thresholds = []
    test_probabilities = np.zeros((len(test_rows), len(attributes)), dtype=np.float64)
    started = time.perf_counter()
    for index, attribute in enumerate(attributes):
        classifier = LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            C=1.0,
            max_iter=2000,
            random_state=20260919,
        )
        classifier.fit(x_train, y_train[:, index])
        validation_probability = classifier.predict_proba(x_validation)[:, 1]
        threshold = _best_f1_threshold(y_validation[:, index], validation_probability)
        test_probabilities[:, index] = classifier.predict_proba(x_test)[:, 1]
        coefficients.append(classifier.coef_[0])
        intercepts.append(float(classifier.intercept_[0]))
        thresholds.append(threshold)
        print(f"trained {index + 1}/{len(attributes)} {attribute}: threshold={threshold:.3f}")
    elapsed = time.perf_counter() - started
    predictions = test_probabilities >= np.array(thresholds)[None, :]
    output_rows: dict[str, Any] = {}
    for row_index, row in enumerate(test_rows):
        output_rows[str(row["id"])] = {
            "prediction": {
                attribute: int(predictions[row_index, col])
                for col, attribute in enumerate(attributes)
            },
            "probabilities": {
                attribute: {
                    "present": round(float(test_probabilities[row_index, col]), 8),
                    "absent": round(1.0 - float(test_probabilities[row_index, col]), 8),
                }
                for col, attribute in enumerate(attributes)
            },
        }
    np.savez_compressed(
        WORK / "trained_modernbert_head.npz",
        coefficients=np.stack(coefficients),
        intercepts=np.array(intercepts),
        thresholds=np.array(thresholds),
        attributes=np.array(attributes),
    )
    write_json(
        WORK / "modernbert_trained_results.json",
        {
            "metadata": {
                "method": "Frozen ModernBERT encoder with 27 supervised logistic heads",
                "training_rows": 700,
                "threshold_tuning_rows": 150,
                "test_rows": 150,
                "threshold_objective": "Maximum per-label F1 on validation only",
                "thresholds": dict(zip(attributes, thresholds)),
                "head_training_elapsed_seconds": elapsed,
                "inference_timing": "Measured separately by the benchmark timing stage",
            },
            "rows": output_rows,
        },
    )
    print(json.dumps({"elapsed_seconds": elapsed, "test_rows": len(test_rows)}, indent=2))


def time_trained_inference(batch_size: int = 4, max_length: int = 1024) -> None:
    split = read_json(WORK / "split.json")
    test_rows = [row for row in split["rows"] if row["split"] == "test"]
    head = np.load(WORK / "trained_modernbert_head.npz")
    weights = head["coefficients"]
    intercepts = head["intercepts"]
    thresholds = head["thresholds"]
    torch, tokenizer, model = _load_encoder()
    started = time.perf_counter()
    produced = 0
    for start in range(0, len(test_rows), batch_size):
        chunk = test_rows[start : start + batch_size]
        encoded = tokenizer(
            [row["conversation"] for row in chunk],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.inference_mode():
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        features = pooled.float().cpu().numpy()
        logits = features @ weights.T + intercepts[None, :]
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        produced += int(np.sum((probabilities >= thresholds[None, :]) | (probabilities < thresholds[None, :])))
    elapsed = time.perf_counter() - started
    write_json(
        WORK / "modernbert_trained_timing.json",
        {
            "elapsed_seconds": elapsed,
            "decisions": len(test_rows) * len(split["metadata"]["attributes"]),
            "timing_scope": "Local CPU inference after model and trained head load; 150 transcript batches producing 27 labels each.",
            "produced": produced,
        },
    )
    print(json.dumps(read_json(WORK / "modernbert_trained_timing.json"), indent=2))


def score_zero_shot(
    batch_size: int = 64,
    *,
    model_path: str | Path = MODEL_PATH,
    output_path: Path | None = None,
    part: str = "test",
) -> None:
    if part not in {"validation", "test"}:
        raise ValueError("part must be 'validation' or 'test'")
    split = read_json(WORK / "split.json")
    attributes = split["metadata"]["attributes"]
    selected_rows = [row for row in split["rows"] if row["split"] == part]
    comparisons: list[Comparison] = []
    mapping: list[tuple[int, str]] = []
    for row in selected_rows:
        for attribute, description in ATTRIBUTE_DESCRIPTIONS.items():
            comparisons.append(
                Comparison(
                    decision_id=attribute,
                    candidate_id="true",
                    premise=row["conversation"],
                    hypothesis=description[0].upper() + description[1:].rstrip(".") + ".",
                )
            )
            mapping.append((int(row["id"]), attribute))
    resolved_model = str(model_path)
    scorer = TransformersNliScorer(resolved_model, device="cpu", batch_size=batch_size)
    started = time.perf_counter()
    evidence = scorer.score(comparisons)
    elapsed = time.perf_counter() - started
    output_rows: dict[str, Any] = {
        str(row["id"]): {"prediction": {}, "probabilities": {}}
        for row in selected_rows
    }
    for (row_id, attribute), item in zip(mapping, evidence):
        present = float(item.entailment_probability)
        output_rows[str(row_id)]["prediction"][attribute] = int(present >= 0.5)
        output_rows[str(row_id)]["probabilities"][attribute] = {
            "present": round(present, 8),
            "absent": round(1.0 - present, 8),
        }
    write_json(
        output_path or WORK / "modernbert_zero_shot_results.json",
        {
            "metadata": {
                "model": resolved_model,
                "part": part,
                "rows": len(selected_rows),
                "decision_rule": "Present when NLI entailment probability is at least 0.50",
                "elapsed_seconds": elapsed,
                "decisions": len(comparisons),
                "timing_scope": (
                    "Local CPU scoring after model load; "
                    f"{len(comparisons):,} transcript-attribute NLI pairs."
                ),
            },
            "rows": output_rows,
        },
    )
    print(json.dumps({"elapsed_seconds": elapsed, "decisions": len(comparisons)}, indent=2))


def score_jev() -> None:
    split = read_json(WORK / "split.json")
    test_rows = [row for row in split["rows"] if row["split"] == "test"]
    _load_dotenv(TYPESAFE_ENV)
    api_key = _resolve_api_key()
    output_path = WORK / "jev_results.json"
    if output_path.exists():
        output = read_json(output_path)
    else:
        output = {"metadata": {}, "rows": {}}
    started = time.perf_counter()
    previously_completed = len(output["rows"])
    for index, row in enumerate(test_rows, start=1):
        key = str(row["id"])
        if key in output["rows"]:
            continue
        response = call_api(build_jev_payload(row), api_key, timeout=120.0, max_retries=5)
        answers = response.get("answers") or {}
        prediction: dict[str, int] = {}
        probabilities: dict[str, dict[str, float]] = {}
        confidence: dict[str, float | None] = {}
        for attribute in ATTRIBUTE_DESCRIPTIONS:
            answer = answers.get(attribute)
            if not answer:
                raise RuntimeError(f"JEV response lacks {attribute} for row {row['id']}")
            probs = {str(k): float(v) for k, v in (answer.get("probabilities") or {}).items()}
            choice = answer.get("choice") or max(probs, key=probs.get)
            prediction[attribute] = int(choice == "present")
            probabilities[attribute] = {
                "present": round(probs.get("present", 0.0), 8),
                "absent": round(probs.get("absent", 0.0), 8),
            }
            confidence[attribute] = answer.get("confidence")
        output["rows"][key] = {
            "prediction": prediction,
            "probabilities": probabilities,
            "confidence": confidence,
            "model": response.get("model"),
            "usage": response.get("usage"),
        }
        write_json(output_path, output)
        print(f"JEV {index}/{len(test_rows)} row {row['id']}", flush=True)
    elapsed = time.perf_counter() - started
    output["metadata"] = {
        "model": DEFAULT_MODEL,
        "endpoint": DEFAULT_ENDPOINT,
        "decision_rule": "API-selected present/absent choice; no truth-tuned gate",
        "elapsed_seconds_current_run": elapsed,
        "new_calls_current_run": len(output["rows"]) - previously_completed,
        "decisions": len(test_rows) * len(ATTRIBUTE_DESCRIPTIONS),
        "timing_scope": "Sequential hosted API calls; includes network and response parsing. Cached calls are excluded from current-run timing.",
    }
    write_json(output_path, output)
    print(json.dumps(output["metadata"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=("prepare", "embed", "train", "time-trained", "zero-shot", "jev"),
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--model",
        help="Hugging Face model id or local model directory for the zero-shot stage.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Result JSON path for the zero-shot stage.",
    )
    parser.add_argument(
        "--part",
        choices=("validation", "test"),
        default="test",
        help="Split partition for the zero-shot stage (default: test).",
    )
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    if args.stage == "prepare":
        prepare()
    elif args.stage == "embed":
        extract_embeddings(batch_size=args.batch_size or 4)
    elif args.stage == "train":
        train_head()
    elif args.stage == "time-trained":
        time_trained_inference(batch_size=args.batch_size or 4)
    elif args.stage == "zero-shot":
        score_zero_shot(
            batch_size=args.batch_size or 64,
            model_path=args.model or MODEL_PATH,
            output_path=args.output,
            part=args.part,
        )
    elif args.stage == "jev":
        score_jev()


if __name__ == "__main__":
    main()
