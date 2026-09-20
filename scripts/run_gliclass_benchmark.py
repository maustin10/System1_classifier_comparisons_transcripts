#!/usr/bin/env python3
"""Run a GLiClass checkpoint on the locked validation and test splits.

The 27 natural-language attribute descriptions are supplied together when the
context window permits, or split into deterministic label groups to preserve
the full transcript. Validation probabilities are saved so thresholds can be
calibrated without looking at held-out test answers.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data" / "benchmark_1000"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_classifiers_shared import ATTRIBUTE_DESCRIPTIONS  # noqa: E402


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--name", required=True, help="Result-file/model key")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--label-batches",
        type=int,
        default=1,
        help="Split the 27 labels across this many forward passes per transcript",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test row limit")
    parser.add_argument("--output-dir", type=Path, default=WORK)
    parser.add_argument(
        "--parts",
        nargs="+",
        choices=("validation", "test"),
        default=("validation", "test"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = read(WORK / "split.json")
    attributes = split["metadata"]["attributes"]
    if attributes != list(ATTRIBUTE_DESCRIPTIONS):
        raise RuntimeError("Attribute schema differs from the benchmark schema")

    import torch
    from gliclass import GLiClassModel, ZeroShotClassificationPipeline
    from transformers import AutoTokenizer

    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        local_files_only=True,
        add_prefix_space=True,
    )
    model = GLiClassModel.from_pretrained(
        args.model_dir,
        local_files_only=True,
    )
    model.to("cpu")
    model.eval()
    load_seconds = time.perf_counter() - load_started

    encoder_limit = int(getattr(model.config.encoder_config, "max_position_embeddings", 512))
    max_length = min(1024, encoder_limit)
    pipeline = ZeroShotClassificationPipeline(
        model,
        tokenizer,
        max_classes=len(attributes),
        max_length=max_length,
        classification_type="multi-label",
        device="cpu",
        progress_bar=True,
    )
    labels = [ATTRIBUTE_DESCRIPTIONS[attribute] for attribute in attributes]
    label_to_attribute = dict(zip(labels, attributes, strict=True))
    if args.label_batches < 1 or args.label_batches > len(labels):
        raise ValueError("--label-batches must be between 1 and the number of labels")
    label_groups = [labels[index :: args.label_batches] for index in range(args.label_batches)]

    # Confirm the checkpoint really emits all 27 labels together.  The model
    # configs retain a historical 25-class default, while current GLiClass uses
    # dynamic label allocation at inference.
    sample = next(row for row in split["rows"] if row["split"] == args.parts[0])
    warmup = [
        item
        for group in label_groups
        for item in pipeline(
            sample["conversation"],
            group,
            threshold=0.0,
            batch_size=1,
        )[0]
    ]
    if len(warmup) != len(attributes):
        raise RuntimeError(
            f"Checkpoint returned {len(warmup)} labels; expected {len(attributes)}"
        )

    for part in args.parts:
        rows = [row for row in split["rows"] if row["split"] == part]
        if args.limit is not None:
            rows = rows[: args.limit]
        texts = [row["conversation"] for row in rows]

        # This measures the exact serialized GLiClass input, including all 27
        # label descriptions, before model truncation.
        serialized = [
            pipeline.pipe.prepare_input(text, group)
            for text in texts
            for group in label_groups
        ]
        input_lengths = [
            len(tokenizer(value, add_special_tokens=True, truncation=False)["input_ids"])
            for value in serialized
        ]

        started = time.perf_counter()
        grouped_results = [
            pipeline(
                texts,
                group,
                threshold=0.0,
                batch_size=args.batch_size,
            )
            for group in label_groups
        ]
        raw_results = [
            [item for group_results in grouped_results for item in group_results[row_index]]
            for row_index in range(len(rows))
        ]
        elapsed = time.perf_counter() - started

        output_rows: dict[str, Any] = {}
        for row, result in zip(rows, raw_results, strict=True):
            scores = {
                label_to_attribute[item["label"]]: float(item["score"])
                for item in result
            }
            if list(scores) != attributes:
                # The pipeline preserves label order today; reconstruct it
                # explicitly so downstream schema validation is deterministic.
                if set(scores) != set(attributes):
                    raise RuntimeError(f"Wrong label schema for row {row['id']}")
                scores = {attribute: scores[attribute] for attribute in attributes}
            output_rows[str(row["id"])] = {
                "prediction": {
                    attribute: int(scores[attribute] >= 0.5)
                    for attribute in attributes
                },
                "probabilities": {
                    attribute: {
                        "present": round(scores[attribute], 8),
                        "absent": round(1.0 - scores[attribute], 8),
                    }
                    for attribute in attributes
                },
            }

        suffix = "validation_results" if part == "validation" else "results"
        output = {
            "metadata": {
                "model": args.model_dir.name,
                "model_path": str(args.model_dir),
                "method": "GLiClass uni-encoder zero-shot multi-label classification",
                "part": part,
                "rows": len(rows),
                "attributes": len(attributes),
                "labels_per_forward_pass_min": min(len(group) for group in label_groups),
                "labels_per_forward_pass_max": max(len(group) for group in label_groups),
                "forward_passes_per_transcript": len(label_groups),
                "threshold": 0.5,
                "batch_size": args.batch_size,
                "max_length": max_length,
                "pretruncation_input_tokens_min": min(input_lengths),
                "pretruncation_input_tokens_mean": sum(input_lengths) / len(input_lengths),
                "pretruncation_input_tokens_max": max(input_lengths),
                "truncated_forward_pass_inputs": sum(
                    length > max_length for length in input_lengths
                ),
                "transcripts_with_any_truncation": sum(
                    any(
                        input_lengths[row_index * len(label_groups) + group_index] > max_length
                        for group_index in range(len(label_groups))
                    )
                    for row_index in range(len(rows))
                ),
                "model_load_seconds": load_seconds,
                "elapsed_seconds": elapsed,
                "decisions_per_second": len(rows) * len(attributes) / elapsed,
                "timing_scope": "CPU inference after model load and one warm-up row; input serialization and output JSON are excluded.",
                "runtime": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "device": "cpu",
                    "machine": platform.machine(),
                },
            },
            "rows": output_rows,
        }
        path = args.output_dir / f"{args.name}_{suffix}.json"
        write(path, output)
        print(json.dumps({"output": str(path), **output["metadata"]}, indent=2))


if __name__ == "__main__":
    main()
