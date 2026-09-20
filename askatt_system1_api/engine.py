"""Scoring engines used by the AskATT System1 compatibility service."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence


@dataclass(frozen=True)
class EngineResult:
    scores: list[float]
    input_tokens: int
    elapsed_seconds: float
    truncated: bool


class ScoringEngine(Protocol):
    model_name: str

    def score(self, state: str, labels: Sequence[str]) -> EngineResult:
        """Return one independent 0..1 score per label."""


class GLiClassEngine:
    """Lazy local GLiClass multi-label scorer.

    One call serializes one state plus every supplied label into a shared
    uni-encoder input. The service may issue multiple calls when the configured
    label batch limit is exceeded.
    """

    def __init__(
        self,
        model_dir: str | Path,
        *,
        device: str = "auto",
        max_labels: int = 64,
        max_length: int | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device_request = device
        self.max_labels = max_labels
        self.max_length_request = max_length
        self.model_name = f"askatt-{self.model_dir.name}"
        self._pipeline = None
        self._tokenizer = None
        self._max_length = None
        self._device = None
        self._lock = threading.Lock()

    @property
    def device(self) -> str:
        self._load()
        return str(self._device)

    @property
    def max_length(self) -> int:
        self._load()
        return int(self._max_length)

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        with self._lock:
            if self._pipeline is not None:
                return
            if not self.model_dir.is_dir():
                raise FileNotFoundError(
                    f"GLiClass model directory does not exist: {self.model_dir}"
                )
            import torch
            from gliclass import GLiClassModel, ZeroShotClassificationPipeline
            from transformers import AutoTokenizer

            if self.device_request == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device_request

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir,
                local_files_only=True,
                add_prefix_space=True,
            )
            model = GLiClassModel.from_pretrained(
                self.model_dir,
                local_files_only=True,
            )
            model.to(device)
            model.eval()
            encoder_limit = int(
                getattr(model.config.encoder_config, "max_position_embeddings", 512)
            )
            max_length = min(self.max_length_request or encoder_limit, encoder_limit)
            pipeline = ZeroShotClassificationPipeline(
                model,
                tokenizer,
                max_classes=self.max_labels,
                max_length=max_length,
                classification_type="multi-label",
                device=device,
                progress_bar=False,
            )
            self._tokenizer = tokenizer
            self._pipeline = pipeline
            self._max_length = max_length
            self._device = device

    def score(self, state: str, labels: Sequence[str]) -> EngineResult:
        if not labels:
            return EngineResult([], 0, 0.0, False)
        if len(labels) > self.max_labels:
            raise ValueError(
                f"Engine accepts at most {self.max_labels} labels per forward pass"
            )
        self._load()
        assert self._pipeline is not None
        assert self._tokenizer is not None
        assert self._max_length is not None

        serialized = self._pipeline.pipe.prepare_input(state, list(labels))
        input_tokens = len(
            self._tokenizer(
                serialized,
                add_special_tokens=True,
                truncation=False,
            )["input_ids"]
        )
        started = time.perf_counter()
        with self._lock:
            raw = self._pipeline(
                state,
                list(labels),
                threshold=0.0,
                batch_size=1,
            )[0]
        elapsed = time.perf_counter() - started
        by_label = {str(item["label"]): float(item["score"]) for item in raw}
        missing = [label for label in labels if label not in by_label]
        if missing:
            raise RuntimeError(f"GLiClass omitted {len(missing)} requested labels")
        return EngineResult(
            scores=[min(1.0, max(0.0, by_label[label])) for label in labels],
            input_tokens=input_tokens,
            elapsed_seconds=elapsed,
            truncated=input_tokens > self._max_length,
        )


class DeterministicTestEngine:
    """Small deterministic engine for HTTP and schema tests."""

    model_name = "askatt-deterministic-test-engine"

    def score(self, state: str, labels: Sequence[str]) -> EngineResult:
        del state
        scores = []
        for label in labels:
            lowered = label.lower()
            if any(token in lowered for token in ("urgent", "billing", "present", "yes")):
                scores.append(0.85)
            else:
                scores.append(0.15)
        return EngineResult(
            scores=scores,
            input_tokens=10 + len(labels) * 4,
            elapsed_seconds=0.001,
            truncated=False,
        )

