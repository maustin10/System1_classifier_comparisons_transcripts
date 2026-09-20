#!/usr/bin/env python3
"""Classify typed decisions with a local, zero-shot NLI model.

The module deliberately separates four concerns:

1. ``state`` is factual evidence supplied by the caller.
2. ``decisions`` define the question and permitted typed answers.
3. A scorer turns each (state, question, candidate) pair into NLI evidence.
4. Application code converts evidence into valid Choice, Boolean, and Score
   results. The model never generates JSON or arbitrary category names.

Run ``python3 poc/zero_shot_decision_poc.py --help`` for command-line usage.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol, Sequence

DEFAULT_MODEL = "MoritzLaurer/ModernBERT-large-zeroshot-v2.0"
DecisionKind = Literal["choice", "boolean", "score"]
_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")


class InputValidationError(ValueError):
    """Raised when a decision request cannot produce a safe typed result."""


@dataclass(frozen=True)
class Candidate:
    """One permitted answer for a Choice or Score decision."""

    identifier: str
    description: str
    numeric_value: float | None = None


@dataclass(frozen=True)
class Decision:
    """A typed question interpreted against a shared state payload."""

    identifier: str
    kind: DecisionKind
    question: str
    candidates: tuple[Candidate, ...]
    true_description: str | None = None
    minimum_probability: float = 0.65
    minimum_margin: float = 0.15
    review_option_id: str | None = None


@dataclass(frozen=True)
class DecisionRequest:
    """A shared state object and one or more declared decisions."""

    state: Mapping[str, Any]
    decisions: tuple[Decision, ...]


@dataclass(frozen=True)
class Comparison:
    """A single premise/hypothesis pair sent to the NLI classifier."""

    decision_id: str
    candidate_id: str
    premise: str
    hypothesis: str
    numeric_value: float | None = None


@dataclass(frozen=True)
class NliEvidence:
    """The entailment signal for one comparison.

    ``entailment_logit`` is used to rank mutually exclusive candidates.
    ``entailment_probability`` is used for an independent Boolean claim.
    """

    entailment_logit: float
    entailment_probability: float


class NliScorer(Protocol):
    """An interchangeable classifier backend for premise/hypothesis pairs."""

    def score(self, comparisons: Sequence[Comparison]) -> list[NliEvidence]:
        """Return one NLI evidence value for each comparison, in order."""


def _local_model_hint(model_dir: Path) -> str | None:
    """Diagnose a local model directory that is present but unusable.

    Returns None when nothing is obviously wrong, so the caller can fall back to
    the generic hint.

    The common failure is an incomplete transfer. A Hugging Face cache stores
    large files as symlinks into a two-character sharded blobs/ tree
    (blobs/b9/b987...), and archiving that cache without dereferencing symlinks
    (plain `tar cz` rather than `tar czh`) silently produces a tree where
    config.json and tokenizer.json are real files but model.safetensors is a
    dangling link. transformers then reports only "no file named
    model.safetensors", which reads like the wrong path was passed rather than a
    broken copy.
    """
    if not model_dir.is_dir():
        return None

    weight_names = ("model.safetensors", "pytorch_model.bin")
    broken = [
        name
        for name in weight_names
        if (link := model_dir / name).is_symlink() and not link.exists()
    ]
    if broken:
        targets = "\n".join(
            f"      {name} -> {(model_dir / name).readlink()}" for name in broken
        )
        return "\n".join(
            [
                f"Model weights are missing from {model_dir}.",
                "",
                "The directory exists and config/tokenizer files load, but the weight",
                "file is a DANGLING SYMLINK -- the copy of this model was incomplete:",
                targets,
                "",
                "A Hugging Face cache keeps large files as symlinks into a sharded",
                "blobs/ tree (blobs/b9/b987...). Archiving it without dereferencing",
                "symlinks copies the links but not the data.",
                "",
                "Re-export on the machine that has the full model, dereferencing links:",
                "    tar czhf model.tar.gz .cache/huggingface/hub/<repo-dir>",
                "         ^ the 'h' flag is what inlines the real file contents",
                "",
                "Or download as plain files with no symlinks at all:",
                "    hf download <repo-id> --local-dir ./model-dir",
            ]
        )

    present = {p.name for p in model_dir.iterdir()} if model_dir.is_dir() else set()
    if not present & set(weight_names):
        return "\n".join(
            [
                f"No model weights found in {model_dir}.",
                "",
                f"Expected one of: {', '.join(weight_names)}",
                f"Directory contains: {', '.join(sorted(present)) or '(empty)'}",
                "",
                "If this is a Hugging Face cache directory, pass the snapshot subdirectory:",
                "    <cache>/models--<org>--<name>/snapshots/<commit-sha>",
            ]
        )
    return None


def _model_load_hint(model_name: str, error: Exception) -> str:
    """Explain a model-load failure in terms of its actual cause.

    Hugging Face reports every failed fetch as "couldn't connect ... check your
    internet connection", which is actively misleading behind a filtering proxy:
    the connection succeeds, but an HTML block page is returned in place of the
    model files. huggingface_hub then complains about a missing 'X-Repo-Commit'
    header. Surface that distinction so the reader stops debugging their network.
    """
    # A local path that exists but is unusable has a precise, actionable cause,
    # so prefer that over any network-oriented advice.
    local = _local_model_hint(Path(model_name))
    if local is not None:
        return local

    detail = str(error)

    # Hugging Face re-raises a generic "couldn't connect" OSError, keeping the
    # informative FileMetadataError only in the __cause__/__context__ chain, so
    # the block-page signature must be looked for across the whole chain rather
    # than in str(error) alone.
    chain: list[str] = []
    cursor: BaseException | None = error
    seen: set[int] = set()
    while cursor is not None and id(cursor) not in seen:
        seen.add(id(cursor))
        chain.append(str(cursor))
        cursor = cursor.__cause__ or cursor.__context__
    combined = "\n".join(chain)

    blocked = "X-Repo-Commit" in combined or "does not seem to be served" in combined
    lines = [f"Could not load model {model_name!r}."]

    if blocked:
        lines += [
            "",
            "The Hugging Face endpoint answered, but not with model files -- most",
            "likely an HTTP proxy returned a block/interstitial page instead. On the",
            "AT&T corporate network huggingface.co is blocked by the CSO proxy, so",
            "model weights cannot be downloaded on-network at all.",
            "",
            "Options:",
            "  1. Pre-download the model on an unrestricted network, then copy the",
            "     cache to this machine (default location ~/.cache/huggingface/hub):",
            f"       huggingface-cli download {model_name}",
            "     Afterwards force offline use so no fetch is attempted:",
            "       export HF_HUB_OFFLINE=1",
            "  2. Point --model at an already-downloaded local directory:",
            "       --model /path/to/local/model-dir",
            "  3. Request a HuggingFace remote in Artifactory and set HF_ENDPOINT",
            "     to it (none is currently provisioned).",
        ]
    else:
        lines += [
            "",
            "If this machine is behind a TLS-intercepting proxy, point Python at a",
            "CA bundle that includes the corporate root:",
            "  export SSL_CERT_FILE=~/.config/tokenball/att-ca-bundle.pem",
            "  export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE",
            "",
            "Or use an already-downloaded model directory: --model /path/to/model-dir",
        ]

    lines += ["", f"Underlying error: {detail.splitlines()[0][:200]}"]
    return "\n".join(lines)


class TransformersNliScorer:
    """Local Hugging Face NLI scorer using one forward pass per batch chunk.

    Dependencies are deliberately imported only when this class is created.
    Parsing, schema validation, JSON serialization, and the unit tests therefore
    work without installing PyTorch or downloading a model.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        device: str = "auto",
        batch_size: int = 64,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "Missing local inference dependencies. Install them with "
                "'python3 -m pip install -r poc/requirements-zero-shot-decisions.txt'."
            ) from error

        self._torch = torch
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        except OSError as error:
            raise RuntimeError(_model_load_hint(model_name, error)) from error
        self._device = self._resolve_device(device)
        self._model.to(self._device)
        self._model.eval()
        self._batch_size = batch_size
        self._entailment_index = self._find_entailment_index(self._model.config.label2id)

    def score(self, comparisons: Sequence[Comparison]) -> list[NliEvidence]:
        evidence: list[NliEvidence] = []
        for start in range(0, len(comparisons), self._batch_size):
            chunk = comparisons[start:start + self._batch_size]
            encoded = self._tokenizer(
                [comparison.premise for comparison in chunk],
                [comparison.hypothesis for comparison in chunk],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded = {name: value.to(self._device) for name, value in encoded.items()}
            with self._torch.inference_mode():
                logits = self._model(**encoded).logits
                probabilities = self._torch.softmax(logits, dim=-1)

            for row_logits, row_probabilities in zip(logits, probabilities, strict=True):
                evidence.append(
                    NliEvidence(
                        entailment_logit=float(row_logits[self._entailment_index].item()),
                        entailment_probability=float(
                            row_probabilities[self._entailment_index].item()
                        ),
                    )
                )
        return evidence

    def _resolve_device(self, requested_device: str) -> str:
        if requested_device != "auto":
            return requested_device
        if self._torch.cuda.is_available():
            return "cuda"
        if getattr(self._torch.backends, "mps", None) and self._torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _find_entailment_index(label_to_id: Mapping[str, int]) -> int:
        for label, index in label_to_id.items():
            if label.casefold().replace("_", " ") == "entailment":
                return int(index)
        available = ", ".join(sorted(label_to_id))
        raise RuntimeError(
            "The selected model is not an NLI classifier with an 'entailment' label. "
            f"Available labels: {available}"
        )


def parse_request(payload: Mapping[str, Any]) -> DecisionRequest:
    """Validate untrusted JSON and convert it into a typed request.

    The state is intentionally flexible so callers can pass nested JSON. Decision
    definitions are strict: this lets the program construct only declared choice
    IDs and explicitly bounded numeric scores.
    """
    if not isinstance(payload, Mapping):
        raise InputValidationError("The top-level JSON value must be an object.")
    state = payload.get("state")
    if not isinstance(state, Mapping) or not state:
        raise InputValidationError("'state' must be a non-empty JSON object.")
    raw_decisions = payload.get("decisions")
    if not isinstance(raw_decisions, list) or not raw_decisions:
        raise InputValidationError("'decisions' must be a non-empty JSON array.")

    decisions = tuple(_parse_decision(raw_decision) for raw_decision in raw_decisions)
    decision_ids = [decision.identifier for decision in decisions]
    if len(set(decision_ids)) != len(decision_ids):
        raise InputValidationError("Every decision must have a unique 'id'.")
    return DecisionRequest(state=state, decisions=decisions)


def evaluate(request: DecisionRequest, scorer: NliScorer) -> dict[str, Any]:
    """Batch-score a request and build safe, typed JSON-compatible results."""
    comparisons = _build_comparisons(request)
    evidence = scorer.score(comparisons)
    if len(evidence) != len(comparisons):
        raise RuntimeError(
            f"Scorer returned {len(evidence)} rows for {len(comparisons)} comparisons."
        )

    evidence_by_decision: dict[str, list[tuple[Comparison, NliEvidence]]] = {}
    for comparison, row_evidence in zip(comparisons, evidence, strict=True):
        evidence_by_decision.setdefault(comparison.decision_id, []).append(
            (comparison, row_evidence)
        )

    results = [
        _build_result(decision, evidence_by_decision[decision.identifier])
        for decision in request.decisions
    ]
    return {
        "model_contract": "zero-shot-nli-decision-poc-v1",
        "result_count": len(results),
        "results": results,
    }


def _parse_decision(raw_decision: Any) -> Decision:
    if not isinstance(raw_decision, Mapping):
        raise InputValidationError("Each decision must be a JSON object.")

    identifier = _required_identifier(raw_decision, "id")
    kind = raw_decision.get("kind")
    if kind not in {"choice", "boolean", "score"}:
        raise InputValidationError(
            f"Decision '{identifier}' has invalid kind {kind!r}; use choice, boolean, or score."
        )
    question = _required_text(raw_decision, "question", identifier)
    minimum_probability = _probability_setting(
        raw_decision, "minimum_probability", identifier, default=0.65
    )
    minimum_margin = _probability_setting(
        raw_decision, "minimum_margin", identifier, default=0.15
    )

    if kind == "boolean":
        true_description = _required_text(raw_decision, "true_description", identifier)
        return Decision(
            identifier=identifier,
            kind="boolean",
            question=question,
            candidates=(),
            true_description=true_description,
            minimum_probability=minimum_probability,
            minimum_margin=minimum_margin,
        )

    key = "options" if kind == "choice" else "levels"
    raw_candidates = raw_decision.get(key)
    if not isinstance(raw_candidates, list) or len(raw_candidates) < 2:
        raise InputValidationError(
            f"Decision '{identifier}' needs at least two entries in '{key}'."
        )
    candidates = tuple(
        _parse_candidate(raw_candidate, identifier, needs_numeric_value=kind == "score")
        for raw_candidate in raw_candidates
    )
    candidate_ids = [candidate.identifier for candidate in candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise InputValidationError(f"Decision '{identifier}' has duplicate candidate IDs.")

    if kind == "score":
        numeric_values = [candidate.numeric_value for candidate in candidates]
        if len(set(numeric_values)) != len(numeric_values):
            raise InputValidationError(f"Score decision '{identifier}' has duplicate level values.")
        return Decision(
            identifier=identifier,
            kind="score",
            question=question,
            candidates=candidates,
            minimum_probability=minimum_probability,
            minimum_margin=minimum_margin,
        )

    review_option_id = raw_decision.get("review_option_id")
    if review_option_id is not None:
        if not isinstance(review_option_id, str) or review_option_id not in candidate_ids:
            raise InputValidationError(
                f"Choice decision '{identifier}' has a review_option_id that is not an option ID."
            )
    return Decision(
        identifier=identifier,
        kind="choice",
        question=question,
        candidates=candidates,
        minimum_probability=minimum_probability,
        minimum_margin=minimum_margin,
        review_option_id=review_option_id,
    )


def _parse_candidate(
    raw_candidate: Any,
    decision_id: str,
    *,
    needs_numeric_value: bool,
) -> Candidate:
    if not isinstance(raw_candidate, Mapping):
        raise InputValidationError(f"Candidates in decision '{decision_id}' must be objects.")
    identifier = _required_identifier(raw_candidate, "id")
    description = _required_text(raw_candidate, "description", decision_id)
    numeric_value: float | None = None
    if needs_numeric_value:
        value = raw_candidate.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InputValidationError(
                f"Score level '{identifier}' in decision '{decision_id}' needs a numeric 'value'."
            )
        numeric_value = float(value)
    return Candidate(identifier=identifier, description=description, numeric_value=numeric_value)


def _required_identifier(raw_value: Mapping[str, Any], key: str) -> str:
    value = raw_value.get(key)
    if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value):
        raise InputValidationError(
            f"'{key}' must match {_ID_PATTERN.pattern!r}, for example 'owning_team'."
        )
    return value


def _required_text(raw_value: Mapping[str, Any], key: str, decision_id: str) -> str:
    value = raw_value.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InputValidationError(f"Decision '{decision_id}' needs a non-empty '{key}' string.")
    return value.strip()


def _probability_setting(
    raw_value: Mapping[str, Any],
    key: str,
    decision_id: str,
    *,
    default: float,
) -> float:
    value = raw_value.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise InputValidationError(
            f"Decision '{decision_id}' has invalid '{key}'; use a number from 0 to 1."
        )
    return float(value)


def _build_comparisons(request: DecisionRequest) -> list[Comparison]:
    state_text = json.dumps(request.state, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    comparisons: list[Comparison] = []
    for decision in request.decisions:
        premise = f"State evidence:\n{state_text}\n\nDecision question:\n{decision.question}"
        if decision.kind == "boolean":
            comparisons.append(
                Comparison(
                    decision_id=decision.identifier,
                    candidate_id="true",
                    premise=premise,
                    hypothesis=(
                        "The correct answer is true. "
                        f"The following condition applies: {decision.true_description}"
                    ),
                )
            )
            continue

        for candidate in decision.candidates:
            noun = "option" if decision.kind == "choice" else "score level"
            comparisons.append(
                Comparison(
                    decision_id=decision.identifier,
                    candidate_id=candidate.identifier,
                    premise=premise,
                    hypothesis=(
                        f"The correct {noun} is '{candidate.identifier}'. "
                        f"Definition: {candidate.description}"
                    ),
                    numeric_value=candidate.numeric_value,
                )
            )
    return comparisons


def _build_result(
    decision: Decision,
    rows: Sequence[tuple[Comparison, NliEvidence]],
) -> dict[str, Any]:
    if decision.kind == "boolean":
        return _boolean_result(decision, rows)
    probabilities = _softmax([evidence.entailment_logit for _, evidence in rows])
    candidate_probabilities = {
        comparison.candidate_id: probability
        for (comparison, _), probability in zip(rows, probabilities, strict=True)
    }
    if decision.kind == "choice":
        return _choice_result(decision, candidate_probabilities)
    return _score_result(decision, rows, candidate_probabilities)


def _boolean_result(
    decision: Decision,
    rows: Sequence[tuple[Comparison, NliEvidence]],
) -> dict[str, Any]:
    if len(rows) != 1:
        raise RuntimeError(f"Boolean decision '{decision.identifier}' must have one comparison.")
    probability_true = rows[0][1].entailment_probability
    answer = probability_true >= decision.minimum_probability
    return {
        "id": decision.identifier,
        "kind": "boolean",
        "answer": answer,
        "probability_true": _rounded(probability_true),
        "review_required": not answer,
        "review_reason": (
            None
            if answer
            else f"P(true) {_rounded(probability_true)} is below "
            f"minimum_probability {decision.minimum_probability}."
        ),
    }


def _choice_result(decision: Decision, probabilities: Mapping[str, float]) -> dict[str, Any]:
    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    recommended_option, top_probability = ranked[0]
    second_probability = ranked[1][1]
    margin = top_probability - second_probability
    review_reasons: list[str] = []
    if top_probability < decision.minimum_probability:
        review_reasons.append(
            f"top probability {_rounded(top_probability)} is below "
            f"minimum_probability {decision.minimum_probability}"
        )
    if margin < decision.minimum_margin:
        review_reasons.append(
            f"top-two margin {_rounded(margin)} is below "
            f"minimum_margin {decision.minimum_margin}"
        )
    if recommended_option == decision.review_option_id:
        review_reasons.append("the designated review option scored highest")

    review_required = bool(review_reasons)
    answer: str | None = recommended_option
    if review_required and decision.review_option_id is not None:
        answer = decision.review_option_id
    elif review_required:
        answer = None

    return {
        "id": decision.identifier,
        "kind": "choice",
        "answer": answer,
        "recommended_option": recommended_option,
        "probabilities": {key: _rounded(value) for key, value in probabilities.items()},
        "top_probability": _rounded(top_probability),
        "top_two_margin": _rounded(margin),
        "review_required": review_required,
        "review_reason": "; ".join(review_reasons) if review_reasons else None,
    }


def _score_result(
    decision: Decision,
    rows: Sequence[tuple[Comparison, NliEvidence]],
    probabilities: Mapping[str, float],
) -> dict[str, Any]:
    level_values = {candidate.identifier: candidate.numeric_value for candidate in decision.candidates}
    value = sum(probabilities[level_id] * level_values[level_id] for level_id in probabilities)
    dominant_level, dominant_probability = max(probabilities.items(), key=lambda item: item[1])
    return {
        "id": decision.identifier,
        "kind": "score",
        "value": _rounded(value),
        "dominant_level": dominant_level,
        "dominant_level_probability": _rounded(dominant_probability),
        "level_probabilities": {key: _rounded(score) for key, score in probabilities.items()},
        "review_required": False,
        "review_reason": None,
    }


def _softmax(logits: Sequence[float]) -> list[float]:
    maximum = max(logits)
    exponentials = [math.exp(logit - maximum) for logit in logits]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def _rounded(value: float) -> float:
    return round(value, 6)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Path to a decision request JSON file.")
    parser.add_argument("--output", type=Path, help="Optional output path. Defaults to stdout.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Hugging Face model (default: {DEFAULT_MODEL}).")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda, or another torch device.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Maximum comparison rows per model forward pass (default: 64).",
    )
    arguments = parser.parse_args()

    try:
        payload = json.loads(arguments.input.read_text(encoding="utf-8"))
        request = parse_request(payload)
        scorer = TransformersNliScorer(
            arguments.model,
            device=arguments.device,
            batch_size=arguments.batch_size,
        )
        result = evaluate(request, scorer)
    except (InputValidationError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if arguments.output:
        arguments.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
