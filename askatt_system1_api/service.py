"""JEV-compatible Choice and Noul request/response translation."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Iterable

from .engine import EngineResult, ScoringEngine

MAX_QUESTIONS = 64
MAX_CHOICE_OPTIONS = 255
MAX_STATE_CHARS = 100_000
MAX_INSTRUCTION_CHARS = 20_000
MAX_ID_CHARS = 200


class RequestValidationError(ValueError):
    """Raised when a request does not satisfy the supported System One shape."""


@dataclass(frozen=True)
class CompiledLabel:
    question_id: str
    question_type: str
    label: str
    option: str | None = None
    question_context: str | None = None


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _softmax(values: Iterable[float]) -> list[float]:
    logits = list(values)
    peak = max(logits)
    weights = [math.exp(value - peak) for value in logits]
    total = sum(weights)
    return [weight / total for weight in weights]


def _logit(probability: float) -> float:
    clipped = min(1.0 - 1e-6, max(1e-6, probability))
    return math.log(clipped / (1.0 - clipped))


def _distribution_confidence(probabilities: Iterable[float]) -> float:
    values = list(probabilities)
    if len(values) <= 1:
        return 1.0
    entropy = -sum(value * math.log(max(value, 1e-12)) for value in values)
    return min(1.0, max(0.0, 1.0 - entropy / math.log(len(values))))


def _complementary_binary_options(criteria: dict[str, Any]) -> tuple[str, str] | None:
    """Return (positive, negative) ids for conventional binary Choice options."""
    normalized = {str(option).strip().lower(): str(option) for option in criteria}
    for positive, negative in (
        ("present", "absent"),
        ("yes", "no"),
        ("true", "false"),
    ):
        if set(normalized) == {positive, negative}:
            return normalized[positive], normalized[negative]
    return None


class AskATTSystem1Service:
    """Compile Choice/Noul questions into one or more GLiClass shared passes."""

    def __init__(
        self,
        engine: ScoringEngine,
        *,
        labels_per_pass: int = 64,
        choice_instruction_mode: str = "state",
    ) -> None:
        if labels_per_pass < 1:
            raise ValueError("labels_per_pass must be positive")
        if choice_instruction_mode not in {"state", "discard"}:
            raise ValueError("choice_instruction_mode must be 'state' or 'discard'")
        self.engine = engine
        self.labels_per_pass = labels_per_pass
        self.choice_instruction_mode = choice_instruction_mode

    def _validate_and_compile(
        self, payload: dict[str, Any]
    ) -> tuple[str, list[CompiledLabel], dict[str, dict[str, Any]]]:
        if not isinstance(payload, dict):
            raise RequestValidationError("Request body must be a JSON object")
        model = payload.get("model")
        if not isinstance(model, str) or not model.strip():
            raise RequestValidationError("model must be a non-empty string")
        if "state" not in payload:
            raise RequestValidationError("Missing required field: state")
        if not isinstance(payload["state"], (str, dict, list)):
            raise RequestValidationError("state must be a string, object, or array")
        state = _as_text(payload["state"])
        if len(state) > MAX_STATE_CHARS:
            raise RequestValidationError(
                f"state exceeds the {MAX_STATE_CHARS}-character compatibility limit"
            )
        questions = payload.get("questions")
        if not isinstance(questions, dict) or not questions:
            raise RequestValidationError("questions must be a non-empty object")
        if len(questions) > MAX_QUESTIONS:
            raise RequestValidationError(
                f"questions contains {len(questions)} entries; maximum is {MAX_QUESTIONS}"
            )

        compiled: list[CompiledLabel] = []
        validated: dict[str, dict[str, Any]] = {}
        for question_id, raw_question in questions.items():
            if not isinstance(question_id, str) or not question_id:
                raise RequestValidationError("Every question id must be a non-empty string")
            if len(question_id) > MAX_ID_CHARS:
                raise RequestValidationError(
                    f"question id {question_id!r} exceeds {MAX_ID_CHARS} characters"
                )
            if not isinstance(raw_question, dict):
                raise RequestValidationError(f"question {question_id!r} must be an object")
            kind = raw_question.get("type")
            if kind not in {"noul", "choice"}:
                raise RequestValidationError(
                    f"question {question_id!r}: supported types are 'noul' and 'choice'"
                )
            if "instructions" not in raw_question:
                raise RequestValidationError(
                    f"question {question_id!r}: missing instructions"
                )
            if not isinstance(raw_question["instructions"], (str, dict, list)):
                raise RequestValidationError(
                    f"question {question_id!r}: instructions must be a string, object, or array"
                )
            instructions = _as_text(raw_question["instructions"])
            if len(instructions) > MAX_INSTRUCTION_CHARS:
                raise RequestValidationError(
                    f"question {question_id!r}: instructions exceed {MAX_INSTRUCTION_CHARS} characters"
                )

            if kind == "noul":
                criteria = raw_question.get("criteria") or {}
                if not isinstance(criteria, dict):
                    raise RequestValidationError(
                        f"question {question_id!r}: Noul criteria must be an object"
                    )
                true_rule = criteria.get("true")
                label = instructions if true_rule is None else _as_text(true_rule)
                compiled.append(CompiledLabel(question_id, kind, label))
            else:
                criteria = raw_question.get("criteria")
                if not isinstance(criteria, dict) or len(criteria) < 2:
                    raise RequestValidationError(
                        f"question {question_id!r}: Choice criteria must contain at least two options"
                    )
                if len(criteria) > MAX_CHOICE_OPTIONS:
                    raise RequestValidationError(
                        f"question {question_id!r}: Choice accepts at most {MAX_CHOICE_OPTIONS} options"
                    )
                binary = _complementary_binary_options(criteria)
                items = (
                    [(binary[0], criteria[binary[0]])]
                    if binary is not None
                    else list(criteria.items())
                )
                for option, description in items:
                    if not isinstance(option, str) or not option:
                        raise RequestValidationError(
                            f"question {question_id!r}: option ids must be non-empty strings"
                        )
                    label = (
                        f"{instructions}: {option}"
                        if description is None
                        else _as_text(description)
                    )
                    question_context = (
                        instructions
                        if binary is None and self.choice_instruction_mode == "state"
                        else None
                    )
                    compiled.append(
                        CompiledLabel(
                            question_id,
                            kind,
                            label,
                            option,
                            question_context,
                        )
                    )
            validated[question_id] = raw_question
        return state, compiled, validated

    def _score_labels(
        self, state: str, compiled: list[CompiledLabel]
    ) -> tuple[list[float], dict[str, Any]]:
        scores = [0.0] * len(compiled)
        input_tokens = 0
        elapsed_seconds = 0.0
        truncation = False
        passes = 0
        groups: dict[tuple[str, str] | None, list[tuple[int, CompiledLabel]]] = {}
        for index, item in enumerate(compiled):
            key = (
                (item.question_id, item.question_context)
                if item.question_context is not None
                else None
            )
            groups.setdefault(key, []).append((index, item))
        for key, group in groups.items():
            scoped_state = (
                state
                if key is None
                else f"{state}\n\nClassification question: {key[1]}"
            )
            for offset in range(0, len(group), self.labels_per_pass):
                batch = group[offset : offset + self.labels_per_pass]
                result: EngineResult = self.engine.score(
                    scoped_state, [item.label for _, item in batch]
                )
                if len(result.scores) != len(batch):
                    raise RuntimeError("Scoring engine returned the wrong number of scores")
                for (index, _), score in zip(batch, result.scores, strict=True):
                    scores[index] = score
                input_tokens += result.input_tokens
                elapsed_seconds += result.elapsed_seconds
                truncation = truncation or result.truncated
                passes += 1
        return scores, {
            "input_tokens": input_tokens,
            "output_tokens": 0,
            "forward_passes": passes,
            "inference_seconds": elapsed_seconds,
            "truncated": truncation,
            "compiled_labels": len(compiled),
            "choice_instruction_mode": self.choice_instruction_mode,
        }

    def evaluate(self, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        state, compiled, questions = self._validate_and_compile(payload)
        scores, metadata = self._score_labels(state, compiled)
        indexed = list(zip(compiled, scores, strict=True))
        answers: dict[str, dict[str, Any]] = {}
        for question_id, question in questions.items():
            kind = question["type"]
            relevant = [(item, score) for item, score in indexed if item.question_id == question_id]
            if kind == "noul":
                answers[question_id] = {
                    "type": "noul",
                    "noul": round(float(relevant[0][1]), 8),
                }
                continue
            criteria = question["criteria"]
            binary = _complementary_binary_options(criteria)
            if binary is not None:
                positive, negative = binary
                positive_probability = min(1.0, max(0.0, float(relevant[0][1])))
                by_option = {
                    positive: positive_probability,
                    negative: 1.0 - positive_probability,
                }
                option_names = list(criteria)
                probabilities = [by_option[str(option)] for option in option_names]
            else:
                option_names = [item.option for item, _ in relevant]
                probabilities = _softmax(_logit(score) for _, score in relevant)
            distribution = {
                str(option): round(probability, 8)
                for option, probability in zip(option_names, probabilities, strict=True)
            }
            # Re-normalize after rounding so the public contract still sums to one.
            correction = round(1.0 - sum(distribution.values()), 8)
            winner_index = max(range(len(probabilities)), key=probabilities.__getitem__)
            winner = str(option_names[winner_index])
            distribution[winner] = round(distribution[winner] + correction, 8)
            answers[question_id] = {
                "type": "choice",
                "choice": winner,
                "probabilities": distribution,
                "confidence": round(_distribution_confidence(probabilities), 8),
            }
        response = {
            "model": self.engine.model_name,
            "answers": answers,
            "usage": {
                "input_tokens": metadata["input_tokens"],
                "output_tokens": metadata["output_tokens"],
            },
        }
        return response, metadata
