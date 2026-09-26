"""JEV-compatible Choice, Noul, and Score request/response translation."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Iterable

from .engine import EngineResult, ScoringEngine

MAX_QUESTIONS = 64
MAX_CHOICE_OPTIONS = 255
MAX_SCORE_LEVELS = 10
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
    routing_profile: str = "base"


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


def _question_routing_profile(
    question_id: str, question: dict[str, Any], instructions: str
) -> str:
    """Choose the base model or specialized tool-routing adapter."""
    requested = question.get("routing_profile", "auto")
    if requested not in {"auto", "base", "tool_routing"}:
        raise RequestValidationError(
            f"question {question_id!r}: routing_profile must be auto, base, or tool_routing"
        )
    if requested != "auto":
        return str(requested)
    question_name = question_id.lower().replace("-", "_")
    instruction_text = instructions.lower()
    criteria_text = _as_text(question.get("criteria") or {}).lower()
    tool_marker = (
        "tool" in question_name
        or "function" in question_name
        or "tool" in instruction_text
        or "callable" in instruction_text
        or "no_tool" in criteria_text
        or "use tool" in criteria_text
        or any(
            phrase in instruction_text
            for phrase in (
                "which available tool",
                "which offered tool",
                "which tool",
                "should any offered tool",
                "should a tool",
                "invoke a tool",
                "call a function",
                "which function",
            )
        )
    )
    return "tool_routing" if tool_marker else "base"


class AskATTSystem1Service:
    """Compile Choice, Noul, and Score questions into GLiClass passes."""

    def __init__(
        self,
        engine: ScoringEngine,
        *,
        tool_engine: ScoringEngine | None = None,
        labels_per_pass: int = 64,
        choice_instruction_mode: str = "state",
        noul_boundary_mode: str = "positive",
    ) -> None:
        if labels_per_pass < 1:
            raise ValueError("labels_per_pass must be positive")
        if choice_instruction_mode not in {"state", "discard"}:
            raise ValueError("choice_instruction_mode must be 'state' or 'discard'")
        if noul_boundary_mode not in {"positive", "paired"}:
            raise ValueError("noul_boundary_mode must be 'positive' or 'paired'")
        self.engine = engine
        self.tool_engine = tool_engine
        self.labels_per_pass = labels_per_pass
        self.choice_instruction_mode = choice_instruction_mode
        self.noul_boundary_mode = noul_boundary_mode

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
            if kind not in {"noul", "choice", "score"}:
                raise RequestValidationError(
                    f"question {question_id!r}: supported types are 'noul', 'choice', and 'score'"
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
            routing_profile = _question_routing_profile(
                question_id, raw_question, instructions
            )

            if kind == "noul":
                criteria = raw_question.get("criteria") or {}
                if not isinstance(criteria, dict):
                    raise RequestValidationError(
                        f"question {question_id!r}: Noul criteria must be an object"
                    )
                true_rule = criteria.get("true")
                false_rule = criteria.get("false")
                if (
                    self.noul_boundary_mode == "paired"
                    and true_rule is not None
                    and false_rule is not None
                ):
                    compiled.extend(
                        [
                            CompiledLabel(
                                question_id,
                                kind,
                                _as_text(true_rule),
                                option="true",
                                routing_profile=routing_profile,
                            ),
                            CompiledLabel(
                                question_id,
                                kind,
                                _as_text(false_rule),
                                option="false",
                                routing_profile=routing_profile,
                            ),
                        ]
                    )
                else:
                    label = instructions if true_rule is None else _as_text(true_rule)
                    compiled.append(
                        CompiledLabel(
                            question_id,
                            kind,
                            label,
                            option="true",
                            routing_profile=routing_profile,
                        )
                    )
            elif kind == "choice":
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
                            routing_profile,
                        )
                    )
            else:
                criteria = raw_question.get("criteria")
                if not isinstance(criteria, list) or not 2 <= len(criteria) <= MAX_SCORE_LEVELS:
                    raise RequestValidationError(
                        f"question {question_id!r}: Score criteria must contain 2 to {MAX_SCORE_LEVELS} ordered levels"
                    )
                for index, description in enumerate(criteria):
                    if not isinstance(description, (str, dict, list)):
                        raise RequestValidationError(
                            f"question {question_id!r}: Score levels must be text, objects, or arrays"
                        )
                    compiled.append(
                        CompiledLabel(
                            question_id,
                            kind,
                            _as_text(description),
                            str(index),
                            instructions if self.choice_instruction_mode == "state" else None,
                            routing_profile,
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
        groups: dict[
            tuple[str, str | None, str | None], list[tuple[int, CompiledLabel]]
        ] = {}
        for index, item in enumerate(compiled):
            key = (
                item.routing_profile,
                item.question_id if item.question_context is not None else None,
                item.question_context,
            )
            groups.setdefault(key, []).append((index, item))
        passes_by_profile: dict[str, int] = {"base": 0, "tool_routing": 0}
        labels_by_profile: dict[str, int] = {"base": 0, "tool_routing": 0}
        for key, group in groups.items():
            routing_profile, _question_id, question_context = key
            scoped_state = (
                state
                if question_context is None
                else f"{state}\n\nClassification question: {question_context}"
            )
            selected_engine = (
                self.tool_engine
                if routing_profile == "tool_routing" and self.tool_engine is not None
                else self.engine
            )
            for offset in range(0, len(group), self.labels_per_pass):
                batch = group[offset : offset + self.labels_per_pass]
                result: EngineResult = selected_engine.score(
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
                passes_by_profile[routing_profile] += 1
                labels_by_profile[routing_profile] += len(batch)
        question_profiles = {
            item.question_id: item.routing_profile for item in compiled
        }
        return scores, {
            "input_tokens": input_tokens,
            "output_tokens": 0,
            "forward_passes": passes,
            "inference_seconds": elapsed_seconds,
            "truncated": truncation,
            "compiled_labels": len(compiled),
            "choice_instruction_mode": self.choice_instruction_mode,
            "noul_boundary_mode": self.noul_boundary_mode,
            "question_routing_profiles": question_profiles,
            "passes_by_profile": passes_by_profile,
            "labels_by_profile": labels_by_profile,
            "engines": {
                "base": self.engine.model_name,
                "tool_routing": (
                    self.tool_engine.model_name
                    if self.tool_engine is not None
                    else self.engine.model_name
                ),
            },
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
                by_option = {item.option: score for item, score in relevant}
                if "false" in by_option:
                    positive_probability = _softmax(
                        (_logit(by_option["true"]), _logit(by_option["false"]))
                    )[0]
                else:
                    positive_probability = float(by_option["true"])
                answers[question_id] = {
                    "type": "noul",
                    "noul": round(positive_probability, 8),
                }
                continue
            if kind == "score":
                option_names = [str(item.option) for item, _ in relevant]
                probabilities = _softmax(_logit(score) for _, score in relevant)
                distribution = {
                    option: round(probability, 8)
                    for option, probability in zip(option_names, probabilities, strict=True)
                }
                winner_index = max(range(len(probabilities)), key=probabilities.__getitem__)
                correction = round(1.0 - sum(distribution.values()), 8)
                distribution[str(winner_index)] = round(
                    distribution[str(winner_index)] + correction, 8
                )
                expected_score = sum(
                    index * probability for index, probability in enumerate(probabilities)
                )
                answers[question_id] = {
                    "type": "score",
                    "score": round(expected_score, 8),
                    "legend": question["criteria"],
                    "probabilities": distribution,
                    "confidence": round(_distribution_confidence(probabilities), 8),
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
            "model": (
                f"askatt-hybrid[{self.engine.model_name}|{self.tool_engine.model_name}]"
                if self.tool_engine is not None
                else self.engine.model_name
            ),
            "answers": answers,
            "usage": {
                "input_tokens": metadata["input_tokens"],
                "output_tokens": metadata["output_tokens"],
            },
            "askatt_routing": {
                "question_profiles": metadata["question_routing_profiles"],
                "engines": metadata["engines"],
            },
        }
        return response, metadata
