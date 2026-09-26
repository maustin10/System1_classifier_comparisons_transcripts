from __future__ import annotations

import math

from fastapi.testclient import TestClient

from askatt_system1_api.app import create_app
from askatt_system1_api.engine import DeterministicTestEngine, EngineResult
from askatt_system1_api.service import AskATTSystem1Service


def client() -> TestClient:
    service = AskATTSystem1Service(DeterministicTestEngine(), labels_per_pass=64)
    return TestClient(create_app(service))


def test_mixed_choice_and_noul_contract() -> None:
    response = client().post(
        "/v1/systemone",
        json={
            "model": "jev-latest",
            "state": {"message": "This is urgent and concerns billing."},
            "questions": {
                "urgent": {
                    "type": "noul",
                    "instructions": "Is this urgent?",
                },
                "department": {
                    "type": "choice",
                    "instructions": "Which department?",
                    "criteria": {
                        "billing": "Billing and payments",
                        "technical": "Technical failures",
                        "sales": "New purchases",
                    },
                },
            },
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "askatt-deterministic-test-engine"
    assert body["answers"]["urgent"]["type"] == "noul"
    assert 0 <= body["answers"]["urgent"]["noul"] <= 1
    choice = body["answers"]["department"]
    assert choice["type"] == "choice"
    assert choice["choice"] == "billing"
    assert math.isclose(sum(choice["probabilities"].values()), 1.0, abs_tol=1e-8)
    assert 0 <= choice["confidence"] <= 1
    assert response.headers["x-askatt-forward-passes"] == "2"
    assert response.headers["x-askatt-compiled-labels"] == "4"
    assert response.headers["x-askatt-choice-instruction-mode"] == "state"
    assert body["usage"]["output_tokens"] == 0


def test_choice_requires_two_options() -> None:
    response = client().post(
        "/v1/systemone",
        json={
            "model": "jev-latest",
            "state": "hello",
            "questions": {
                "bad": {
                    "type": "choice",
                    "instructions": "Pick one",
                    "criteria": {"only": "Only option"},
                }
            },
        },
    )
    assert response.status_code == 422
    assert "at least two options" in response.json()["detail"]


def test_score_returns_ordered_distribution_and_expected_value() -> None:
    response = client().post(
        "/v1/systemone",
        json={
            "model": "jev-latest",
            "state": "hello",
            "questions": {
                "rating": {
                    "type": "score",
                    "instructions": "Rate it",
                    "criteria": ["low", "high"],
                }
            },
        },
    )
    assert response.status_code == 200
    score = response.json()["answers"]["rating"]
    assert score["type"] == "score"
    assert score["legend"] == ["low", "high"]
    assert set(score["probabilities"]) == {"0", "1"}
    assert math.isclose(sum(score["probabilities"].values()), 1.0, abs_tol=1e-8)
    assert math.isclose(score["score"], score["probabilities"]["1"], abs_tol=1e-8)
    assert 0 <= score["confidence"] <= 1


def test_label_batching_is_visible() -> None:
    service = AskATTSystem1Service(DeterministicTestEngine(), labels_per_pass=2)
    response, metadata = service.evaluate(
        {
            "model": "jev-latest",
            "state": "hello",
            "questions": {
                "a": {"type": "noul", "instructions": "A?"},
                "b": {"type": "noul", "instructions": "B?"},
                "c": {"type": "noul", "instructions": "C?"},
            },
        }
    )
    assert len(response["answers"]) == 3
    assert metadata["forward_passes"] == 2
    assert metadata["compiled_labels"] == 3


def test_general_choice_instruction_is_scored_with_the_state() -> None:
    class RecordingEngine:
        model_name = "recording-engine"

        def __init__(self) -> None:
            self.states: list[str] = []

        def score(self, state: str, labels: list[str]) -> EngineResult:
            self.states.append(state)
            return EngineResult([0.9] + [0.1] * (len(labels) - 1), 20, 0.001, False)

    engine = RecordingEngine()
    service = AskATTSystem1Service(engine)
    service.evaluate(
        {
            "model": "jev-latest",
            "state": "The caller has a billing problem.",
            "questions": {
                "department": {
                    "type": "choice",
                    "instructions": "Which department should own the request?",
                    "criteria": {
                        "billing": "Billing and payments",
                        "technical": "Technical failures",
                    },
                }
            },
        }
    )
    assert engine.states == [
        "The caller has a billing problem.\n\n"
        "Classification question: Which department should own the request?"
    ]


def test_legacy_discard_mode_is_explicit() -> None:
    service = AskATTSystem1Service(
        DeterministicTestEngine(), choice_instruction_mode="discard"
    )
    _, metadata = service.evaluate(
        {
            "model": "jev-latest",
            "state": "hello",
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Which route?",
                    "criteria": {"a": "Route A", "b": "Route B"},
                }
            },
        }
    )
    assert metadata["choice_instruction_mode"] == "discard"
    assert metadata["forward_passes"] == 1


def test_binary_choice_is_the_same_probability_as_noul() -> None:
    response = client().post(
        "/v1/systemone",
        json={
            "model": "jev-latest",
            "state": "The caller disputes a charge on the bill.",
            "questions": {
                "noul": {
                    "type": "noul",
                    "instructions": "Unknown charges",
                    "criteria": {"true": "Unknown charges on the bill"},
                },
                "choice": {
                    "type": "choice",
                    "instructions": "Unknown charges",
                    "criteria": {
                        "present": "Unknown charges on the bill",
                        "absent": "No unknown charges on the bill",
                    },
                },
            },
        },
    )
    assert response.status_code == 200
    answers = response.json()["answers"]
    assert answers["noul"]["noul"] == answers["choice"]["probabilities"]["present"]


def test_noul_uses_both_true_and_false_boundaries_when_supplied() -> None:
    class BoundaryEngine:
        model_name = "boundary-engine"

        def __init__(self) -> None:
            self.labels: list[str] = []

        def score(self, state: str, labels: list[str]) -> EngineResult:
            del state
            self.labels = list(labels)
            return EngineResult([0.8, 0.2], 20, 0.001, False)

    engine = BoundaryEngine()
    service = AskATTSystem1Service(engine, noul_boundary_mode="paired")
    response, metadata = service.evaluate(
        {
            "model": "jev-latest",
            "state": "The caller cannot work.",
            "questions": {
                "blocked": {
                    "type": "noul",
                    "instructions": "Is work blocked?",
                    "criteria": {
                        "true": "The caller explicitly cannot work",
                        "false": "The caller can continue working",
                    },
                }
            },
        }
    )
    assert engine.labels == [
        "The caller explicitly cannot work",
        "The caller can continue working",
    ]
    assert metadata["compiled_labels"] == 2
    assert response["answers"]["blocked"]["noul"] > 0.9


def test_model_and_state_are_required_and_typed() -> None:
    bad_model = client().post(
        "/v1/systemone",
        json={"state": "hello", "questions": {"a": {"type": "noul", "instructions": "A"}}},
    )
    assert bad_model.status_code == 422
    assert "model" in bad_model.json()["detail"]

    bad_state = client().post(
        "/v1/systemone",
        json={
            "model": "jev-latest",
            "state": 123,
            "questions": {"a": {"type": "noul", "instructions": "A"}},
        },
    )
    assert bad_state.status_code == 422
    assert "state" in bad_state.json()["detail"]


def test_hybrid_routes_only_tool_questions_to_specialist() -> None:
    class RecordingEngine:
        def __init__(self, name: str, first_score: float) -> None:
            self.model_name = name
            self.first_score = first_score
            self.calls: list[tuple[str, list[str]]] = []

        def score(self, state: str, labels: list[str]) -> EngineResult:
            self.calls.append((state, list(labels)))
            scores = [self.first_score] + [1.0 - self.first_score] * (len(labels) - 1)
            return EngineResult(scores, 10, 0.001, False)

    base = RecordingEngine("base", 0.9)
    tool = RecordingEngine("tool-lora", 0.1)
    service = AskATTSystem1Service(base, tool_engine=tool)
    response, metadata = service.evaluate(
        {
            "model": "jev-latest",
            "state": "Please route this request.",
            "questions": {
                "department": {
                    "type": "choice",
                    "instructions": "Which department should own this?",
                    "criteria": {"billing": "Billing", "technical": "Technical"},
                },
                "selected_tool": {
                    "type": "choice",
                    "instructions": "Which available tool best handles this request?",
                    "criteria": {"tool_a": "Tool A", "tool_b": "Tool B"},
                },
            },
        }
    )
    assert len(base.calls) == 1
    assert len(tool.calls) == 1
    assert metadata["question_routing_profiles"] == {
        "department": "base",
        "selected_tool": "tool_routing",
    }
    assert metadata["passes_by_profile"] == {"base": 1, "tool_routing": 1}
    assert response["askatt_routing"]["engines"] == {
        "base": "base",
        "tool_routing": "tool-lora",
    }


def test_explicit_routing_profile_overrides_auto_detection() -> None:
    service = AskATTSystem1Service(DeterministicTestEngine())
    _, metadata = service.evaluate(
        {
            "model": "jev-latest",
            "state": "hello",
            "questions": {
                "selected_tool": {
                    "type": "choice",
                    "instructions": "Which tool?",
                    "routing_profile": "base",
                    "criteria": {"a": "A", "b": "B"},
                }
            },
        }
    )
    assert metadata["question_routing_profiles"] == {"selected_tool": "base"}


def test_no_tool_question_auto_routes_to_tool_specialist() -> None:
    service = AskATTSystem1Service(
        DeterministicTestEngine(), tool_engine=DeterministicTestEngine()
    )
    _, metadata = service.evaluate(
        {
            "model": "jev-latest",
            "state": "Explain photosynthesis.",
            "questions": {
                "routing_irrelevance": {
                    "type": "choice",
                    "instructions": (
                        "Should the available tool be called now, or is no supplied "
                        "tool callable for this request?"
                    ),
                    "criteria": {
                        "tool_1": "Use tool weather.get for forecasts",
                        "no_tool": "No supplied tool should be called",
                    },
                }
            },
        }
    )
    assert metadata["question_routing_profiles"] == {
        "routing_irrelevance": "tool_routing"
    }
