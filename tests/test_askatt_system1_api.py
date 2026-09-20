from __future__ import annotations

import math

from fastapi.testclient import TestClient

from askatt_system1_api.app import create_app
from askatt_system1_api.engine import DeterministicTestEngine
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
    assert response.headers["x-askatt-forward-passes"] == "1"
    assert response.headers["x-askatt-compiled-labels"] == "4"
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


def test_score_is_explicitly_unsupported() -> None:
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
    assert response.status_code == 422
    assert "'noul' and 'choice'" in response.json()["detail"]


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
