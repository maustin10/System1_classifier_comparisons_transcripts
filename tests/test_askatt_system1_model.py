from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from askatt_system1_api.app import create_app
from askatt_system1_api.engine import GLiClassEngine
from askatt_system1_api.service import AskATTSystem1Service


pytestmark = pytest.mark.skipif(
    os.getenv("ASKATT_RUN_MODEL_TESTS") != "1",
    reason="set ASKATT_RUN_MODEL_TESTS=1 to load the local GLiClass checkpoint",
)


def test_real_model_mixed_request() -> None:
    root = Path(__file__).resolve().parents[1]
    model_dir = root / "models" / "gliclass-modern-large-v3.0"
    service = AskATTSystem1Service(GLiClassEngine(model_dir, device="cpu"))
    response = TestClient(create_app(service)).post(
        "/v1/systemone",
        json={
            "model": "jev-latest",
            "state": "The caller says an unfamiliar fee appeared on the bill.",
            "questions": {
                "unknown_charge": {
                    "type": "noul",
                    "instructions": "Is an unknown charge present?",
                    "criteria": {"true": "Unknown charges on the bill"},
                },
                "department": {
                    "type": "choice",
                    "instructions": "Route the issue",
                    "criteria": {
                        "billing": "Billing, charges, and payments",
                        "technical": "Internet or device technical failure",
                        "sales": "New purchases or upgrades",
                    },
                },
            },
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answers"]["unknown_charge"]["noul"] > 0.5
    assert body["answers"]["department"]["choice"] == "billing"
    assert response.headers["x-askatt-forward-passes"] == "2"
    assert response.headers["x-askatt-truncated"] == "false"
