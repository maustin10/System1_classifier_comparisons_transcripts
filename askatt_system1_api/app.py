"""FastAPI application exposing the TypeSafe-compatible endpoint path."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse

from .engine import GLiClassEngine
from .service import AskATTSystem1Service, RequestValidationError


def _default_service() -> AskATTSystem1Service:
    model_dir = Path(
        os.environ.get(
            "ASKATT_GLICLASS_MODEL",
            Path(__file__).resolve().parents[1] / "models" / "gliclass-modern-large-v3.0",
        )
    )
    labels_per_pass = int(os.environ.get("ASKATT_LABELS_PER_PASS", "64"))
    engine = GLiClassEngine(
        model_dir,
        device=os.environ.get("ASKATT_DEVICE", "auto"),
        max_labels=labels_per_pass,
    )
    return AskATTSystem1Service(engine, labels_per_pass=labels_per_pass)


def create_app(service: AskATTSystem1Service | None = None) -> FastAPI:
    app = FastAPI(
        title="AskATT System1 API",
        version="0.1.0",
        description="JEV-shaped Choice and Noul evaluation backed by GLiClass.",
    )
    active_service = service or _default_service()

    def authorize(authorization: str | None = Header(default=None)) -> None:
        expected = os.environ.get("ASKATT_SYSTEM1_API_KEY", "").strip()
        if not expected:
            return
        if authorization != f"Bearer {expected}":
            raise HTTPException(status_code=401, detail="Unauthorized")

    @app.get("/healthz")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": active_service.engine.model_name}

    @app.post("/v1/systemone", dependencies=[Depends(authorize)])
    def systemone(payload: dict[str, Any]) -> JSONResponse:
        try:
            response, metadata = active_service.evaluate(payload)
        except RequestValidationError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        headers = {
            "X-AskATT-Inference-Ms": f"{metadata['inference_seconds'] * 1000:.3f}",
            "X-AskATT-Forward-Passes": str(metadata["forward_passes"]),
            "X-AskATT-Compiled-Labels": str(metadata["compiled_labels"]),
            "X-AskATT-Truncated": str(metadata["truncated"]).lower(),
        }
        return JSONResponse(response, headers=headers)

    return app


app = create_app()

