"""Run the local API with ``python -m askatt_system1_api``."""
from __future__ import annotations

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "askatt_system1_api.app:app",
        host=os.environ.get("ASKATT_HOST", "127.0.0.1"),
        port=int(os.environ.get("ASKATT_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()

