"""HTTP surface: POST /v1/systemone, GET /v1/models, GET / playground."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from jev_any_llm.api import Client
from jev_any_llm.errors import JevAnyLlmError

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse
except ImportError:  # pragma: no cover - exercised via optional dep
    FastAPI = None  # type: ignore
    Request = None  # type: ignore
    HTMLResponse = None  # type: ignore
    JSONResponse = None  # type: ignore

_PLAYGROUND = Path(__file__).with_name("playground.html")


def create_app(client: Client | None = None) -> Any:
    if FastAPI is None:
        raise RuntimeError("jev_any_llm.serve requires fastapi: pip install 'jev-any-llm[serve]'")

    app = FastAPI(title="jev-any-llm", version="0.1.0")
    mode = os.environ.get("JEV_ANY_LLM_MODE", "isolated")
    profile = os.environ.get("JEV_ANY_LLM_TEMPERATURE_PROFILE")
    app.state.client = client or Client(
        model=os.environ.get("JEV_ANY_LLM_MODEL", "mock"),
        backend_kind=os.environ.get("JEV_ANY_LLM_BACKEND", "mock"),
        base_url=os.environ.get("JEV_ANY_LLM_OPENAI_BASE_URL"),
        api_key=os.environ.get("JEV_ANY_LLM_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY"),
        mode=mode,
        temperature_profile=profile,
    )

    @app.get("/", response_class=HTMLResponse)
    def playground() -> HTMLResponse:
        return HTMLResponse(_PLAYGROUND.read_text(encoding="utf-8"))

    @app.get("/v1/models")
    def list_models() -> dict:
        model_id = app.state.client.model
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "owned_by": "jev-any-llm",
                    "backend": getattr(app.state.client.backend, "name", "wrap"),
                    "mode": getattr(app.state.client, "mode", "isolated"),
                }
            ],
        }

    @app.post("/v1/systemone")
    @app.post("/v1/decide")
    async def systemone(request: Request) -> JSONResponse:
        try:
            body = await request.json()
            mode_override = body.pop("mode", None) if isinstance(body, dict) else None
            response = app.state.client.decide_raw(body, mode=mode_override)
            return JSONResponse(response.to_dict())
        except JevAnyLlmError as exc:
            status = 400 if exc.code in ("validation_error", "schema_error") else 502
            return JSONResponse(exc.to_dict(), status_code=status)

    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve jev-any-llm decide() over HTTP")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--model", default=os.environ.get("JEV_ANY_LLM_MODEL", "mock"))
    parser.add_argument("--backend", default=os.environ.get("JEV_ANY_LLM_BACKEND", "mock"))
    parser.add_argument("--base-url", default=os.environ.get("JEV_ANY_LLM_OPENAI_BASE_URL"))
    parser.add_argument(
        "--mode",
        default=os.environ.get("JEV_ANY_LLM_MODE", "isolated"),
        choices=("isolated", "branched"),
    )
    parser.add_argument(
        "--temperature-profile",
        default=os.environ.get("JEV_ANY_LLM_TEMPERATURE_PROFILE"),
        help="Path to TemperatureProfile JSON (optional)",
    )
    args = parser.parse_args(argv)

    import uvicorn

    client = Client(
        model=args.model,
        backend_kind=args.backend,
        base_url=args.base_url,
        api_key=os.environ.get("JEV_ANY_LLM_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY"),
        mode=args.mode,
        temperature_profile=args.temperature_profile,
    )
    app = create_app(client)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
