"""HTTP surface: POST /v1/systemone and GET /v1/models."""

from __future__ import annotations

import argparse
import os
from typing import Any

from jev_any_llm.api import Client
from jev_any_llm.errors import JevAnyLlmError

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
except ImportError:  # pragma: no cover - exercised via optional dep
    FastAPI = None  # type: ignore
    Request = None  # type: ignore
    JSONResponse = None  # type: ignore


def create_app(client: Client | None = None) -> Any:
    if FastAPI is None:
        raise RuntimeError("jev_any_llm.serve requires fastapi: pip install 'jev-any-llm[serve]'")

    app = FastAPI(title="jev-any-llm", version="0.1.0")
    app.state.client = client or Client(
        model=os.environ.get("JEV_ANY_LLM_MODEL", "mock"),
        backend_kind=os.environ.get("JEV_ANY_LLM_BACKEND", "mock"),
        base_url=os.environ.get("JEV_ANY_LLM_OPENAI_BASE_URL"),
        api_key=os.environ.get("JEV_ANY_LLM_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY"),
    )

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
                }
            ],
        }

    @app.post("/v1/systemone")
    @app.post("/v1/decide")
    async def systemone(request: Request) -> JSONResponse:
        try:
            body = await request.json()
            response = app.state.client.decide_raw(body)
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
    args = parser.parse_args(argv)

    import uvicorn

    client = Client(
        model=args.model,
        backend_kind=args.backend,
        base_url=args.base_url,
        api_key=os.environ.get("JEV_ANY_LLM_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY"),
    )
    app = create_app(client)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
