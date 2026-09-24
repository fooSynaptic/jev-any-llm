#!/usr/bin/env python3
"""Tokenizer smoke for the three bench models (strict single-token aliases).

Default model ids match the jev_mode_benchmark scorecards. Override with
``JEV_SMOKE_MODELS`` (comma-separated paths or hub ids) or positional args.

Does not load weights — tokenizer only. Exit 0 when all closed sets resolve.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


DEFAULT_MODELS = [
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.8-27B",
    "deepseek-ai/DeepSeek-V4.1-Flash",
]


def _resolve_models(argv_models: list[str]) -> list[str]:
    if argv_models:
        return argv_models
    env = os.environ.get("JEV_SMOKE_MODELS", "").strip()
    if env:
        return [part.strip() for part in env.split(",") if part.strip()]
    root = os.environ.get("JEV_OPENMODELS_ROOT", "").strip()
    if root:
        local_names = ["Qwen3.5-4B", "Qwen3.8-27B", "DeepSeek-V4.1-Flash"]
        found = []
        for name in local_names:
            path = Path(root) / name
            if path.exists():
                found.append(str(path))
        if found:
            return found
    return list(DEFAULT_MODELS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "models",
        nargs="*",
        help="Tokenizer paths / hub ids (default: three bench models)",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON report")
    args = parser.parse_args(argv)

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print(
            "transformers required: pip install 'jev-any-llm[hf]'",
            file=sys.stderr,
        )
        return 2

    from jev_any_llm.tokens import validate_closed_set_tokenizer

    models = _resolve_models(args.models)
    reports = []
    failed = False
    for model_id in models:
        print(f"== {model_id}", flush=True)
        try:
            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        except Exception as exc:  # noqa: BLE001 — smoke must continue siblings
            failed = True
            entry = {"model": model_id, "ok": False, "error": str(exc)}
            reports.append(entry)
            print(f"  FAIL load: {exc}", flush=True)
            continue
        report = validate_closed_set_tokenizer(tok)
        report["model"] = model_id
        reports.append(report)
        if report["ok"]:
            print("  OK single-token Yes/No, A.., 0..", flush=True)
        else:
            failed = True
            print(f"  FAIL: {json.dumps(report['sets'], ensure_ascii=False)}", flush=True)

    if args.json:
        print(json.dumps({"ok": not failed, "models": reports}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
