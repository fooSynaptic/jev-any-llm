#!/usr/bin/env python3
"""Rename kev → jev-any-llm / jev_any_llm across the tree."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".cache", ".venv", ".pytest_cache", "__pycache__", "_local", "kev.egg-info", "scripts"}
EXTS = {".py", ".md", ".toml", ".svg", ".txt", ".yml", ".yaml"}

SUBS = [
    ("JevAnyLlmValidationError", "JevAnyLlmValidationError"),
    ("JevAnyLlmBackendError", "JevAnyLlmBackendError"),
    ("JevAnyLlmSchemaError", "JevAnyLlmSchemaError"),
    ("JevAnyLlmError", "JevAnyLlmError"),
    ("JEV_ANY_LLM_OPENAI_BASE_URL", "JEV_ANY_LLM_OPENAI_BASE_URL"),
    ("JEV_ANY_LLM_OPENAI_API_KEY", "JEV_ANY_LLM_OPENAI_API_KEY"),
    ("JEV_ANY_LLM_BACKEND", "JEV_ANY_LLM_BACKEND"),
    ("JEV_ANY_LLM_MODEL", "JEV_ANY_LLM_MODEL"),
    ("JEV_ANY_LLM_*", "JEV_ANY_LLM_*"),
    ("jev-any-llm-rlcd", "jev-any-llm-rlcd"),
    ("jev-any-llm-serve", "jev-any-llm-serve"),
    ("jev-any-llm", "jev-any-llm"),
    ("pip install 'jev-any-llm[", "pip install 'jev-any-llm["),
    ('pip install "jev-any-llm[', 'pip install "jev-any-llm['),
    ("from jev_any_llm.", "from jev_any_llm."),
    ("import jev_any_llm.", "import jev_any_llm."),
    ("from jev_any_llm import", "from jev_any_llm import"),
    ("import kev\n", "import jev_any_llm\n"),
    ("src/jev_any_llm/", "src/jev_any_llm/"),
    ("src/jev_any_llm{", "src/jev_any_llm{"),
    ("jev_any_llm.serve", "jev_any_llm.serve"),
    ('code = "jev_any_llm_error"', 'code = "jev_any_llm_error"'),
    ('owned_by": "jev-any-llm"', 'owned_by": "jev-any-llm"'),
    ("FastAPI(title=\"jev-any-llm\"", 'FastAPI(title="jev-any-llm"'),
    ("Serve jev-any-llm decide()", "Serve jev-any-llm decide()"),
    ("requires fastapi: pip install 'jev-any-llm[serve]'", "requires fastapi: pip install 'jev-any-llm[serve]'"),
]

PROSE = [
    (r"\bKev’s\b", "jev-any-llm's"),
    (r"\bKev's\b", "jev-any-llm's"),
    (r"\bKev\b", "jev-any-llm"),
]


def main() -> None:
    changed: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix not in EXTS and path.name != "pyproject.toml":
            continue
        text = path.read_text(encoding="utf-8")
        orig = text
        for old, new in SUBS:
            text = text.replace(old, new)
        for pat, repl in PROSE:
            text = re.sub(pat, repl, text)
        text = text.replace("jev-any-llm", "jev-any-llm")
        if path.name == "pyproject.toml":
            text = text.replace('name = "kev"', 'name = "jev-any-llm"')
            text = text.replace(
                'authors = [{ name = "kev" }]',
                'authors = [{ name = "jev-any-llm" }]',
            )
            # after jev-any-llm-serve → jev-any-llm-serve and jev_any_llm.serve → jev_any_llm.serve
            text = text.replace(
                'jev-any-llm-serve = "jev_any_llm.serve:main"',
                'jev-any-llm-serve = "jev_any_llm.serve:main"',
            )
            if 'jev-any-llm-serve =' not in text and "serve:main" in text:
                text = re.sub(
                    r'^.*serve:main.*$',
                    'jev-any-llm-serve = "jev_any_llm.serve:main"',
                    text,
                    flags=re.M,
                )
        if path.name == "errors.py" or "errors.py" in str(path):
            text = text.replace('code = "jev_any_llm_error"', 'code = "jev_any_llm_error"')
        if text != orig:
            path.write_text(text, encoding="utf-8")
            changed.append(str(path.relative_to(ROOT)))
    print(f"updated {len(changed)} files")
    for item in changed:
        print(" ", item)


if __name__ == "__main__":
    main()
