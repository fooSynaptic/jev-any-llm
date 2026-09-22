"""Phases 2–3 native heads — stub until the compiled path lands."""

from __future__ import annotations


class NativeBackend:
    """Placeholder for single-forward Choice / Noul / Score heads."""

    name = "native"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Native single-forward backend is Phase 2–3. Use the wrap path "
            "(Client / OpenAICompatBackend / HuggingFaceBackend) for Phase 1."
        )
