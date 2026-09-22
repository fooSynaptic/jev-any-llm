"""Typed errors for the wrap contract and backends."""

from __future__ import annotations


class JevAnyLlmError(Exception):
    """Base error. ``code`` is a stable machine string for HTTP / logs."""

    code = "jev_any_llm_error"

    def __init__(self, message: str, *, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict:
        return {"error": self.code, "message": self.message, "details": self.details}


class JevAnyLlmValidationError(JevAnyLlmError):
    code = "validation_error"


class JevAnyLlmBackendError(JevAnyLlmError):
    code = "backend_error"


class JevAnyLlmSchemaError(JevAnyLlmError):
    code = "schema_error"
