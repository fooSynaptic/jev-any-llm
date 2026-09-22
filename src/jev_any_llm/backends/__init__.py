from jev_any_llm.backends.base import GenerateBackend, LogprobResult  # noqa: F401
from jev_any_llm.backends.mock import MockBackend
from jev_any_llm.backends.openai_compat import OpenAICompatBackend

__all__ = [
    "GenerateBackend",
    "LogprobResult",
    "MockBackend",
    "OpenAICompatBackend",
    "load_hf_backend",
]


def load_hf_backend(*args, **kwargs):
    from jev_any_llm.backends.hf_local import HuggingFaceBackend

    return HuggingFaceBackend(*args, **kwargs)