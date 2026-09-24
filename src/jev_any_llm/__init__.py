"""jev-any-llm: wrap any instruct LLM into Jev-mode prediction."""

from jev_any_llm.api import Client, choice, noul, score
from jev_any_llm.calibration import TemperatureProfile
from jev_any_llm.contract import DecideRequest, DecideResponse
from jev_any_llm.errors import JevAnyLlmError, JevAnyLlmValidationError

__all__ = [
    "Client",
    "DecideRequest",
    "DecideResponse",
    "JevAnyLlmError",
    "JevAnyLlmValidationError",
    "TemperatureProfile",
    "choice",
    "noul",
    "score",
]

__version__ = "0.1.0"
