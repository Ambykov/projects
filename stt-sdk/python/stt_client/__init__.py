"""STT Client SDK for Ru Voice Assistant."""
from .audio import AudioSource, WavAudioSource, MicAudioSource
from .client import STTClient
from .errors import STTError, RetryPolicy
from .models import PartialResult, FinalResult, ErrorMessage, ReadyMessage

__version__ = "0.1.1"

__all__ = [
    "STTClient",
    "RetryPolicy",
    "STTError",
    "AudioSource",
    "WavAudioSource",
    "MicAudioSource",
    "PartialResult",
    "FinalResult",
    "ErrorMessage",
    "ReadyMessage",
]
