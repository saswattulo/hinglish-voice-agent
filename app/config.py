"""Runtime settings, read once from the environment (and .env if present)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

CARTESIA_VERSION = "2026-08-14"


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _float(name: str, default: float) -> float:
    raw = _env(name)
    return float(raw) if raw else default


def _bool(name: str, default: bool) -> bool:
    raw = _env(name).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    cartesia_api_key: str = field(default_factory=lambda: _env("CARTESIA_API_KEY"))
    anthropic_api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY"))

    # Speech-to-text (Cartesia Ink, turn-detection endpoint)
    stt_model: str = field(default_factory=lambda: _env("STT_MODEL", "ink-2"))
    stt_sample_rate: int = 16000
    stt_turn_end_timeout_ms: int = field(
        default_factory=lambda: int(_env("STT_TURN_END_TIMEOUT_MS", "1600"))
    )
    # Clinic words the recogniser should not miss.
    stt_keyterms: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            t.strip()
            for t in _env(
                "STT_KEYTERMS", "appointment,Dr. Sharma,Dr. Iyer,Dr. Khan,OPD,Sanjeevani"
            ).split(",")
            if t.strip()
        )
    )

    # Text-to-speech (Cartesia Sonic)
    tts_model: str = field(default_factory=lambda: _env("TTS_MODEL", "sonic-3.6"))
    tts_voice_id: str = field(
        default_factory=lambda: _env("TTS_VOICE_ID", "694f9389-aac1-45b6-b726-9d9369183238")
    )
    tts_language: str = field(default_factory=lambda: _env("TTS_LANGUAGE", "hi"))
    tts_speed: float = field(default_factory=lambda: _float("TTS_SPEED", 1.0))
    tts_emotion: str = field(default_factory=lambda: _env("TTS_EMOTION", "content"))
    tts_sample_rate: int = 24000
    tts_max_buffer_delay_ms: int = field(
        default_factory=lambda: int(_env("TTS_MAX_BUFFER_DELAY_MS", "200"))
    )

    # LLM (Claude)
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "claude-opus-5-5"))
    llm_effort: str = field(default_factory=lambda: _env("LLM_EFFORT", "low"))
    llm_max_tokens: int = field(default_factory=lambda: int(_env("LLM_MAX_TOKENS", "1024")))

    # Start the LLM on STT's "eager end" signal and keep the work if the user really stopped.
    speculative_llm: bool = field(default_factory=lambda: _bool("SPECULATIVE_LLM", True))

    metrics_dir: Path = field(default_factory=lambda: Path(_env("METRICS_DIR", "logs")))

    def missing_keys(self) -> list[str]:
        missing = []
        if not self.cartesia_api_key:
            missing.append("CARTESIA_API_KEY")
        if not self.anthropic_api_key:
            missing.append("ANTHROPIC_API_KEY")
        return missing


settings = Settings()
