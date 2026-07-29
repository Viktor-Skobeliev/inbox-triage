"""Runtime settings, read once from the environment.

The provider is configuration, not code: point ``LLM_PROVIDER`` at OpenRouter,
OpenAI, Gemini or anything else that speaks the OpenAI chat API, or set
``LLM_BASE_URL`` directly for a gateway that is not in the presets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_PROVIDER = "openrouter"
DEFAULT_CACHE_DIR = ".cache/llm"

# Checked in order, so an existing provider-specific key in the environment
# works without renaming anything.
KEY_ENV_NAMES: dict[str, tuple[str, ...]] = {
    "openrouter": ("LLM_API_KEY", "OPENROUTER_API_KEY"),
    "openai": ("LLM_API_KEY", "OPENAI_API_KEY"),
    "gemini": ("LLM_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "groq": ("LLM_API_KEY", "GROQ_API_KEY"),
    "together": ("LLM_API_KEY", "TOGETHER_API_KEY"),
}
FALLBACK_KEY_NAMES = ("LLM_API_KEY",)


@dataclass(frozen=True)
class Settings:
    provider: str
    api_key: str | None
    model: str | None
    base_url: str | None
    temperature: float
    max_attempts: int
    concurrency: int
    json_mode: bool
    cache_dir: Path | None

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _bool_env(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _first_key(provider: str) -> str | None:
    for name in KEY_ENV_NAMES.get(provider, FALLBACK_KEY_NAMES):
        value = (os.getenv(name) or "").strip()
        if value:
            return value
    return None


def load_settings(env_file: Path | None = None) -> Settings:
    load_dotenv(dotenv_path=env_file, override=False)
    provider = (os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
    # An explicitly empty value means "no cache", which is not the same as the
    # variable being absent. `or` would collapse the two.
    cache_declared = os.getenv("LLM_CACHE_DIR")
    cache_raw = (DEFAULT_CACHE_DIR if cache_declared is None else cache_declared).strip()
    return Settings(
        provider=provider,
        api_key=_first_key(provider),
        model=(os.getenv("LLM_MODEL") or "").strip() or None,
        base_url=(os.getenv("LLM_BASE_URL") or "").strip() or None,
        temperature=_float_env("LLM_TEMPERATURE", 0.0),
        max_attempts=max(1, _int_env("LLM_MAX_ATTEMPTS", 3)),
        concurrency=max(1, _int_env("LLM_CONCURRENCY", 4)),
        json_mode=_bool_env("LLM_JSON_MODE", True),
        cache_dir=Path(cache_raw) if cache_raw else None,
    )
