"""Settings resolution and the response cache."""

from __future__ import annotations

from pathlib import Path

import pytest

from inbox_triage.config import load_settings
from inbox_triage.llm.base import CachingClient
from inbox_triage.llm.fake import FakeLLMClient, valid_json

ALL_KEY_NAMES = (
    "LLM_API_KEY",
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GROQ_API_KEY",
    "TOGETHER_API_KEY",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        *ALL_KEY_NAMES,
        "LLM_PROVIDER",
        "LLM_MODEL",
        "LLM_BASE_URL",
        "LLM_TEMPERATURE",
        "LLM_MAX_ATTEMPTS",
        "LLM_CONCURRENCY",
        "LLM_JSON_MODE",
        "LLM_CACHE_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    # Otherwise a developer's own .env leaks into the assertions.
    monkeypatch.setattr("inbox_triage.config.load_dotenv", lambda *a, **k: False)


class TestKeyResolution:
    def test_generic_key_is_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_API_KEY", "generic")
        assert load_settings().api_key == "generic"

    def test_provider_specific_key_is_picked_up(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "from-openai")
        assert load_settings().api_key == "from-openai"

    def test_generic_key_wins_over_provider_specific(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_API_KEY", "generic")
        monkeypatch.setenv("OPENAI_API_KEY", "specific")
        assert load_settings().api_key == "generic"

    def test_key_from_another_provider_is_not_borrowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # An OpenAI key would simply be rejected by OpenRouter, so silently
        # reusing it would produce a confusing 401 instead of a clear message.
        monkeypatch.setenv("LLM_PROVIDER", "openrouter")
        monkeypatch.setenv("OPENAI_API_KEY", "specific")
        assert load_settings().api_key is None

    def test_blank_key_counts_as_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_API_KEY", "   ")
        assert load_settings().has_api_key is False


class TestDefaultsAndParsing:
    def test_defaults(self) -> None:
        settings = load_settings()
        assert settings.provider == "openrouter"
        assert settings.temperature == 0.0
        assert settings.max_attempts == 3
        assert settings.concurrency == 4
        assert settings.json_mode is True
        assert settings.cache_dir == Path(".cache/llm")

    def test_garbage_numbers_fall_back_instead_of_crashing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_TEMPERATURE", "hot")
        monkeypatch.setenv("LLM_MAX_ATTEMPTS", "many")
        settings = load_settings()
        assert settings.temperature == 0.0
        assert settings.max_attempts == 3

    def test_attempts_and_concurrency_cannot_drop_below_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_MAX_ATTEMPTS", "0")
        monkeypatch.setenv("LLM_CONCURRENCY", "-5")
        settings = load_settings()
        assert settings.max_attempts == 1
        assert settings.concurrency == 1

    def test_json_mode_accepts_the_usual_spellings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for value, expected in (("false", False), ("0", False), ("on", True), ("YES", True)):
            monkeypatch.setenv("LLM_JSON_MODE", value)
            assert load_settings().json_mode is expected

    def test_empty_cache_dir_disables_the_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_CACHE_DIR", "")
        assert load_settings().cache_dir is None

    def test_provider_is_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "OpenRouter")
        assert load_settings().provider == "openrouter"


class TestCache:
    def test_second_identical_call_does_not_reach_the_inner_client(self, tmp_path: Path) -> None:
        inner = FakeLLMClient(lambda _: valid_json())
        cached = CachingClient(inner, tmp_path / "c", fingerprint="fp")

        cached.generate("той самий промпт", system="sys")
        cached.generate("той самий промпт", system="sys")

        assert len(inner.calls) == 1
        assert cached.hits == 1

    def test_a_different_system_prompt_is_a_different_key(self, tmp_path: Path) -> None:
        inner = FakeLLMClient(lambda _: valid_json())
        cached = CachingClient(inner, tmp_path / "c", fingerprint="fp")

        cached.generate("промпт", system="перша роль")
        cached.generate("промпт", system="інша роль")

        assert len(inner.calls) == 2

    def test_fingerprint_change_invalidates_everything(self, tmp_path: Path) -> None:
        # Model, temperature and prompt version go into the fingerprint, so a
        # changed prompt cannot be answered from an older run's cache.
        inner_one = FakeLLMClient(lambda _: valid_json())
        CachingClient(inner_one, tmp_path / "c", fingerprint="v1").generate("промпт")

        inner_two = FakeLLMClient(lambda _: valid_json())
        CachingClient(inner_two, tmp_path / "c", fingerprint="v2").generate("промпт")

        assert len(inner_two.calls) == 1

    def test_corrupted_cache_entry_is_discarded_not_fatal(self, tmp_path: Path) -> None:
        inner = FakeLLMClient(lambda _: valid_json())
        cached = CachingClient(inner, tmp_path / "c", fingerprint="fp")
        cached.generate("промпт")

        for entry in (tmp_path / "c").glob("*.json"):
            entry.write_text("{ broken", encoding="utf-8")

        response = cached.generate("промпт")
        assert response.text
        assert len(inner.calls) == 2

    def test_cached_response_reports_zero_tokens(self, tmp_path: Path) -> None:
        # A cache hit costs nothing, and the run metadata must not pretend it did.
        inner = FakeLLMClient(lambda _: valid_json())
        cached = CachingClient(inner, tmp_path / "c", fingerprint="fp")
        cached.generate("промпт")
        second = cached.generate("промпт")
        assert second.cached is True
        assert second.usage.total == 0

    def test_concurrent_writers_do_not_corrupt_entries(self, tmp_path: Path) -> None:
        # --concurrency puts several threads on the same cache directory, so
        # entries are written through a temporary file and renamed.
        from concurrent.futures import ThreadPoolExecutor

        inner = FakeLLMClient(lambda p: valid_json(short_summary=f"суть {len(p)}"))
        cached = CachingClient(inner, tmp_path / "c", fingerprint="fp")
        prompts = [f"запит номер {i}" for i in range(24)]

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda p: cached.generate(p), prompts))
        with ThreadPoolExecutor(max_workers=8) as pool:
            second = list(pool.map(lambda p: cached.generate(p), prompts))

        assert all(response.text for response in second)
        assert cached.hits == len(prompts)
        # No half-written files left behind.
        assert not list((tmp_path / "c").glob("*.tmp"))

    def test_unicode_prompts_round_trip(self, tmp_path: Path) -> None:
        inner = FakeLLMClient(lambda _: valid_json(short_summary="Суть із апострофом: обʼєкт."))
        cached = CachingClient(inner, tmp_path / "c", fingerprint="fp")
        cached.generate("запит з ʼапострофомʼ та емодзі 🙌")
        second = cached.generate("запит з ʼапострофомʼ та емодзі 🙌")
        assert "обʼєкт" in second.text
