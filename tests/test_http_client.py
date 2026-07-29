"""HTTP client behaviour, driven through a mock transport.

No network, no key, no waiting: the sleep function is replaced so backoff is
exercised without actually pausing.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from inbox_triage.llm.base import LLMError
from inbox_triage.llm.http_client import (
    PROVIDER_BASE_URLS,
    ChatClient,
    resolve_base_url,
    resolve_model,
)


def ok_body(content: str = '{"ok": true}', **usage: int) -> dict[str, Any]:
    return {
        "choices": [
            {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens", 120),
            "completion_tokens": usage.get("completion_tokens", 45),
        },
    }


def build(handler: Any, **kwargs: Any) -> ChatClient:
    slept: list[float] = kwargs.pop("slept", [])
    return ChatClient(
        api_key="test-key",
        model=kwargs.pop("model", "test/model"),
        base_url=kwargs.pop("base_url", "https://example.test/v1"),
        transport=httpx.MockTransport(handler),
        sleep=slept.append,
        **kwargs,
    )


class TestProviderResolution:
    def test_known_providers_have_a_base_url(self) -> None:
        for provider in PROVIDER_BASE_URLS:
            assert resolve_base_url(provider).startswith("https://")

    def test_explicit_base_url_wins(self) -> None:
        assert resolve_base_url("openrouter", "https://gateway.local/v1/") == (
            "https://gateway.local/v1"
        )

    def test_unknown_provider_without_base_url_is_a_clear_error(self) -> None:
        with pytest.raises(LLMError, match="unknown provider"):
            resolve_base_url("no-such-provider")

    def test_unknown_provider_is_fine_with_an_explicit_url(self) -> None:
        assert (
            resolve_base_url("whatever", "https://gateway.local/v1") == "https://gateway.local/v1"
        )

    def test_model_override_beats_the_preset(self) -> None:
        chosen = resolve_model("openrouter", "anthropic/claude-sonnet-4")
        assert chosen == "anthropic/claude-sonnet-4"

    def test_provider_without_a_default_model_returns_empty(self) -> None:
        assert resolve_model("custom-gateway") == ""


class TestRequestShape:
    def test_authorization_header_and_payload(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["auth"] = request.headers.get("authorization")
            seen["url"] = str(request.url)
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=ok_body())

        client = build(handler)
        client.generate("розбери запит", system="ти асистент")

        assert seen["auth"] == "Bearer test-key"
        assert seen["url"].endswith("/chat/completions")
        assert seen["body"]["messages"][0] == {"role": "system", "content": "ти асистент"}
        assert seen["body"]["messages"][1]["content"] == "розбери запит"
        assert seen["body"]["temperature"] == 0.0
        assert seen["body"]["response_format"] == {"type": "json_object"}

    def test_system_message_is_omitted_when_absent(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=ok_body())

        build(handler).generate("привіт")
        assert len(seen["body"]["messages"]) == 1
        assert seen["body"]["messages"][0]["role"] == "user"

    def test_json_mode_can_be_disabled(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=ok_body())

        build(handler, json_mode=False).generate("привіт")
        assert "response_format" not in seen["body"]

    def test_openrouter_attribution_headers_are_sent_when_configured(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["referer"] = request.headers.get("http-referer")
            seen["title"] = request.headers.get("x-title")
            return httpx.Response(200, json=ok_body())

        build(handler, app_url="https://example.org", app_title="inbox-triage").generate("привіт")
        assert seen["referer"] == "https://example.org"
        assert seen["title"] == "inbox-triage"


class TestResponseHandling:
    def test_content_and_usage_are_returned(self) -> None:
        client = build(lambda _: httpx.Response(200, json=ok_body('{"a": 1}')))
        response = client.generate("привіт")
        assert response.text == '{"a": 1}'
        assert response.usage.prompt_tokens == 120
        assert response.usage.total == 165

    def test_missing_usage_block_is_not_fatal(self) -> None:
        body = ok_body()
        body.pop("usage")
        response = build(lambda _: httpx.Response(200, json=body)).generate("привіт")
        assert response.usage.total == 0

    def test_empty_content_is_an_error_with_the_finish_reason(self) -> None:
        body = {"choices": [{"message": {"content": ""}, "finish_reason": "content_filter"}]}
        with pytest.raises(LLMError, match="content_filter"):
            build(lambda _: httpx.Response(200, json=body)).generate("привіт")

    def test_null_content_is_an_error(self) -> None:
        body = {"choices": [{"message": {"content": None}, "finish_reason": "length"}]}
        with pytest.raises(LLMError, match="empty content"):
            build(lambda _: httpx.Response(200, json=body)).generate("привіт")

    def test_no_choices_is_an_error(self) -> None:
        with pytest.raises(LLMError, match="no choices"):
            build(lambda _: httpx.Response(200, json={"choices": []})).generate("привіт")

    def test_error_object_returned_with_status_200(self) -> None:
        # OpenRouter does this when an upstream provider refuses.
        body = {"error": {"message": "upstream provider is down", "code": 502}}
        with pytest.raises(LLMError, match="upstream provider is down"):
            build(lambda _: httpx.Response(200, json=body)).generate("привіт")

    def test_non_json_body(self) -> None:
        with pytest.raises(LLMError, match="not JSON"):
            build(lambda _: httpx.Response(200, text="<html>gateway</html>")).generate("привіт")


class TestFailures:
    def test_auth_error_is_not_retried(self) -> None:
        calls: list[int] = []

        def handler(_: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(401, json={"error": {"message": "invalid key"}})

        with pytest.raises(LLMError, match="auth rejected"):
            build(handler).generate("привіт")
        assert len(calls) == 1

    def test_unknown_model_gives_an_actionable_message(self) -> None:
        handler = lambda _: httpx.Response(404, json={"error": {"message": "no such model"}})  # noqa: E731
        with pytest.raises(LLMError, match="LLM_MODEL"):
            build(handler).generate("привіт")

    def test_rate_limit_is_retried_then_succeeds(self) -> None:
        calls: list[int] = []
        slept: list[float] = []

        def handler(_: httpx.Request) -> httpx.Response:
            calls.append(1)
            if len(calls) < 3:
                return httpx.Response(429, json={"error": {"message": "slow down"}})
            return httpx.Response(200, json=ok_body())

        client = build(handler, slept=slept)
        assert client.generate("привіт").text == '{"ok": true}'
        assert len(calls) == 3
        # Backoff grows rather than hammering the API.
        assert len(slept) == 2 and slept[1] > slept[0]

    def test_persistent_rate_limit_gives_up_with_context(self) -> None:
        calls: list[int] = []

        def handler(_: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(429, json={"error": {"message": "quota exhausted"}})

        with pytest.raises(LLMError, match="quota exhausted"):
            build(handler, slept=[]).generate("привіт")
        assert len(calls) == 4

    def test_timeout_is_retried(self) -> None:
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            if len(calls) == 1:
                raise httpx.ReadTimeout("too slow", request=request)
            return httpx.Response(200, json=ok_body())

        assert build(handler, slept=[]).generate("привіт").text == '{"ok": true}'
        assert len(calls) == 2

    def test_json_mode_is_dropped_when_the_model_rejects_it(self) -> None:
        # Not every model behind a gateway supports response_format; losing the
        # row over it would be worse than parsing plain text.
        bodies: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            bodies.append(body)
            if "response_format" in body:
                return httpx.Response(
                    400, json={"error": {"message": "response_format is not supported"}}
                )
            return httpx.Response(200, json=ok_body())

        assert build(handler).generate("привіт").text == '{"ok": true}'
        assert len(bodies) == 2
        assert "response_format" in bodies[0]
        assert "response_format" not in bodies[1]

    def test_other_400_is_not_silently_retried_forever(self) -> None:
        calls: list[int] = []

        def handler(_: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(400, json={"error": {"message": "context length exceeded"}})

        with pytest.raises(LLMError, match="context length"):
            build(handler).generate("привіт")
        assert len(calls) == 1


class TestSecrets:
    def test_api_key_never_appears_in_the_client_name(self) -> None:
        client = build(lambda _: httpx.Response(200, json=ok_body()))
        assert "test-key" not in client.name

    def test_api_key_never_appears_in_an_error_message(self) -> None:
        handler = lambda _: httpx.Response(500, json={"error": {"message": "boom"}})  # noqa: E731
        client = build(handler, slept=[])
        with pytest.raises(LLMError) as caught:
            client.generate("привіт")
        assert "test-key" not in str(caught.value)
