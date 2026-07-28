"""One HTTP client for any OpenAI-compatible chat API.

OpenRouter, OpenAI and Google all speak the same ``/chat/completions`` shape,
so the provider is a base URL and a key, not a separate SDK each. Whoever runs
this picks a provider in ``.env`` and nothing in the pipeline changes.

Retries here cover transport only: rate limits, timeouts, 5xx. A response that
arrives but is unusable belongs to the validation layer, not to this one.

One degradation is deliberate: ``response_format=json_object`` is not supported
by every model behind OpenRouter. If the server rejects it, the request is
repeated without it rather than failing the row, since the parser can recover
JSON from plain text anyway.
"""

from __future__ import annotations

import random
import time
from typing import Any

import httpx

from ..models import TokenUsage
from .base import LLMError, LLMResponse

PROVIDER_BASE_URLS: dict[str, str] = {
    "openrouter": "https://openrouter.ai/api/v1",
    "openai": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "groq": "https://api.groq.com/openai/v1",
    "together": "https://api.together.xyz/v1",
}

DEFAULT_MODELS: dict[str, str] = {
    "openrouter": "google/gemini-2.5-flash",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
    "groq": "llama-3.3-70b-versatile",
    "together": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
}

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
MAX_TRANSPORT_ATTEMPTS = 4
BASE_BACKOFF_SECONDS = 2.0
DEFAULT_TIMEOUT_SECONDS = 90.0


def resolve_base_url(provider: str, override: str | None = None) -> str:
    if override:
        return override.rstrip("/")
    try:
        return PROVIDER_BASE_URLS[provider]
    except KeyError:
        known = ", ".join(sorted(PROVIDER_BASE_URLS))
        raise LLMError(
            f"unknown provider '{provider}'. Known: {known}. "
            "For anything else set LLM_BASE_URL explicitly."
        ) from None


def resolve_model(provider: str, override: str | None = None) -> str:
    if override:
        return override
    return DEFAULT_MODELS.get(provider, "")


class ChatClient:
    """Minimal chat-completions client.

    Deliberately not the vendor SDK: the whole surface used here is one POST,
    and a direct call keeps the dependency list short and the failure modes
    visible.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        temperature: float = 0.0,
        max_output_tokens: int = 2048,
        json_mode: bool = True,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        transport: httpx.BaseTransport | None = None,
        sleep: Any = time.sleep,
        app_url: str | None = None,
        app_title: str | None = None,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._json_mode = json_mode
        self._sleep = sleep
        self.name = f"{base_url.split('//')[-1].split('/')[0]}:{model}"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        # OpenRouter attributes traffic with these; harmless elsewhere.
        if app_url:
            headers["HTTP-Referer"] = app_url
        if app_title:
            headers["X-Title"] = app_title

        self._client = httpx.Client(
            base_url=base_url, headers=headers, timeout=timeout, transport=transport
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> ChatClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _payload(self, prompt: str, system: str | None, json_mode: bool) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_output_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def generate(self, prompt: str, *, system: str | None = None) -> LLMResponse:
        last_error = ""
        attempt = 0

        while attempt < MAX_TRANSPORT_ATTEMPTS:
            attempt += 1
            try:
                response = self._client.post(
                    "/chat/completions", json=self._payload(prompt, system, self._json_mode)
                )
            # TransportError also covers ConnectError and ConnectTimeout: a
            # refused connection is every bit as transient as a slow read.
            except httpx.TransportError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            except httpx.HTTPError as exc:
                raise LLMError(f"{self.name}: transport error: {exc}") from exc
            else:
                if response.status_code == 200:
                    return self._read(response)

                detail = _error_detail(response)

                # Some models behind a gateway reject JSON mode. Drop it for
                # the whole run rather than for this call, otherwise every row
                # pays the same rejected request again. It is not a transport
                # failure either, so it must not consume an attempt.
                if response.status_code == 400 and self._json_mode and "response_format" in detail:
                    self._json_mode = False
                    attempt -= 1
                    continue

                if response.status_code in {401, 403}:
                    raise LLMError(f"{self.name}: auth rejected ({response.status_code}). {detail}")
                if response.status_code == 404:
                    raise LLMError(
                        f"{self.name}: model or endpoint not found (404). "
                        f"Check LLM_MODEL and LLM_BASE_URL. {detail}"
                    )
                if response.status_code not in RETRYABLE_STATUS:
                    raise LLMError(f"{self.name}: HTTP {response.status_code}. {detail}")

                last_error = f"HTTP {response.status_code}: {detail}"

            if attempt >= MAX_TRANSPORT_ATTEMPTS:
                break
            self._sleep(BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, 0.5))

        raise LLMError(
            f"{self.name}: giving up after {MAX_TRANSPORT_ATTEMPTS} attempts. {last_error}"
        )

    def _read(self, response: httpx.Response) -> LLMResponse:
        try:
            body = response.json()
        except ValueError as exc:
            raise LLMError(f"{self.name}: response body is not JSON: {exc}") from exc

        # OpenRouter can answer 200 with an error object instead of choices.
        if isinstance(body.get("error"), dict):
            raise LLMError(f"{self.name}: {body['error'].get('message', body['error'])}")

        choices = body.get("choices") or []
        if not choices:
            raise LLMError(f"{self.name}: response contains no choices")

        message = choices[0].get("message") or {}
        text = (message.get("content") or "").strip()
        if not text:
            reason = choices[0].get("finish_reason", "unknown")
            raise LLMError(f"{self.name}: empty content (finish_reason: {reason})")

        usage_block = body.get("usage") or {}
        return LLMResponse(
            text=text,
            usage=TokenUsage(
                prompt_tokens=int(usage_block.get("prompt_tokens") or 0),
                completion_tokens=int(usage_block.get("completion_tokens") or 0),
            ),
        )


def _as_int(value: Any) -> int:
    """Token counts arrive as strings, floats or nulls depending on the gateway."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _error_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:300]
    error = body.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or error)[:300]
    if isinstance(error, str):
        return error[:300]
    return str(body)[:300]
