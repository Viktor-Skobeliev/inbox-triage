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
import re
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
                    # No provider detail here: it is the message most likely to
                    # contain the key itself, and this text ends up in the
                    # output file.
                    raise LLMError(
                        f"{self.name}: auth rejected ({response.status_code}), check LLM_API_KEY"
                    )
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
        """Read a 200 response defensively.

        Every field below has been seen wrong behind a gateway, and a raw
        AttributeError here would escape as something the pipeline cannot turn
        into a failed row: it would take the whole run down instead.
        """
        try:
            body = response.json()
        except ValueError as exc:
            raise LLMError(f"{self.name}: response body is not JSON: {exc}") from exc

        if not isinstance(body, dict):
            raise LLMError(f"{self.name}: response body is {type(body).__name__}, expected object")

        # OpenRouter can answer 200 with an error object instead of choices.
        error = body.get("error")
        if isinstance(error, dict):
            raise LLMError(f"{self.name}: {error.get('message', error)}")
        if isinstance(error, str) and error:
            raise LLMError(f"{self.name}: {error}")

        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMError(f"{self.name}: response contains no choices")

        first = choices[0]
        if not isinstance(first, dict):
            raise LLMError(f"{self.name}: malformed choice: {type(first).__name__}")

        message = first.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        text = content.strip() if isinstance(content, str) else ""
        if not text:
            reason = first.get("finish_reason", "unknown")
            raise LLMError(f"{self.name}: empty content (finish_reason: {reason})")

        usage_block = body.get("usage")
        usage_block = usage_block if isinstance(usage_block, dict) else {}
        return LLMResponse(
            text=text,
            usage=TokenUsage(
                prompt_tokens=_as_int(usage_block.get("prompt_tokens")),
                completion_tokens=_as_int(usage_block.get("completion_tokens")),
            ),
        )


def _as_int(value: Any) -> int:
    """Token counts arrive as strings, floats or nulls depending on the gateway."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


# Providers echo the submitted key back in some error messages. Anything that
# reaches an error string can end up in output.json, so it is redacted here.
SECRET_PATTERN = re.compile(r"(sk-[A-Za-z0-9_\-]{8,}|AIza[A-Za-z0-9_\-]{20,}|Bearer\s+\S+)")


def redact(text: str) -> str:
    return SECRET_PATTERN.sub("[REDACTED]", text)


def _error_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return redact(response.text[:300])

    if not isinstance(body, dict):
        # Gateways and proxies do return a bare list here.
        return redact(str(body)[:300])

    error = body.get("error")
    if isinstance(error, dict):
        return redact(str(error.get("message") or error)[:300])
    if isinstance(error, str):
        return redact(error[:300])
    return redact(str(body)[:300])
