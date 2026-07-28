"""LLM boundary.

The rest of the pipeline only knows ``LLMClient``. That keeps the provider
swappable, and more importantly it lets the whole test suite run offline
against a scripted fake, so the validation and repair paths are exercised
without spending a token or needing a key.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from ..models import TokenUsage


@dataclass
class LLMResponse:
    text: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    cached: bool = False


class LLMError(RuntimeError):
    """Provider-side failure: transport, quota, refusal."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class LLMClient(Protocol):
    name: str

    def generate(self, prompt: str, *, system: str | None = None) -> LLMResponse: ...


class CachingClient:
    """Disk cache keyed by the exact request.

    Two reasons it exists, both named in the README: a rerun of the same input
    costs nothing, and it makes a run reproducible even though the provider
    gives no such guarantee. Delete the cache directory to force fresh calls.
    """

    def __init__(self, inner: LLMClient, cache_dir: Path, *, fingerprint: str) -> None:
        self._inner = inner
        self._dir = cache_dir
        self._fingerprint = fingerprint
        self.name = f"cached:{inner.name}"
        self.hits = 0
        self._lock = threading.Lock()
        self._dir.mkdir(parents=True, exist_ok=True)

    def _key(self, prompt: str, system: str | None) -> Path:
        digest = hashlib.sha256(
            json.dumps(
                {"fp": self._fingerprint, "system": system, "prompt": prompt},
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        return self._dir / f"{digest}.json"

    def generate(self, prompt: str, *, system: str | None = None) -> LLMResponse:
        path = self._key(prompt, system)
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                with self._lock:
                    self.hits += 1
                return LLMResponse(text=payload["text"], usage=TokenUsage(), cached=True)
            except (OSError, ValueError, KeyError):
                path.unlink(missing_ok=True)

        response = self._inner.generate(prompt, system=system)
        # A cache that cannot be written is not a reason to fail the run.
        with contextlib.suppress(OSError):
            path.write_text(
                json.dumps({"text": response.text}, ensure_ascii=False), encoding="utf-8"
            )
        return response
