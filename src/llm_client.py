"""
Pluggable LLM client for Stage 0 (query enrichment).

No provider/API key had been chosen yet when this was built, so this
module ships:
  * a common `LLMClient` interface
  * ready adapters for Gemini, OpenAI and Anthropic (each ~10 lines,
    activate by setting env vars — see below)
  * a dependency-free `MockLLMClient` fallback that uses templated
    paraphrasing rules instead of an API call, so the pipeline always
    returns a valid response even with no key configured (never crashes,
    never blocks the rest of the team on an API key)

Configure via environment variables:
    LLM_PROVIDER = "gemini" | "openai" | "anthropic" | "mock"   (default: mock)
    GOOGLE_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY as needed
    LLM_MODEL   (optional override of the default model per provider)

Easiest way to set these: copy .env.example (project root) to .env and fill
in the two lines. python-dotenv (if installed — it's in requirements.txt)
loads .env automatically on import, below. No .env / no dotenv installed?
Nothing breaks — get_llm_client() falls back to the mock client exactly as
it always did. .env is gitignored so a real key never gets committed.

Whichever provider is used, `enrichment.py` always programmatically
validates the LLM's output (Stage 0's own "never trust the LLM to
self-constrain" rule, same principle Member B applies to Stage 1) — the
LLM proposes, the code disposes.
"""
from __future__ import annotations

import concurrent.futures
import os
import re
from abc import ABC, abstractmethod
from typing import List

try:
    from dotenv import load_dotenv  # optional dependency, see requirements.txt

    load_dotenv()  # no-op if there's no .env file; never raises
except ImportError:
    pass  # python-dotenv not installed — env vars still work if set another way


class LLMClient(ABC):
    """Minimal interface: give it a task, get raw text back."""

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        ...

    @property
    def model_name(self) -> str:
        """Identifier for what actually produced a response, surfaced in the
        API's meta.model field (spec Appendix B's worked example shows
        "model": "gpt-4o-mini" there). Defaults to "mock" so a client that
        doesn't override this (MockLLMClient) still reports an honest,
        specific value instead of nothing.
        """
        return "mock"


class MockLLMClient(LLMClient):
    """Deterministic, offline, rule-based stand-in for a real LLM.

    Used automatically when LLM_PROVIDER is unset/"mock", or as the
    automatic fallback if a configured provider errors out. It doesn't
    "understand" the complaint the way an LLM would, so `enrichment.py`
    layers real normalization/variation-generation rules on top rather
    than trusting this for anything beyond a text echo — this class only
    exists so the rest of the system never has a hard dependency on an
    API key being present.
    """

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return user_prompt


# Every network call below gets an explicit, short timeout. Found by testing,
# not by inspection: a blocked/unreachable host (e.g. a provider whose domain
# isn't on this network's egress allowlist — the same class of issue that
# blocked huggingface.co for the embedding model) doesn't fail fast on its
# own here. A bare `curl` to it returns instantly with a clear 403 from the
# proxy, but the Gemini SDK's own retry/backoff logic swallowed that and hung
# well past 60 seconds with no exception raised. Since this call sits in
# every request's critical path (see pipeline.py's module docstring), an
# unbounded hang is far worse than a clean, fast failure into the mock
# fallback — it would silently blow the "fast" requirement instead of
# visibly degrading quality. 8 seconds is generous for a short paraphrase/
# classification call but short enough that a hung provider can't matter.
_LLM_TIMEOUT_SECONDS = 8


class GeminiLLMClient(LLMClient):
    def __init__(self, model: str | None = None):
        import google.generativeai as genai  # lazy import

        api_key = os.environ["GOOGLE_API_KEY"]
        genai.configure(api_key=api_key)
        self._model_name = model or os.environ.get("LLM_MODEL", "gemini-3.5-flash-lite")
        self._model = genai.GenerativeModel(self._model_name)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        resp = self._model.generate_content(
            [system_prompt, user_prompt],
            request_options={"timeout": _LLM_TIMEOUT_SECONDS},
        )
        return resp.text

    @property
    def model_name(self) -> str:
        return self._model_name


class OpenAILLMClient(LLMClient):
    def __init__(self, model: str | None = None):
        from openai import OpenAI  # lazy import

        self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=_LLM_TIMEOUT_SECONDS)
        self._model = model or os.environ.get("LLM_MODEL", "gpt-4o-mini")

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content or ""

    @property
    def model_name(self) -> str:
        return self._model


class AnthropicLLMClient(LLMClient):
    def __init__(self, model: str | None = None):
        import anthropic  # lazy import

        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], timeout=_LLM_TIMEOUT_SECONDS)
        self._model = model or os.environ.get("LLM_MODEL", "claude-haiku-4-5")

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(block.text for block in resp.content if hasattr(block, "text"))

    @property
    def model_name(self) -> str:
        return self._model


class _TimeoutGuardedClient(LLMClient):
    """Wraps a real provider client and enforces a hard wall-clock timeout
    around complete() from the OUTSIDE, independent of whatever timeout
    parameter the underlying SDK claims to honor. Necessary because one
    isn't enough: found by testing that a blocked/unreachable host can make
    a provider's own internal retry/handshake logic hang far past its
    documented per-request timeout (observed directly: Gemini's gRPC
    transport hung 60+ seconds against a network that blocks its endpoint,
    despite `request_options={"timeout": 8}` on the call itself — the hang
    happens at a connection layer that parameter doesn't reach). Since this
    call sits in every request's critical path, an unbounded hang here is a
    much worse failure than a fast, clean fallback to the mock generator.

    A background thread that exceeds the timeout is abandoned, not killed
    (Python cannot forcibly stop a running thread) — but the caller is
    never blocked longer than `timeout_seconds` regardless, which is the
    property that actually matters for the API's own latency budget.
    """

    def __init__(self, inner: LLMClient, timeout_seconds: float = _LLM_TIMEOUT_SECONDS):
        self._inner = inner
        self._timeout = timeout_seconds
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        future = self._executor.submit(self._inner.complete, system_prompt, user_prompt)
        return future.result(timeout=self._timeout)  # raises TimeoutError past the deadline

    @property
    def model_name(self) -> str:
        return self._inner.model_name


# get_llm_client() is called on every enrich() invocation (see enrichment.py),
# so a real provider client — and the thread pool _TimeoutGuardedClient opens
# for it — is built once and reused, not reconstructed per request. Keyed by
# (provider, model) so a test that changes LLM_PROVIDER/LLM_MODEL mid-run
# still gets a fresh client rather than a stale cached one.
_client_cache: dict = {}


def get_llm_client() -> LLMClient:
    """Factory: picks the provider from LLM_PROVIDER, falls back to the
    offline mock if unset, unrecognized, or the provider fails to
    initialize (e.g. missing API key) — so a missing key degrades the
    quality of Stage 0's output but never breaks the API. Any real provider
    is wrapped in _TimeoutGuardedClient so a blocked/slow network degrades
    to the mock generator within `_LLM_TIMEOUT_SECONDS`, not an unbounded
    hang.
    """
    provider = os.environ.get("LLM_PROVIDER", "mock").lower()
    cache_key = (provider, os.environ.get("LLM_MODEL"))
    if cache_key in _client_cache:
        return _client_cache[cache_key]

    client: LLMClient
    try:
        if provider == "gemini":
            client = _TimeoutGuardedClient(GeminiLLMClient())
        elif provider == "openai":
            client = _TimeoutGuardedClient(OpenAILLMClient())
        elif provider == "anthropic":
            client = _TimeoutGuardedClient(AnthropicLLMClient())
        else:
            client = MockLLMClient()
    except Exception:
        client = MockLLMClient()

    _client_cache[cache_key] = client
    return client
