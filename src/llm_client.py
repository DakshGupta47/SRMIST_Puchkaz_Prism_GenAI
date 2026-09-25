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

Whichever provider is used, `enrichment.py` always programmatically
validates the LLM's output (Stage 0's own "never trust the LLM to
self-constrain" rule, same principle Member B applies to Stage 1) — the
LLM proposes, the code disposes.
"""
from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from typing import List


class LLMClient(ABC):
    """Minimal interface: give it a task, get raw text back."""

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        ...


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


class GeminiLLMClient(LLMClient):
    def __init__(self, model: str | None = None):
        import google.generativeai as genai  # lazy import

        api_key = os.environ["GOOGLE_API_KEY"]
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model or os.environ.get("LLM_MODEL", "gemini-2.0-flash"))

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        resp = self._model.generate_content([system_prompt, user_prompt])
        return resp.text


class OpenAILLMClient(LLMClient):
    def __init__(self, model: str | None = None):
        from openai import OpenAI  # lazy import

        self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
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


class AnthropicLLMClient(LLMClient):
    def __init__(self, model: str | None = None):
        import anthropic  # lazy import

        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self._model = model or os.environ.get("LLM_MODEL", "claude-haiku-4-5")

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(block.text for block in resp.content if hasattr(block, "text"))


def get_llm_client() -> LLMClient:
    """Factory: picks the provider from LLM_PROVIDER, falls back to the
    offline mock if unset, unrecognized, or the provider fails to
    initialize (e.g. missing API key) — so a missing key degrades the
    quality of Stage 0's output but never breaks the API.
    """
    provider = os.environ.get("LLM_PROVIDER", "mock").lower()
    try:
        if provider == "gemini":
            return GeminiLLMClient()
        if provider == "openai":
            return OpenAILLMClient()
        if provider == "anthropic":
            return AnthropicLLMClient()
    except Exception:
        pass
    return MockLLMClient()
