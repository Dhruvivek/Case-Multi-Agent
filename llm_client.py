"""Provider-agnostic LLM boundary.

Agent code depends only on the `LLMClient` protocol below and never imports
a provider SDK directly. `EnvLLMClient` is the only place that is allowed to
do so, chosen at call time via the `LLM_PROVIDER` environment variable.
"""

from __future__ import annotations

import json
import os
from typing import Protocol


class LLMError(RuntimeError):
    """Raised when the LLM boundary cannot produce a usable JSON response."""


class LLMClient(Protocol):
    def call_llm(self, prompt: str, system: str, response_schema: dict) -> dict:
        """Return a JSON-decoded response constrained to `response_schema`."""
        ...


_REFORMAT_INSTRUCTION = (
    "Your previous reply was not valid JSON matching the required schema. "
    "Reply again with only valid JSON matching the schema, and nothing else."
)


class EnvLLMClient:
    """`LLMClient` that dispatches to a provider chosen by `LLM_PROVIDER`.

    Supported values: "gemini" (default), "openai". Retries once with a
    stricter reformatting instruction on malformed JSON, then raises
    `LLMError` rather than fabricating a result.
    """

    def __init__(self, provider: str | None = None) -> None:
        self._provider = provider or os.environ.get("LLM_PROVIDER", "gemini")

    def call_llm(self, prompt: str, system: str, response_schema: dict) -> dict:
        raw_text = self._call_provider(prompt, system, response_schema)
        try:
            return _parse_json_object(raw_text)
        except LLMError:
            raw_text = self._call_provider(
                f"{prompt}\n\n{_REFORMAT_INSTRUCTION}", system, response_schema
            )
            return _parse_json_object(raw_text)

    def _call_provider(self, prompt: str, system: str, response_schema: dict) -> str:
        if self._provider == "gemini":
            return _call_gemini(prompt, system, response_schema)
        if self._provider == "openai":
            return _call_openai(prompt, system, response_schema)
        raise LLMError(f"Unsupported LLM_PROVIDER: {self._provider!r}")


def _parse_json_object(raw_text: str) -> dict:
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise LLMError(f"LLM response was not valid JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise LLMError("LLM response JSON was not an object.")
    return parsed


def _call_gemini(prompt: str, system: str, response_schema: dict) -> str:
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(
        model_name=os.environ.get("GEMINI_MODEL", "gemini-1.5-flash"),
        system_instruction=system,
    )
    response = model.generate_content(
        prompt,
        generation_config={
            "response_mime_type": "application/json",
            "response_schema": response_schema,
        },
    )
    return response.text


def _call_openai(prompt: str, system: str, response_schema: dict) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content
