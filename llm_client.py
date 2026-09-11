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
    stricter reformatting instruction when the response is malformed JSON
    or does not satisfy `response_schema`, then raises `LLMError` rather
    than fabricating or returning a result the caller cannot trust.
    """

    def __init__(self, provider: str | None = None) -> None:
        self._provider = provider or os.environ.get("LLM_PROVIDER", "gemini")

    def call_llm(self, prompt: str, system: str, response_schema: dict) -> dict:
        raw_text = self._call_provider(prompt, system, response_schema)
        try:
            return _parse_json_object(raw_text, response_schema)
        except LLMError:
            raw_text = self._call_provider(
                f"{prompt}\n\n{_REFORMAT_INSTRUCTION}", system, response_schema
            )
            return _parse_json_object(raw_text, response_schema)

    def _call_provider(self, prompt: str, system: str, response_schema: dict) -> str:
        if self._provider == "gemini":
            return _call_gemini(prompt, system, response_schema)
        if self._provider == "openai":
            return _call_openai(prompt, system, response_schema)
        raise LLMError(f"Unsupported LLM_PROVIDER: {self._provider!r}")


def _parse_json_object(raw_text: str, response_schema: dict) -> dict:
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise LLMError(f"LLM response was not valid JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise LLMError("LLM response JSON was not an object.")
    _validate_schema(parsed, response_schema, "response")
    return parsed


_SCHEMA_TYPE_CHECKS = {
    "object": lambda value: isinstance(value, dict),
    "array": lambda value: isinstance(value, list),
    "string": lambda value: isinstance(value, str),
    "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
    "number": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool),
    "boolean": lambda value: isinstance(value, bool),
}


def _validate_schema(value: object, schema: dict, path: str) -> None:
    """Raise `LLMError` describing the first mismatch between `value` and `schema`.

    Supports the subset of JSON Schema used by the agent response schemas:
    `type`, `enum`, `nullable`, `required`, `properties`, and `items`.
    """
    if value is None:
        if schema.get("nullable"):
            return
        raise LLMError(f"{path} must not be null.")

    enum_values = schema.get("enum")
    if enum_values is not None and value not in enum_values:
        raise LLMError(f"{path} must be one of {enum_values!r}, got {value!r}.")

    expected_type = schema.get("type")
    if expected_type is not None:
        type_check = _SCHEMA_TYPE_CHECKS.get(expected_type)
        if type_check is None or not type_check(value):
            raise LLMError(
                f"{path} must be of type {expected_type!r}, got {type(value).__name__!r}."
            )
        if expected_type == "object":
            _validate_object(value, schema, path)
        elif expected_type == "array":
            _validate_array(value, schema, path)


def _validate_object(value: dict, schema: dict, path: str) -> None:
    for required_key in schema.get("required", []):
        if required_key not in value:
            raise LLMError(f"{path} is missing required field {required_key!r}.")
    for key, item_schema in schema.get("properties", {}).items():
        if key in value:
            _validate_schema(value[key], item_schema, f"{path}.{key}")


def _validate_array(value: list, schema: dict, path: str) -> None:
    item_schema = schema.get("items")
    if item_schema is None:
        return
    for index, item in enumerate(value):
        _validate_schema(item, item_schema, f"{path}[{index}]")


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
