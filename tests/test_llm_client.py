from __future__ import annotations

import json

import pytest

import llm_client
from llm_client import EnvLLMClient, LLMError

_EVIDENCE_SCHEMA = {
    "type": "object",
    "required": ["evidence"],
    "properties": {
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "statement", "classification"],
                "properties": {
                    "id": {"type": "string"},
                    "statement": {"type": "string"},
                    "classification": {
                        "type": "string",
                        "enum": ["observed_fact", "inference"],
                    },
                },
            },
        }
    },
}


def test_call_llm_retries_once_after_malformed_json(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter(["not json", '{"evidence": []}'])
    prompts_seen: list[str] = []

    def fake_call_gemini(prompt: str, system: str, response_schema: dict) -> str:
        prompts_seen.append(prompt)
        return next(responses)

    monkeypatch.setattr(llm_client, "_call_gemini", fake_call_gemini)

    client = EnvLLMClient(provider="gemini")
    result = client.call_llm("Extract evidence.", "system prompt", {})

    assert result == {"evidence": []}
    assert prompts_seen[0] == "Extract evidence."
    assert llm_client._REFORMAT_INSTRUCTION in prompts_seen[1]


def test_call_llm_raises_after_two_malformed_responses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        llm_client, "_call_gemini", lambda prompt, system, response_schema: "still not json"
    )

    client = EnvLLMClient(provider="gemini")

    with pytest.raises(LLMError):
        client.call_llm("Extract evidence.", "system prompt", {})


def test_call_llm_raises_for_unsupported_provider() -> None:
    client = EnvLLMClient(provider="carrier-pigeon")

    with pytest.raises(LLMError, match="Unsupported"):
        client.call_llm("Extract evidence.", "system prompt", {})


def test_call_llm_retries_once_after_schema_invalid_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """Valid JSON that violates the response schema is treated the same as malformed JSON."""
    missing_required_field = json.dumps({"evidence": [{"id": "E-01"}]})
    valid_response = json.dumps(
        {
            "evidence": [
                {"id": "E-01", "statement": "The door was locked.", "classification": "observed_fact"}
            ]
        }
    )
    responses = iter([missing_required_field, valid_response])
    prompts_seen: list[str] = []

    def fake_call_gemini(prompt: str, system: str, response_schema: dict) -> str:
        prompts_seen.append(prompt)
        return next(responses)

    monkeypatch.setattr(llm_client, "_call_gemini", fake_call_gemini)

    client = EnvLLMClient(provider="gemini")
    result = client.call_llm("Extract evidence.", "system prompt", _EVIDENCE_SCHEMA)

    assert result["evidence"][0]["id"] == "E-01"
    assert prompts_seen[0] == "Extract evidence."
    assert llm_client._REFORMAT_INSTRUCTION in prompts_seen[1]


def test_call_llm_raises_after_two_schema_invalid_responses(monkeypatch: pytest.MonkeyPatch) -> None:
    invalid_classification = json.dumps(
        {
            "evidence": [
                {"id": "E-01", "statement": "The door was locked.", "classification": "guess"}
            ]
        }
    )
    monkeypatch.setattr(
        llm_client, "_call_gemini", lambda prompt, system, response_schema: invalid_classification
    )

    client = EnvLLMClient(provider="gemini")

    with pytest.raises(LLMError, match="classification"):
        client.call_llm("Extract evidence.", "system prompt", _EVIDENCE_SCHEMA)


def test_call_llm_raises_when_required_field_missing_twice(monkeypatch: pytest.MonkeyPatch) -> None:
    missing_evidence_key = json.dumps({})
    monkeypatch.setattr(
        llm_client, "_call_gemini", lambda prompt, system, response_schema: missing_evidence_key
    )

    client = EnvLLMClient(provider="gemini")

    with pytest.raises(LLMError, match="evidence"):
        client.call_llm("Extract evidence.", "system prompt", _EVIDENCE_SCHEMA)
