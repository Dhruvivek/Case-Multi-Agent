from __future__ import annotations

import pytest

import llm_client
from llm_client import EnvLLMClient, LLMError


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
