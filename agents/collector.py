"""Evidence Collector: turns raw mystery text into structured evidence."""

from __future__ import annotations

from case_file import CaseFile, EvidenceClassification, EvidenceItem
from llm_client import LLMClient

SYSTEM_PROMPT = (
    "You are the Evidence Collector on a fictional detective team. Read the "
    "supplied mystery text and extract every distinct clue as an evidence "
    "item. Assign each item a short unique ID (e.g. E-01), a concise "
    "statement, and a classification of either 'observed_fact' (something "
    "the text directly states happened) or 'inference' (something that can "
    "reasonably be inferred but was not stated outright). Do not invent "
    "details the text does not support."
)

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "statement": {"type": "string"},
                    "classification": {
                        "type": "string",
                        "enum": [item.value for item in EvidenceClassification],
                    },
                },
                "required": ["id", "statement", "classification"],
            },
        }
    },
    "required": ["evidence"],
}


class EvidenceCollector:
    """Reads `case_file.mystery_text` and writes `case_file.evidence`."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def run(self, case_file: CaseFile) -> CaseFile:
        raw_response = self._llm.call_llm(
            self._build_prompt(case_file.mystery_text), SYSTEM_PROMPT, RESPONSE_SCHEMA
        )
        case_file.evidence = _parse_evidence(raw_response)
        return case_file

    @staticmethod
    def _build_prompt(mystery_text: str) -> str:
        return f"Mystery text:\n{mystery_text}\n\nExtract the evidence as JSON."


def _parse_evidence(raw_response: dict) -> list[EvidenceItem]:
    seen_ids: set[str] = set()
    evidence: list[EvidenceItem] = []
    for raw_item in raw_response.get("evidence", []):
        item = _parse_evidence_item(raw_item)
        if item.id in seen_ids:
            raise ValueError(f"Evidence IDs must be unique; {item.id!r} was repeated.")
        seen_ids.add(item.id)
        evidence.append(item)
    return evidence


def _parse_evidence_item(raw_item: dict) -> EvidenceItem:
    raw_classification = raw_item.get("classification")
    try:
        classification = EvidenceClassification(raw_classification)
    except ValueError as error:
        allowed = ", ".join(item.value for item in EvidenceClassification)
        raise ValueError(
            f"Evidence classification must be one of {allowed}, got {raw_classification!r}."
        ) from error
    return EvidenceItem(
        id=raw_item["id"],
        statement=raw_item["statement"],
        classification=classification,
    )
