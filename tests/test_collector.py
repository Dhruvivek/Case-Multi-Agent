from __future__ import annotations

from dataclasses import dataclass

import pytest

from agents.collector import EvidenceCollector
from case_file import CaseFile


@dataclass
class InvalidEvidenceLLM:
    def call_llm(self, prompt: str, system: str, response_schema: dict) -> dict:
        return {
            "evidence": [
                {
                    "id": "E-01",
                    "statement": "The butler left at 9 PM.",
                    "classification": "unsupported_label",
                }
            ]
        }


def test_collector_only_updates_the_evidence_section_of_a_case_file() -> None:
    collector = EvidenceCollector(InvalidEvidenceLLM())
    case_file = CaseFile(mystery_text="The butler left at 9 PM.")

    with pytest.raises(ValueError, match="classification"):
        collector.run(case_file)

    assert case_file.evidence == []
    assert case_file.suspect_profiles == []
    assert case_file.timeline_notes == []
