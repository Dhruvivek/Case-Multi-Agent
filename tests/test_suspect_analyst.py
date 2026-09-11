from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from agents.suspect_analyst import SuspectAnalyst
from case_file import CaseFile, EvidenceClassification, EvidenceItem


@dataclass
class StubLLM:
    response: dict
    prompts: list[str] = field(default_factory=list)

    def call_llm(self, prompt: str, system: str, response_schema: dict) -> dict:
        self.prompts.append(prompt)
        return self.response


def _case_file_with_evidence() -> CaseFile:
    case_file = CaseFile(mystery_text="A necklace vanished from the study overnight.")
    case_file.evidence = [
        EvidenceItem(
            id="E-01",
            statement="The study window lock was forced.",
            classification=EvidenceClassification.OBSERVED_FACT,
        ),
        EvidenceItem(
            id="E-02",
            statement="The housekeeper owed the victim money.",
            classification=EvidenceClassification.OBSERVED_FACT,
        ),
    ]
    return case_file


def test_suspect_analyst_builds_evidence_cited_profiles() -> None:
    llm = StubLLM(
        response={
            "suspects": [
                {
                    "name": "The Housekeeper",
                    "motive": [
                        {
                            "statement": "Owed the victim money.",
                            "status": "supported",
                            "evidence_ids": ["E-02"],
                        }
                    ],
                    "opportunity": [
                        {
                            "statement": "No record places her near the study.",
                            "status": "unknown",
                            "evidence_ids": [],
                        }
                    ],
                }
            ]
        }
    )
    case_file = _case_file_with_evidence()

    SuspectAnalyst(llm).run(case_file)

    assert len(case_file.suspect_profiles) == 1
    profile = case_file.suspect_profiles[0]
    assert profile.suspect == "The Housekeeper"
    assert profile.motive[0].status.value == "supported"
    assert profile.motive[0].evidence_ids == ("E-02",)
    assert profile.opportunity[0].status.value == "unknown"


def test_suspect_analyst_rejects_claims_citing_unknown_evidence_ids() -> None:
    llm = StubLLM(
        response={
            "suspects": [
                {
                    "name": "The Housekeeper",
                    "motive": [
                        {
                            "statement": "Owed the victim money.",
                            "status": "supported",
                            "evidence_ids": ["E-99"],
                        }
                    ],
                    "opportunity": [
                        {
                            "statement": "Unknown.",
                            "status": "unknown",
                            "evidence_ids": [],
                        }
                    ],
                }
            ]
        }
    )
    case_file = _case_file_with_evidence()

    with pytest.raises(ValueError, match="evidence"):
        SuspectAnalyst(llm).run(case_file)


def test_suspect_analyst_rejects_supported_claims_without_citations() -> None:
    llm = StubLLM(
        response={
            "suspects": [
                {
                    "name": "The Housekeeper",
                    "motive": [
                        {
                            "statement": "Owed the victim money.",
                            "status": "supported",
                            "evidence_ids": [],
                        }
                    ],
                    "opportunity": [
                        {
                            "statement": "Unknown.",
                            "status": "unknown",
                            "evidence_ids": [],
                        }
                    ],
                }
            ]
        }
    )
    case_file = _case_file_with_evidence()

    with pytest.raises(ValueError, match="evidence"):
        SuspectAnalyst(llm).run(case_file)


def _minimal_response() -> dict:
    return {
        "suspects": [
            {
                "name": "The Housekeeper",
                "motive": [
                    {"statement": "Owed the victim money.", "status": "unknown", "evidence_ids": []}
                ],
                "opportunity": [
                    {"statement": "Unknown.", "status": "unknown", "evidence_ids": []}
                ],
            }
        ]
    }


def test_suspect_analyst_includes_human_guidance_in_the_prompt_when_present() -> None:
    llm = StubLLM(response=_minimal_response())
    case_file = _case_file_with_evidence()
    case_file.human_notes = ["Evidence E-01 is unavailable; do not rely on it."]

    SuspectAnalyst(llm).run(case_file)

    assert "Evidence E-01 is unavailable; do not rely on it." in llm.prompts[0]


def test_suspect_analyst_prompt_omits_guidance_section_without_human_notes() -> None:
    llm = StubLLM(response=_minimal_response())
    case_file = _case_file_with_evidence()

    SuspectAnalyst(llm).run(case_file)

    assert "human reviewer" not in llm.prompts[0].lower()


def test_suspect_analyst_only_updates_its_own_case_file_section() -> None:
    llm = StubLLM(
        response={
            "suspects": [
                {
                    "name": "The Housekeeper",
                    "motive": [
                        {
                            "statement": "Owed the victim money.",
                            "status": "supported",
                            "evidence_ids": ["E-99"],
                        }
                    ],
                    "opportunity": [],
                }
            ]
        }
    )
    case_file = _case_file_with_evidence()
    original_evidence = list(case_file.evidence)

    with pytest.raises(ValueError):
        SuspectAnalyst(llm).run(case_file)

    assert case_file.evidence == original_evidence
    assert case_file.suspect_profiles == []
    assert case_file.timeline.events == []
    assert case_file.timeline.issues == []
