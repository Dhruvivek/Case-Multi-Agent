from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from orchestrator import InvestigationEventKind, stream_investigation


@dataclass
class StubLLM:
    responses: list[dict]
    prompts: list[str] = field(default_factory=list)

    def call_llm(self, prompt: str, system: str, response_schema: dict) -> dict:
        self.prompts.append(prompt)
        return self.responses.pop(0)


def test_investigation_streams_collected_evidence_into_its_case_file() -> None:
    llm = StubLLM(
        responses=[
            {
                "evidence": [
                    {
                        "id": "E-01",
                        "statement": "The gallery window was open at midnight.",
                        "classification": "observed_fact",
                    },
                    {
                        "id": "E-02",
                        "statement": "The thief may have entered through the window.",
                        "classification": "inference",
                    },
                ]
            },
            {
                "suspects": [
                    {
                        "name": "The Night Guard",
                        "motive": [
                            {
                                "statement": "No clear reason to steal is on record.",
                                "status": "unknown",
                                "evidence_ids": [],
                            }
                        ],
                        "opportunity": [
                            {
                                "statement": "Was on duty when the window was found open.",
                                "status": "supported",
                                "evidence_ids": ["E-01"],
                            }
                        ],
                    }
                ]
            },
        ]
    )

    events = list(
        stream_investigation(
            "At midnight, the gallery window stood open.", llm
        )
    )

    assert [event.kind for event in events] == [
        InvestigationEventKind.EVIDENCE_COLLECTION_STARTED,
        InvestigationEventKind.EVIDENCE_COLLECTION_COMPLETED,
        InvestigationEventKind.SUSPECT_ANALYSIS_STARTED,
        InvestigationEventKind.SUSPECT_ANALYSIS_COMPLETED,
    ]
    evidence_completed_case_file = events[1].case_file
    assert evidence_completed_case_file.mystery_text == "At midnight, the gallery window stood open."
    assert [
        (item.id, item.classification.value) for item in evidence_completed_case_file.evidence
    ] == [
        ("E-01", "observed_fact"),
        ("E-02", "inference"),
    ]

    completed_case_file = events[-1].case_file
    assert [profile.suspect for profile in completed_case_file.suspect_profiles] == [
        "The Night Guard"
    ]
    profile = completed_case_file.suspect_profiles[0]
    assert profile.motive[0].status.value == "unknown"
    assert profile.opportunity[0].evidence_ids == ("E-01",)


def test_empty_mystery_produces_validation_event_without_calling_the_llm() -> None:
    llm = StubLLM(responses=[])

    events = list(stream_investigation("   ", llm))

    assert [event.kind for event in events] == [InvestigationEventKind.VALIDATION_ERROR]
    assert events[0].message == "Enter a fictional mystery before starting an investigation."
    assert llm.prompts == []


def test_collector_rejects_duplicate_evidence_ids() -> None:
    llm = StubLLM(
        responses=[
            {
                "evidence": [
                    {
                        "id": "E-01",
                        "statement": "A note was found.",
                        "classification": "observed_fact",
                    },
                    {
                        "id": "E-01",
                        "statement": "The note names a suspect.",
                        "classification": "inference",
                    },
                ]
            }
        ]
    )

    with pytest.raises(ValueError, match="unique"):
        list(stream_investigation("A note was found.", llm))
