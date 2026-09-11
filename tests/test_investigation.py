from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import pytest

from orchestrator import InvestigationEventKind, stream_investigation


@dataclass
class StubLLM:
    responses: list[dict]
    prompts: list[str] = field(default_factory=list)
    response_lock: threading.Lock = field(default_factory=threading.Lock)

    def call_llm(self, prompt: str, system: str, response_schema: dict) -> dict:
        expected_key = (
            "evidence"
            if "Evidence Collector" in system
            else "events"
            if "Timeline Reconciler" in system
            else "suspects"
        )
        with self.response_lock:
            self.prompts.append(prompt)
            for index, response in enumerate(self.responses):
                if expected_key in response:
                    return self.responses.pop(index)
        raise AssertionError(f"No {expected_key!r} response configured for the LLM stub.")


@dataclass
class ConcurrentSpecialistLLM:
    first_specialist: str
    rendezvous: threading.Barrier = field(default_factory=lambda: threading.Barrier(2))
    first_finished: threading.Event = field(default_factory=threading.Event)
    specialist_prompts: list[str] = field(default_factory=list)

    def call_llm(self, prompt: str, system: str, response_schema: dict) -> dict:
        if "Evidence Collector" in system:
            return {
                "evidence": [
                    {
                        "id": "W-3",
                        "statement": "The workshop was locked at sunset.",
                        "classification": "observed_fact",
                    }
                ]
            }

        self.specialist_prompts.append(prompt)
        self.rendezvous.wait(timeout=2)
        specialist = "timeline" if "Timeline Reconciler" in system else "suspect"
        if specialist == self.first_specialist:
            self.first_finished.set()
        else:
            assert self.first_finished.wait(timeout=2)
            time.sleep(0.01)

        if specialist == "timeline":
            return {
                "events": [
                    {
                        "statement": "The workshop was locked.",
                        "time": "sunset",
                        "order": 1,
                        "status": "supported",
                        "evidence_ids": ["W-3"],
                    }
                ],
                "issues": [],
            }

        return {
            "suspects": [
                {
                    "name": "The Sculptor",
                    "motive": [
                        {
                            "statement": "No motive is established.",
                            "status": "unknown",
                            "evidence_ids": [],
                        }
                    ],
                    "opportunity": [
                        {
                            "statement": "The available evidence does not place them inside.",
                            "status": "unknown",
                            "evidence_ids": [],
                        }
                    ],
                }
            ]
        }


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
            {
                "events": [
                    {
                        "statement": "The gallery window was open.",
                        "time": "midnight",
                        "order": 1,
                        "status": "supported",
                        "evidence_ids": ["E-01"],
                    }
                ],
                "issues": [
                    {
                        "kind": "gap",
                        "statement": "The entry route is uncertain.",
                        "evidence_ids": ["E-01", "E-02"],
                    }
                ],
            },
        ]
    )

    events = list(
        stream_investigation(
            "At midnight, the gallery window stood open.", llm
        )
    )

    assert [event.kind for event in events[:4]] == [
        InvestigationEventKind.EVIDENCE_COLLECTION_STARTED,
        InvestigationEventKind.EVIDENCE_COLLECTION_COMPLETED,
        InvestigationEventKind.SUSPECT_ANALYSIS_STARTED,
        InvestigationEventKind.TIMELINE_RECONCILIATION_STARTED,
    ]
    assert {event.kind for event in events[4:]} == {
        InvestigationEventKind.SUSPECT_ANALYSIS_COMPLETED,
        InvestigationEventKind.TIMELINE_RECONCILIATION_COMPLETED,
    }
    evidence_completed_case_file = events[1].case_file
    assert (
        evidence_completed_case_file.mystery_text
        == "At midnight, the gallery window stood open."
    )
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
    assert completed_case_file.timeline.events[0].time == "midnight"


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


@pytest.mark.parametrize(
    ("first_specialist", "first_completed_kind"),
    [
        ("timeline", InvestigationEventKind.TIMELINE_RECONCILIATION_COMPLETED),
        ("suspect", InvestigationEventKind.SUSPECT_ANALYSIS_COMPLETED),
    ],
)
def test_specialists_run_concurrently_and_stream_in_completion_order(
    first_specialist: str,
    first_completed_kind: InvestigationEventKind,
) -> None:
    llm = ConcurrentSpecialistLLM(first_specialist=first_specialist)

    events = list(stream_investigation("A wooden figure vanished from a workshop.", llm))

    assert [event.kind for event in events[:4]] == [
        InvestigationEventKind.EVIDENCE_COLLECTION_STARTED,
        InvestigationEventKind.EVIDENCE_COLLECTION_COMPLETED,
        InvestigationEventKind.SUSPECT_ANALYSIS_STARTED,
        InvestigationEventKind.TIMELINE_RECONCILIATION_STARTED,
    ]
    assert events[4].kind is first_completed_kind
    assert {event.kind for event in events[4:]} == {
        InvestigationEventKind.TIMELINE_RECONCILIATION_COMPLETED,
        InvestigationEventKind.SUSPECT_ANALYSIS_COMPLETED,
    }
    assert len(llm.specialist_prompts) == 2
    assert all("W-3" in prompt for prompt in llm.specialist_prompts)
    completed_case_file = events[-1].case_file
    assert completed_case_file.suspect_profiles[0].suspect == "The Sculptor"
    assert completed_case_file.timeline.events[0].evidence_ids == ("W-3",)
