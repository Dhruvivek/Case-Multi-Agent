from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import pytest

from case_file import Specialist, SkepticReviewOutcome
from orchestrator import InvestigationEventKind, stream_investigation


def _expected_response_key(system: str) -> str:
    if "Evidence Collector" in system:
        return "evidence"
    if "Lead Detective" in system:
        return "conclusions"
    if "Skeptic" in system:
        return "findings"
    if "Timeline Reconciler" in system:
        return "events"
    return "suspects"


def _verdict_response(suspect: str = "The Housekeeper", evidence_id: str = "K-1") -> dict:
    return {
        "conclusions": [
            {
                "rank": 1,
                "suspect": suspect,
                "explanation": "The strongest evidence-backed explanation.",
                "evidence_ids": [evidence_id],
            }
        ],
        "confidence": 60,
        "limitations": ["Some claims in the case file remain unresolved."],
    }


@dataclass
class StubLLM:
    responses: list[dict]
    prompts: list[str] = field(default_factory=list)
    response_lock: threading.Lock = field(default_factory=threading.Lock)

    def call_llm(self, prompt: str, system: str, response_schema: dict) -> dict:
        expected_key = _expected_response_key(system)
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
        if "Lead Detective" in system:
            return _verdict_response(suspect="The Sculptor", evidence_id="W-3")
        if "Skeptic" in system:
            return {"findings": []}

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
            {"findings": []},
            _verdict_response(suspect="The Night Guard", evidence_id="E-01"),
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
    assert {event.kind for event in events[4:6]} == {
        InvestigationEventKind.SUSPECT_ANALYSIS_COMPLETED,
        InvestigationEventKind.TIMELINE_RECONCILIATION_COMPLETED,
    }
    assert [event.kind for event in events[6:]] == [
        InvestigationEventKind.SKEPTIC_REVIEW_STARTED,
        InvestigationEventKind.SKEPTIC_REVIEW_APPROVED,
        InvestigationEventKind.LEAD_DETECTIVE_STARTED,
        InvestigationEventKind.LEAD_DETECTIVE_COMPLETED,
    ]
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
    assert len(completed_case_file.skeptic_reviews) == 1
    assert completed_case_file.skeptic_reviews[0].outcome is SkepticReviewOutcome.APPROVED
    assert completed_case_file.verdict is not None
    assert completed_case_file.verdict.conclusions[0].suspect == "The Night Guard"
    assert completed_case_file.verdict.conclusions[0].evidence_ids == ("E-01",)


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
    assert {event.kind for event in events[4:6]} == {
        InvestigationEventKind.TIMELINE_RECONCILIATION_COMPLETED,
        InvestigationEventKind.SUSPECT_ANALYSIS_COMPLETED,
    }
    assert [event.kind for event in events[6:]] == [
        InvestigationEventKind.SKEPTIC_REVIEW_STARTED,
        InvestigationEventKind.SKEPTIC_REVIEW_APPROVED,
        InvestigationEventKind.LEAD_DETECTIVE_STARTED,
        InvestigationEventKind.LEAD_DETECTIVE_COMPLETED,
    ]
    assert len(llm.specialist_prompts) == 2
    assert all("W-3" in prompt for prompt in llm.specialist_prompts)
    completed_case_file = events[-1].case_file
    assert completed_case_file.suspect_profiles[0].suspect == "The Sculptor"
    assert completed_case_file.timeline.events[0].evidence_ids == ("W-3",)


def _evidence_response() -> dict:
    return {
        "evidence": [
            {
                "id": "K-1",
                "statement": "The housekeeper's key card unlocked the study at midnight.",
                "classification": "observed_fact",
            }
        ]
    }


def _suspects_response(claim_statement: str) -> dict:
    return {
        "suspects": [
            {
                "name": "The Housekeeper",
                "motive": [
                    {"statement": "Owed the victim money.", "status": "unknown", "evidence_ids": []}
                ],
                "opportunity": [
                    {"statement": claim_statement, "status": "supported", "evidence_ids": ["K-1"]}
                ],
            }
        ]
    }


def _events_response(event_statement: str) -> dict:
    return {
        "events": [
            {
                "statement": event_statement,
                "time": "midnight",
                "order": 1,
                "status": "supported",
                "evidence_ids": ["K-1"],
            }
        ],
        "issues": [],
    }


def _findings_response(claim: str, specialist: str = "suspect_analyst") -> dict:
    return {
        "findings": [
            {
                "specialist": specialist,
                "claim": claim,
                "kind": "unsupported_reasoning",
                "explanation": "Card use establishes use of the card, not who held it.",
            }
        ]
    }


def test_skeptic_revision_reruns_only_the_flagged_specialist() -> None:
    original_claim = "The key card used at midnight proves she was in the study."
    revised_claim = "The key card was used at midnight; who held it is unknown."
    llm = StubLLM(
        responses=[
            _evidence_response(),
            _suspects_response(original_claim),
            _events_response("The study was unlocked."),
            _findings_response(original_claim),
            _suspects_response(revised_claim),
            {"findings": []},
            _verdict_response(),
        ]
    )

    events = list(stream_investigation("A necklace vanished from the study.", llm))

    assert [event.kind for event in events[6:]] == [
        InvestigationEventKind.SKEPTIC_REVIEW_STARTED,
        InvestigationEventKind.SKEPTIC_REVIEW_REVISION_REQUESTED,
        InvestigationEventKind.SPECIALIST_REVISION_STARTED,
        InvestigationEventKind.SPECIALIST_REVISION_COMPLETED,
        InvestigationEventKind.SKEPTIC_REVIEW_STARTED,
        InvestigationEventKind.SKEPTIC_REVIEW_APPROVED,
        InvestigationEventKind.LEAD_DETECTIVE_STARTED,
        InvestigationEventKind.LEAD_DETECTIVE_COMPLETED,
    ]
    revision_started = events[8]
    assert revision_started.specialist is Specialist.SUSPECT_ANALYST

    final_case_file = events[-1].case_file
    assert final_case_file.suspect_profiles[0].opportunity[0].statement == revised_claim
    assert final_case_file.timeline.events[0].statement == "The study was unlocked."
    assert final_case_file.revised_specialists == {Specialist.SUSPECT_ANALYST}
    assert len(final_case_file.skeptic_reviews) == 2
    assert final_case_file.skeptic_reviews[0].outcome is SkepticReviewOutcome.REVISION_REQUESTED
    assert final_case_file.skeptic_reviews[1].outcome is SkepticReviewOutcome.APPROVED
    assert final_case_file.verdict is not None


def test_skeptic_revision_reruns_multiple_flagged_specialists() -> None:
    original_suspect_claim = "The key card used at midnight proves she was in the study."
    original_event = "The study was unlocked."
    revised_suspect_claim = "The key card was used at midnight; who held it is unknown."
    revised_event = "The study was unlocked, though by whom is unconfirmed."
    llm = StubLLM(
        responses=[
            _evidence_response(),
            _suspects_response(original_suspect_claim),
            _events_response(original_event),
            {
                "findings": [
                    {
                        "specialist": "suspect_analyst",
                        "claim": original_suspect_claim,
                        "kind": "unsupported_reasoning",
                        "explanation": "Card use establishes use of the card, not who held it.",
                    },
                    {
                        "specialist": "timeline_reconciler",
                        "claim": original_event,
                        "kind": "unsupported_reasoning",
                        "explanation": "The evidence does not establish who unlocked it.",
                    },
                ]
            },
            _suspects_response(revised_suspect_claim),
            _events_response(revised_event),
            {"findings": []},
            _verdict_response(),
        ]
    )

    events = list(stream_investigation("A necklace vanished from the study.", llm))

    revision_events = [
        event
        for event in events
        if event.kind
        in (
            InvestigationEventKind.SPECIALIST_REVISION_STARTED,
            InvestigationEventKind.SPECIALIST_REVISION_COMPLETED,
        )
    ]
    assert [event.specialist for event in revision_events] == [
        Specialist.SUSPECT_ANALYST,
        Specialist.SUSPECT_ANALYST,
        Specialist.TIMELINE_RECONCILER,
        Specialist.TIMELINE_RECONCILER,
    ]

    final_case_file = events[-1].case_file
    assert final_case_file.revised_specialists == {
        Specialist.SUSPECT_ANALYST,
        Specialist.TIMELINE_RECONCILER,
    }
    assert final_case_file.suspect_profiles[0].opportunity[0].statement == revised_suspect_claim
    assert final_case_file.timeline.events[0].statement == revised_event
    assert final_case_file.skeptic_reviews[-1].outcome is SkepticReviewOutcome.APPROVED


def test_skeptic_review_becomes_exhausted_after_one_revision_round() -> None:
    original_claim = "The key card used at midnight proves she was in the study."
    revised_claim = "The key card was used at midnight and still names her as present."
    llm = StubLLM(
        responses=[
            _evidence_response(),
            _suspects_response(original_claim),
            _events_response("The study was unlocked."),
            _findings_response(original_claim),
            _suspects_response(revised_claim),
            _findings_response(revised_claim),
            {
                "conclusions": [
                    {
                        "rank": 1,
                        "suspect": "The Housekeeper",
                        "explanation": "The strongest, though not fully certain, explanation.",
                        "evidence_ids": ["K-1"],
                    }
                ],
                "confidence": 40,
                "limitations": [
                    "Who physically held the key card at midnight remains unresolved."
                ],
            },
        ]
    )

    events = list(stream_investigation("A necklace vanished from the study.", llm))

    assert [event.kind for event in events[6:]] == [
        InvestigationEventKind.SKEPTIC_REVIEW_STARTED,
        InvestigationEventKind.SKEPTIC_REVIEW_REVISION_REQUESTED,
        InvestigationEventKind.SPECIALIST_REVISION_STARTED,
        InvestigationEventKind.SPECIALIST_REVISION_COMPLETED,
        InvestigationEventKind.SKEPTIC_REVIEW_STARTED,
        InvestigationEventKind.SKEPTIC_REVIEW_EXHAUSTED,
        InvestigationEventKind.LEAD_DETECTIVE_STARTED,
        InvestigationEventKind.LEAD_DETECTIVE_COMPLETED,
    ]

    final_case_file = events[-1].case_file
    assert final_case_file.revised_specialists == {Specialist.SUSPECT_ANALYST}
    assert len(final_case_file.skeptic_reviews) == 2
    final_review = final_case_file.skeptic_reviews[-1]
    assert final_review.outcome is SkepticReviewOutcome.EXHAUSTED
    assert len(final_review.findings) == 1
    assert final_review.findings[0].claim == revised_claim
    assert final_case_file.verdict is not None
    assert final_case_file.verdict.limitations != ()
