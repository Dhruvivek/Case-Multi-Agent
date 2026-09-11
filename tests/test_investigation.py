from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import pytest

from case_file import (
    CaseFile,
    Conclusion,
    EvidenceClassification,
    EvidenceItem,
    Specialist,
    SkepticReview,
    SkepticReviewOutcome,
    Verdict,
    VerdictReviewError,
    VerdictReviewStatus,
)
from llm_client import LLMError
from orchestrator import (
    EMPTY_GUIDANCE_MESSAGE,
    InvestigationEventKind,
    stream_investigation,
    stream_reinvestigation,
)


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


def _case_file_awaiting_reinvestigation() -> CaseFile:
    case_file = CaseFile(mystery_text="A necklace vanished from the study.")
    case_file.evidence = [
        EvidenceItem(
            id="K-1",
            statement="The housekeeper's key card unlocked the study at midnight.",
            classification=EvidenceClassification.OBSERVED_FACT,
        )
    ]
    case_file.verdict = Verdict(
        conclusions=(
            Conclusion(
                rank=1,
                suspect="The Housekeeper",
                explanation="The key card places her at the scene.",
                evidence_ids=("K-1",),
            ),
        ),
        confidence=60,
    )
    return case_file


def test_reinvestigation_restarts_from_suspect_analysis_without_the_evidence_collector() -> None:
    case_file = _case_file_awaiting_reinvestigation()
    llm = StubLLM(
        responses=[
            _suspects_response("The key card was used at midnight; who held it is unknown."),
            _events_response("The study was unlocked."),
            {"findings": []},
            _verdict_response(),
        ]
    )

    events = list(stream_reinvestigation(case_file, "Evidence K-1 is unavailable.", llm))

    assert [event.kind for event in events] == [
        InvestigationEventKind.REINVESTIGATION_REQUESTED,
        InvestigationEventKind.SUSPECT_ANALYSIS_STARTED,
        InvestigationEventKind.TIMELINE_RECONCILIATION_STARTED,
        InvestigationEventKind.SUSPECT_ANALYSIS_COMPLETED,
        InvestigationEventKind.TIMELINE_RECONCILIATION_COMPLETED,
        InvestigationEventKind.SKEPTIC_REVIEW_STARTED,
        InvestigationEventKind.SKEPTIC_REVIEW_APPROVED,
        InvestigationEventKind.LEAD_DETECTIVE_STARTED,
        InvestigationEventKind.LEAD_DETECTIVE_COMPLETED,
    ]
    assert all("Evidence Collector" not in prompt for prompt in llm.prompts)


def test_reinvestigation_reuses_the_original_mystery_and_evidence() -> None:
    case_file = _case_file_awaiting_reinvestigation()
    original_mystery_text = case_file.mystery_text
    original_evidence = list(case_file.evidence)
    llm = StubLLM(
        responses=[
            _suspects_response("The key card was used at midnight; who held it is unknown."),
            _events_response("The study was unlocked."),
            {"findings": []},
            _verdict_response(),
        ]
    )

    events = list(stream_reinvestigation(case_file, "Evidence K-1 is unavailable.", llm))

    final_case_file = events[-1].case_file
    assert final_case_file.mystery_text == original_mystery_text
    assert final_case_file.evidence == original_evidence


def test_reinvestigation_propagates_the_human_note_to_the_specialist_prompts() -> None:
    case_file = _case_file_awaiting_reinvestigation()
    llm = StubLLM(
        responses=[
            _suspects_response("The key card was used at midnight; who held it is unknown."),
            _events_response("The study was unlocked."),
            {"findings": []},
            _verdict_response(),
        ]
    )
    note = "Evidence K-1 is unavailable; do not rely on it."

    list(stream_reinvestigation(case_file, note, llm))

    assert any(note in prompt for prompt in llm.prompts)


def test_reinvestigation_resets_review_state_and_produces_a_fresh_verdict() -> None:
    case_file = _case_file_awaiting_reinvestigation()
    case_file.skeptic_reviews = [SkepticReview(outcome=SkepticReviewOutcome.APPROVED)]
    case_file.revised_specialists = {Specialist.SUSPECT_ANALYST}
    original_verdict = case_file.verdict
    llm = StubLLM(
        responses=[
            _suspects_response("The key card was used at midnight; who held it is unknown."),
            _events_response("The study was unlocked."),
            {"findings": []},
            _verdict_response(suspect="The Groundskeeper", evidence_id="K-1"),
        ]
    )

    events: list = []
    reinvestigation_review_status: VerdictReviewStatus | None = None
    for event in stream_reinvestigation(case_file, "Evidence K-1 is unavailable.", llm):
        events.append(event)
        if event.kind is InvestigationEventKind.REINVESTIGATION_REQUESTED:
            # Captured mid-stream: `case_file` is mutated and re-yielded in
            # place, so this must be read before later stages overwrite it.
            reinvestigation_review_status = event.case_file.verdict.review_status

    reinvestigation_event = events[0]
    assert reinvestigation_event.kind is InvestigationEventKind.REINVESTIGATION_REQUESTED
    assert reinvestigation_event.message == "Evidence K-1 is unavailable."
    assert reinvestigation_review_status is VerdictReviewStatus.REINVESTIGATION_REQUESTED

    final_case_file = events[-1].case_file
    assert final_case_file.revised_specialists == set()
    assert len(final_case_file.skeptic_reviews) == 1
    assert final_case_file.skeptic_reviews[0].outcome is SkepticReviewOutcome.APPROVED
    assert final_case_file.human_notes == ["Evidence K-1 is unavailable."]
    assert final_case_file.verdict is not original_verdict
    assert final_case_file.verdict.conclusions[0].suspect == "The Groundskeeper"
    assert final_case_file.verdict.review_status is VerdictReviewStatus.AWAITING_REVIEW


def test_reinvestigation_on_an_already_decided_verdict_raises_predictable_error() -> None:
    case_file = _case_file_awaiting_reinvestigation()
    case_file.accept_verdict()
    llm = StubLLM(responses=[])

    with pytest.raises(VerdictReviewError):
        list(stream_reinvestigation(case_file, "Evidence K-1 is unavailable.", llm))

    assert llm.prompts == []


def test_empty_guidance_note_produces_validation_event_without_calling_the_llm() -> None:
    case_file = _case_file_awaiting_reinvestigation()
    llm = StubLLM(responses=[])

    events = list(stream_reinvestigation(case_file, "   ", llm))

    assert [event.kind for event in events] == [InvestigationEventKind.VALIDATION_ERROR]
    assert events[0].message == EMPTY_GUIDANCE_MESSAGE
    assert events[0].case_file is case_file
    assert llm.prompts == []
    assert case_file.human_notes == []
    assert case_file.verdict.review_status is VerdictReviewStatus.AWAITING_REVIEW


def _canned_response(expected_key: str, evidence_id: str) -> dict:
    """A minimal valid response for whichever agent a stage represents."""
    if expected_key == "evidence":
        return {
            "evidence": [
                {
                    "id": evidence_id,
                    "statement": "The window was open at midnight.",
                    "classification": "observed_fact",
                }
            ]
        }
    if expected_key == "suspects":
        return {
            "suspects": [
                {
                    "name": "The Butler",
                    "motive": [
                        {
                            "statement": "Owed the victim money.",
                            "status": "supported",
                            "evidence_ids": [evidence_id],
                        }
                    ],
                    "opportunity": [
                        {
                            "statement": "Was alone in the study.",
                            "status": "supported",
                            "evidence_ids": [evidence_id],
                        }
                    ],
                }
            ]
        }
    if expected_key == "events":
        return {
            "events": [
                {
                    "statement": "The window was opened.",
                    "time": "midnight",
                    "order": 1,
                    "status": "supported",
                    "evidence_ids": [evidence_id],
                }
            ],
            "issues": [],
        }
    if expected_key == "findings":
        return {"findings": []}
    if expected_key == "conclusions":
        return _verdict_response(suspect="The Butler", evidence_id=evidence_id)
    raise AssertionError(f"No canned response for {expected_key!r}.")


@dataclass
class FailingAtLLM:
    """Raises `LLMError` for the agent(s) named by `fail_when`, else succeeds.

    `fail_when` is a system-prompt substring, or a tuple of them to fail more
    than one agent (e.g. both specialists that run concurrently). Models what
    `EnvLLMClient` raises once its own internal reformat retry has also
    failed: a single `LLMError` from `call_llm`. Every other agent gets back
    a minimal valid response so a halt at `fail_when` can be distinguished
    from an unrelated failure earlier in the pipeline. `evidence_id` must
    match any evidence already on the case file (e.g. for a re-investigation,
    which reuses previously collected evidence).
    """

    fail_when: str | tuple[str, ...]
    evidence_id: str = "E-01"
    calls: list[str] = field(default_factory=list)

    def call_llm(self, prompt: str, system: str, response_schema: dict) -> dict:
        self.calls.append(system)
        markers = (self.fail_when,) if isinstance(self.fail_when, str) else self.fail_when
        if any(marker in system for marker in markers):
            raise LLMError(
                "LLM response did not match the required schema: "
                "response is missing required field 'placeholder'."
            )
        return _canned_response(_expected_response_key(system), self.evidence_id)


def test_evidence_collector_boundary_failure_halts_before_any_other_agent_runs() -> None:
    llm = FailingAtLLM(fail_when="Evidence Collector")

    events = list(stream_investigation("A quiet manor at midnight.", llm))

    assert [event.kind for event in events] == [
        InvestigationEventKind.EVIDENCE_COLLECTION_STARTED,
        InvestigationEventKind.STEP_FAILED,
    ]
    failure_event = events[-1]
    assert failure_event.agent_name == "Evidence Collector"
    assert "schema" in failure_event.message.lower()
    assert "Traceback" not in failure_event.message
    assert failure_event.case_file.evidence == []
    assert len(llm.calls) == 1


def test_suspect_analyst_boundary_failure_halts_skeptic_and_lead_detective() -> None:
    llm = FailingAtLLM(fail_when="Suspect Analyst")

    events = list(stream_investigation("A quiet manor at midnight.", llm))

    assert events[-1].kind is InvestigationEventKind.STEP_FAILED
    assert events[-1].agent_name == "Suspect Analyst"
    assert InvestigationEventKind.SKEPTIC_REVIEW_STARTED not in [event.kind for event in events]
    assert InvestigationEventKind.LEAD_DETECTIVE_STARTED not in [event.kind for event in events]

    final_case_file = events[-1].case_file
    assert final_case_file.suspect_profiles == []
    assert final_case_file.skeptic_reviews == []
    assert final_case_file.verdict is None


def test_both_concurrent_specialists_failing_still_halts_before_skeptic_and_lead_detective() -> None:
    """A boundary failure in either concurrently-run specialist must still halt the pipeline.

    Which of the two is reported is a race (whichever `Future` completes
    first), but the halt itself, and that neither the Skeptic nor the Lead
    Detective ever run, must hold regardless of which one that is.
    """
    llm = FailingAtLLM(fail_when=("Suspect Analyst", "Timeline Reconciler"))

    events = list(stream_investigation("A quiet manor at midnight.", llm))

    assert events[-1].kind is InvestigationEventKind.STEP_FAILED
    assert events[-1].agent_name in {"Suspect Analyst", "Timeline Reconciler"}
    assert InvestigationEventKind.SKEPTIC_REVIEW_STARTED not in [event.kind for event in events]
    assert InvestigationEventKind.LEAD_DETECTIVE_STARTED not in [event.kind for event in events]

    final_case_file = events[-1].case_file
    assert final_case_file.skeptic_reviews == []
    assert final_case_file.verdict is None


def test_skeptic_boundary_failure_halts_before_the_lead_detective_runs() -> None:
    llm = FailingAtLLM(fail_when="Skeptic")

    events = list(stream_investigation("A quiet manor at midnight.", llm))

    assert events[-1].kind is InvestigationEventKind.STEP_FAILED
    assert events[-1].agent_name == "Skeptic"
    assert InvestigationEventKind.LEAD_DETECTIVE_STARTED not in [event.kind for event in events]
    assert events[-1].case_file.verdict is None


def test_lead_detective_boundary_failure_leaves_case_file_without_a_verdict() -> None:
    llm = FailingAtLLM(fail_when="Lead Detective")

    events = list(stream_investigation("A quiet manor at midnight.", llm))

    assert [event.kind for event in events[-2:]] == [
        InvestigationEventKind.LEAD_DETECTIVE_STARTED,
        InvestigationEventKind.STEP_FAILED,
    ]
    assert events[-1].agent_name == "Lead Detective"
    assert events[-1].case_file.verdict is None


def test_reinvestigation_boundary_failure_reports_the_failing_specialist() -> None:
    case_file = _case_file_awaiting_reinvestigation()
    llm = FailingAtLLM(fail_when="Timeline Reconciler", evidence_id="K-1")

    events = list(stream_reinvestigation(case_file, "Evidence K-1 is unavailable.", llm))

    assert events[-1].kind is InvestigationEventKind.STEP_FAILED
    assert events[-1].agent_name == "Timeline Reconciler"
    assert InvestigationEventKind.LEAD_DETECTIVE_STARTED not in [event.kind for event in events]
