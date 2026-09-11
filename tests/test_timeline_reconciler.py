from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from agents.timeline_reconciler import TimelineReconciler
from case_file import (
    CaseFile,
    Claim,
    ClaimStatus,
    EvidenceClassification,
    EvidenceItem,
    SuspectProfile,
)


@dataclass
class StubLLM:
    response: dict
    prompts: list[str] = field(default_factory=list)

    def call_llm(self, prompt: str, system: str, response_schema: dict) -> dict:
        self.prompts.append(prompt)
        return self.response


def _case_file_with_evidence() -> CaseFile:
    case_file = CaseFile(mystery_text="A signed manuscript vanished during a reception.")
    case_file.evidence = [
        EvidenceItem(
            id="R-7",
            statement="The archive was locked at 9:00 PM.",
            classification=EvidenceClassification.OBSERVED_FACT,
        ),
        EvidenceItem(
            id="R-9",
            statement="The manuscript was missing at 9:20 PM.",
            classification=EvidenceClassification.OBSERVED_FACT,
        ),
        EvidenceItem(
            id="R-12",
            statement="A volunteer said she stayed outside throughout the reception.",
            classification=EvidenceClassification.OBSERVED_FACT,
        ),
        EvidenceItem(
            id="R-14",
            statement="The volunteer's badge opened the archive at 9:10 PM.",
            classification=EvidenceClassification.OBSERVED_FACT,
        ),
    ]
    return case_file


def _minimal_response() -> dict:
    return {
        "events": [
            {
                "statement": "The archive was locked.",
                "time": "9:00 PM",
                "order": 1,
                "status": "supported",
                "evidence_ids": ["R-7"],
            }
        ],
        "issues": [],
    }


def test_timeline_reconciler_includes_human_guidance_in_the_prompt_when_present() -> None:
    llm = StubLLM(response=_minimal_response())
    case_file = _case_file_with_evidence()
    case_file.human_notes = ["Evidence R-14 is unavailable; do not rely on it."]

    TimelineReconciler(llm).run(case_file)

    assert "Evidence R-14 is unavailable; do not rely on it." in llm.prompts[0]


def test_timeline_reconciler_prompt_omits_guidance_section_without_human_notes() -> None:
    llm = StubLLM(response=_minimal_response())
    case_file = _case_file_with_evidence()

    TimelineReconciler(llm).run(case_file)

    assert "human reviewer" not in llm.prompts[0].lower()


def test_timeline_reconciler_builds_events_and_evidence_cited_issues() -> None:
    llm = StubLLM(
        response={
            "events": [
                {
                    "statement": "The archive was locked.",
                    "time": "9:00 PM",
                    "order": 1,
                    "status": "supported",
                    "evidence_ids": ["R-7"],
                },
                {
                    "statement": "The manuscript was removed sometime before discovery.",
                    "time": None,
                    "order": None,
                    "status": "unknown",
                    "evidence_ids": ["R-7", "R-9"],
                },
            ],
            "issues": [
                {
                    "kind": "gap",
                    "statement": "The exact removal time is unknown.",
                    "evidence_ids": ["R-7", "R-9"],
                },
                {
                    "kind": "contradiction",
                    "statement": "The volunteer's statement conflicts with badge use.",
                    "evidence_ids": ["R-12", "R-14"],
                },
            ],
        }
    )
    case_file = _case_file_with_evidence()

    TimelineReconciler(llm).run(case_file)

    assert [event.time for event in case_file.timeline.events] == ["9:00 PM", None]
    assert case_file.timeline.events[0].evidence_ids == ("R-7",)
    assert case_file.timeline.events[1].status.value == "unknown"
    assert [issue.kind.value for issue in case_file.timeline.issues] == [
        "gap",
        "contradiction",
    ]
    assert case_file.timeline.issues[1].evidence_ids == ("R-12", "R-14")


def test_timeline_reconciler_rejects_an_invented_time_for_an_uncertain_event() -> None:
    llm = StubLLM(
        response={
            "events": [
                {
                    "statement": "The manuscript was removed.",
                    "time": "9:10 PM",
                    "order": 1,
                    "status": "unknown",
                    "evidence_ids": ["R-7", "R-9"],
                }
            ],
            "issues": [],
        }
    )
    case_file = _case_file_with_evidence()

    with pytest.raises(ValueError, match="invented time"):
        TimelineReconciler(llm).run(case_file)

    assert case_file.timeline.events == []


def test_timeline_reconciler_normalizes_supported_order_and_leaves_unknown_unordered() -> None:
    llm = StubLLM(
        response={
            "events": [
                {
                    "statement": "The manuscript was found missing.",
                    "time": "9:20 PM",
                    "order": 2,
                    "status": "supported",
                    "evidence_ids": ["R-9"],
                },
                {
                    "statement": "The exact removal event cannot be ordered.",
                    "time": None,
                    "order": None,
                    "status": "unknown",
                    "evidence_ids": ["R-7", "R-9"],
                },
                {
                    "statement": "The archive was locked.",
                    "time": "9:00 PM",
                    "order": 1,
                    "status": "supported",
                    "evidence_ids": ["R-7"],
                },
            ],
            "issues": [],
        }
    )
    case_file = _case_file_with_evidence()

    TimelineReconciler(llm).run(case_file)

    assert [event.order for event in case_file.timeline.events] == [1, 2, None]


def test_timeline_reconciler_rejects_duplicate_supported_event_orders() -> None:
    llm = StubLLM(
        response={
            "events": [
                {
                    "statement": "The archive was locked.",
                    "time": "9:00 PM",
                    "order": 1,
                    "status": "supported",
                    "evidence_ids": ["R-7"],
                },
                {
                    "statement": "The manuscript was found missing.",
                    "time": "9:20 PM",
                    "order": 1,
                    "status": "supported",
                    "evidence_ids": ["R-9"],
                },
            ],
            "issues": [],
        }
    )

    with pytest.raises(ValueError, match="unique and consecutive"):
        TimelineReconciler(llm).run(_case_file_with_evidence())


@pytest.mark.parametrize("section", ["events", "issues"])
def test_timeline_reconciler_rejects_unknown_evidence_citations(section: str) -> None:
    response = {"events": [], "issues": []}
    response[section] = [
        {
            "statement": "An unsupported timeline claim.",
            "evidence_ids": ["R-404"],
            **(
                {"time": "after dusk", "order": 1, "status": "supported"}
                if section == "events"
                else {"kind": "gap"}
            ),
        }
    ]
    case_file = _case_file_with_evidence()

    with pytest.raises(ValueError, match="not in the case file"):
        TimelineReconciler(StubLLM(response=response)).run(case_file)


def test_timeline_reconciler_only_updates_its_assigned_section() -> None:
    llm = StubLLM(
        response={
            "events": [
                {
                    "statement": "The archive was locked.",
                    "time": "9:00 PM",
                    "order": 1,
                    "status": "supported",
                    "evidence_ids": ["R-7"],
                }
            ],
            "issues": [],
        }
    )
    case_file = _case_file_with_evidence()
    case_file.suspect_profiles = [
        SuspectProfile(
            suspect="The Volunteer",
            motive=(Claim(statement="Motive unknown.", status=ClaimStatus.UNKNOWN),),
            opportunity=(Claim(statement="Opportunity unknown.", status=ClaimStatus.UNKNOWN),),
        )
    ]
    original_evidence = list(case_file.evidence)
    original_profiles = list(case_file.suspect_profiles)

    TimelineReconciler(llm).run(case_file)

    assert case_file.evidence == original_evidence
    assert case_file.suspect_profiles == original_profiles


def test_reference_case_timeline_preserves_material_reasoning_boundaries() -> None:
    participant_case_excerpt = (
        "At 8:00 PM, curator Dr. Mira Sen displayed the Aurora Diamond inside a "
        "locked glass case. At 8:20 PM, a power failure darkened the gallery for "
        "four minutes. At 8:30 PM, staff discovered that the diamond was missing. "
        "The electronic lock records authorized-card access during a power failure. "
        "Arjun Vale said his card remained in his jacket inside the archive."
    )
    case_file = CaseFile(mystery_text=participant_case_excerpt)
    case_file.evidence = [
        EvidenceItem(
            id="REPORT",
            statement=(
                "The diamond was locked in its case at 8:00 PM, the gallery lost power "
                "from 8:20 to 8:24 PM, and the diamond was found missing at 8:30 PM."
            ),
            classification=EvidenceClassification.OBSERVED_FACT,
        ),
        EvidenceItem(
            id="A",
            statement="Valid-card access remains recorded during a power failure.",
            classification=EvidenceClassification.OBSERVED_FACT,
        ),
        EvidenceItem(
            id="B",
            statement="Arjun Vale's card opened the display case at 8:23 PM.",
            classification=EvidenceClassification.OBSERVED_FACT,
        ),
        EvidenceItem(
            id="C",
            statement="Arjun said his card stayed in his jacket inside the archive.",
            classification=EvidenceClassification.OBSERVED_FACT,
        ),
        EvidenceItem(
            id="D",
            statement="At 8:25 PM, Arjun left the archive carrying a flat folder.",
            classification=EvidenceClassification.OBSERVED_FACT,
        ),
    ]
    llm = StubLLM(
        response={
            "events": [
                {
                    "statement": "The diamond was displayed and the case was locked.",
                    "time": "8:00 PM",
                    "order": 1,
                    "status": "supported",
                    "evidence_ids": ["REPORT"],
                },
                {
                    "statement": "The blackout created the likely opportunity window.",
                    "time": "8:20–8:24 PM",
                    "order": 2,
                    "status": "supported",
                    "evidence_ids": ["REPORT", "A"],
                },
                {
                    "statement": (
                        "Arjun's card opened the case; the record does not prove who used it."
                    ),
                    "time": "8:23 PM",
                    "order": 3,
                    "status": "supported",
                    "evidence_ids": ["A", "B"],
                },
                {
                    "statement": "Arjun left the archive carrying a flat folder.",
                    "time": "8:25 PM",
                    "order": 4,
                    "status": "supported",
                    "evidence_ids": ["D"],
                },
                {
                    "statement": "The diamond was reported missing.",
                    "time": "8:30 PM",
                    "order": 5,
                    "status": "supported",
                    "evidence_ids": ["REPORT"],
                },
                {
                    "statement": "The diamond was removed at an unknown time.",
                    "time": None,
                    "order": None,
                    "status": "unknown",
                    "evidence_ids": ["REPORT", "B"],
                },
            ],
            "issues": [
                {
                    "kind": "gap",
                    "statement": "The exact removal time remains unknown.",
                    "evidence_ids": ["REPORT", "B"],
                },
                {
                    "kind": "contradiction",
                    "statement": "Arjun's account conflicts with use of his card.",
                    "evidence_ids": ["B", "C"],
                },
            ],
        }
    )

    TimelineReconciler(llm).run(case_file)

    assert participant_case_excerpt in llm.prompts[0]
    assert "Facilitator Only Solution" not in llm.prompts[0]
    supported_times = [
        event.time
        for event in case_file.timeline.events
        if event.status is ClaimStatus.SUPPORTED
    ]
    assert supported_times == [
        "8:00 PM",
        "8:20–8:24 PM",
        "8:23 PM",
        "8:25 PM",
        "8:30 PM",
    ]
    assert "does not prove who used it" in case_file.timeline.events[2].statement
    assert case_file.timeline.events[-1].time is None
    assert case_file.timeline.issues[0].kind.value == "gap"
