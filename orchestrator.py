"""Pipeline and streaming coordination for one investigation."""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum, auto

from agents.collector import EvidenceCollector
from agents.skeptic import Skeptic
from agents.suspect_analyst import SuspectAnalyst
from agents.timeline_reconciler import TimelineReconciler
from case_file import CaseFile, Specialist, SkepticReviewOutcome
from llm_client import LLMClient

EMPTY_MYSTERY_MESSAGE = "Enter a fictional mystery before starting an investigation."

_SPECIALIST_AGENTS = {
    Specialist.SUSPECT_ANALYST: SuspectAnalyst,
    Specialist.TIMELINE_RECONCILER: TimelineReconciler,
}


class InvestigationEventKind(Enum):
    VALIDATION_ERROR = auto()
    EVIDENCE_COLLECTION_STARTED = auto()
    EVIDENCE_COLLECTION_COMPLETED = auto()
    SUSPECT_ANALYSIS_STARTED = auto()
    SUSPECT_ANALYSIS_COMPLETED = auto()
    TIMELINE_RECONCILIATION_STARTED = auto()
    TIMELINE_RECONCILIATION_COMPLETED = auto()
    SKEPTIC_REVIEW_STARTED = auto()
    SKEPTIC_REVIEW_APPROVED = auto()
    SKEPTIC_REVIEW_REVISION_REQUESTED = auto()
    SKEPTIC_REVIEW_EXHAUSTED = auto()
    SPECIALIST_REVISION_STARTED = auto()
    SPECIALIST_REVISION_COMPLETED = auto()


@dataclass
class InvestigationEvent:
    kind: InvestigationEventKind
    case_file: CaseFile | None = None
    message: str | None = None
    specialist: Specialist | None = None


def stream_investigation(mystery_text: str, llm: LLMClient) -> Iterator[InvestigationEvent]:
    """Run the investigation pipeline, yielding one event per pipeline stage.

    Raises whatever the underlying agent raises (e.g. a malformed LLM
    response) rather than swallowing it, so the caller sees a visible error
    instead of a fabricated result.
    """
    if not mystery_text.strip():
        yield InvestigationEvent(
            kind=InvestigationEventKind.VALIDATION_ERROR,
            message=EMPTY_MYSTERY_MESSAGE,
        )
        return

    case_file = CaseFile(mystery_text=mystery_text)
    yield InvestigationEvent(
        kind=InvestigationEventKind.EVIDENCE_COLLECTION_STARTED,
        case_file=case_file,
    )

    EvidenceCollector(llm).run(case_file)

    yield InvestigationEvent(
        kind=InvestigationEventKind.EVIDENCE_COLLECTION_COMPLETED,
        case_file=case_file,
    )

    evidence_snapshot = case_file.model_copy(deep=True)
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_kinds: dict[Future[CaseFile], InvestigationEventKind] = {
            executor.submit(
                SuspectAnalyst(llm).run, evidence_snapshot.model_copy(deep=True)
            ): InvestigationEventKind.SUSPECT_ANALYSIS_COMPLETED,
            executor.submit(
                TimelineReconciler(llm).run, evidence_snapshot.model_copy(deep=True)
            ): InvestigationEventKind.TIMELINE_RECONCILIATION_COMPLETED,
        }
        yield InvestigationEvent(
            kind=InvestigationEventKind.SUSPECT_ANALYSIS_STARTED, case_file=case_file
        )
        yield InvestigationEvent(
            kind=InvestigationEventKind.TIMELINE_RECONCILIATION_STARTED, case_file=case_file
        )

        for future in as_completed(future_kinds):
            completed_kind = future_kinds[future]
            specialist_case_file = future.result()
            if completed_kind is InvestigationEventKind.SUSPECT_ANALYSIS_COMPLETED:
                case_file.suspect_profiles = specialist_case_file.suspect_profiles
            else:
                case_file.timeline = specialist_case_file.timeline
            yield InvestigationEvent(kind=completed_kind, case_file=case_file)

    yield from _run_skeptic_review(case_file, llm)


def _run_skeptic_review(case_file: CaseFile, llm: LLMClient) -> Iterator[InvestigationEvent]:
    """Review specialist claims and run at most one revision round.

    A revision round reruns only the flagged specialist(s) once each with
    the applicable feedback, then re-reviews. Findings still open after
    that round surface as an explicit exhausted state instead of looping
    again.
    """
    review = yield from _review_once(case_file, llm)
    if review.outcome is SkepticReviewOutcome.APPROVED:
        return

    yield InvestigationEvent(
        kind=InvestigationEventKind.SKEPTIC_REVIEW_REVISION_REQUESTED, case_file=case_file
    )

    flagged_specialists = sorted(
        {finding.specialist for finding in review.findings},
        key=lambda specialist: specialist.value,
    )
    for specialist in flagged_specialists:
        yield InvestigationEvent(
            kind=InvestigationEventKind.SPECIALIST_REVISION_STARTED,
            case_file=case_file,
            specialist=specialist,
        )
        _SPECIALIST_AGENTS[specialist](llm).run(case_file)
        case_file.revised_specialists = case_file.revised_specialists | {specialist}
        yield InvestigationEvent(
            kind=InvestigationEventKind.SPECIALIST_REVISION_COMPLETED,
            case_file=case_file,
            specialist=specialist,
        )

    final_review = yield from _review_once(case_file, llm)
    if final_review.outcome is SkepticReviewOutcome.APPROVED:
        return

    case_file.skeptic_reviews[-1] = final_review.model_copy(
        update={"outcome": SkepticReviewOutcome.EXHAUSTED}
    )
    yield InvestigationEvent(kind=InvestigationEventKind.SKEPTIC_REVIEW_EXHAUSTED, case_file=case_file)


def _review_once(case_file: CaseFile, llm: LLMClient) -> Iterator[InvestigationEvent]:
    """Run one Skeptic review round, yielding its events, then return it."""
    yield InvestigationEvent(kind=InvestigationEventKind.SKEPTIC_REVIEW_STARTED, case_file=case_file)
    Skeptic(llm).run(case_file)
    review = case_file.skeptic_reviews[-1]
    if review.outcome is SkepticReviewOutcome.APPROVED:
        yield InvestigationEvent(kind=InvestigationEventKind.SKEPTIC_REVIEW_APPROVED, case_file=case_file)
    return review
