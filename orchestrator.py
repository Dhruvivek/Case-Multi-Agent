"""Pipeline and streaming coordination for one investigation."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum, auto

from agents.collector import EvidenceCollector
from agents.lead_detective import LeadDetective
from agents.skeptic import Skeptic
from agents.suspect_analyst import SuspectAnalyst
from agents.timeline_reconciler import TimelineReconciler
from case_file import CaseFile, Specialist, SkepticReviewOutcome
from llm_client import LLMClient, LLMError

EMPTY_MYSTERY_MESSAGE = "Enter a fictional mystery before starting an investigation."
EMPTY_GUIDANCE_MESSAGE = "Enter a guidance note before requesting re-investigation."

_SPECIALIST_AGENTS = {
    Specialist.SUSPECT_ANALYST: SuspectAnalyst,
    Specialist.TIMELINE_RECONCILER: TimelineReconciler,
}

_EVIDENCE_COLLECTOR_NAME = "Evidence Collector"
_SKEPTIC_NAME = "Skeptic"
_LEAD_DETECTIVE_NAME = "Lead Detective"
_SPECIALIST_AGENT_NAMES = {
    Specialist.SUSPECT_ANALYST: "Suspect Analyst",
    Specialist.TIMELINE_RECONCILER: "Timeline Reconciler",
}


class _StepFailed(Exception):
    """Internal signal that one agent's LLM boundary call failed twice.

    Carries the failing agent's display name and a sanitized message so the
    outermost pipeline function can turn it into a visible `STEP_FAILED`
    event and stop the generator, halting any dependent downstream agents.
    Never raised past the public `stream_investigation`/`stream_reinvestigation` API.
    """

    def __init__(self, agent_name: str, message: str) -> None:
        super().__init__(message)
        self.agent_name = agent_name
        self.message = message


def _run_agent(agent_name: str, action: Callable[[], object]) -> None:
    """Run one agent step, converting an `LLMError` into a `_StepFailed`."""
    try:
        action()
    except LLMError as error:
        raise _StepFailed(agent_name, str(error)) from error


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
    LEAD_DETECTIVE_STARTED = auto()
    LEAD_DETECTIVE_COMPLETED = auto()
    REINVESTIGATION_REQUESTED = auto()
    STEP_FAILED = auto()


@dataclass
class InvestigationEvent:
    kind: InvestigationEventKind
    case_file: CaseFile | None = None
    message: str | None = None
    specialist: Specialist | None = None
    agent_name: str | None = None


def stream_investigation(mystery_text: str, llm: LLMClient) -> Iterator[InvestigationEvent]:
    """Run the investigation pipeline, yielding one event per pipeline stage.

    A `LLMError` from the shared LLM boundary (malformed JSON or
    schema-invalid data that survived one reformat retry) is caught at the
    failing agent and surfaced as a `STEP_FAILED` event naming that agent,
    then the generator ends without running any dependent downstream agent.
    Any other exception an agent raises (e.g. a business-rule violation)
    propagates rather than being swallowed, so the caller still sees a
    visible error instead of a fabricated result.
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

    try:
        _run_agent(_EVIDENCE_COLLECTOR_NAME, lambda: EvidenceCollector(llm).run(case_file))
    except _StepFailed as failure:
        yield _step_failed_event(case_file, failure)
        return

    yield InvestigationEvent(
        kind=InvestigationEventKind.EVIDENCE_COLLECTION_COMPLETED,
        case_file=case_file,
    )

    try:
        yield from _run_analysis_and_verdict(case_file, llm)
    except _StepFailed as failure:
        yield _step_failed_event(case_file, failure)


def stream_reinvestigation(
    case_file: CaseFile, note: str, llm: LLMClient
) -> Iterator[InvestigationEvent]:
    """Restart the pipeline from suspect analysis with a human guidance note.

    Reuses `case_file`'s existing mystery text and evidence rather than
    rerunning the Evidence Collector. A blank `note` yields a validation
    event against the unchanged `case_file` instead of starting a pass.
    Raises whatever the underlying agent raises, same as
    `stream_investigation`.
    """
    if not note.strip():
        yield InvestigationEvent(
            kind=InvestigationEventKind.VALIDATION_ERROR,
            message=EMPTY_GUIDANCE_MESSAGE,
            case_file=case_file,
        )
        return

    case_file.request_reinvestigation(note)
    yield InvestigationEvent(
        kind=InvestigationEventKind.REINVESTIGATION_REQUESTED,
        case_file=case_file,
        message=note,
    )

    try:
        yield from _run_analysis_and_verdict(case_file, llm)
    except _StepFailed as failure:
        yield _step_failed_event(case_file, failure)


def _step_failed_event(case_file: CaseFile, failure: _StepFailed) -> InvestigationEvent:
    return InvestigationEvent(
        kind=InvestigationEventKind.STEP_FAILED,
        case_file=case_file,
        agent_name=failure.agent_name,
        message=failure.message,
    )


def _run_analysis_and_verdict(case_file: CaseFile, llm: LLMClient) -> Iterator[InvestigationEvent]:
    """Run specialist analysis through a fresh verdict against `case_file`.

    Shared by a fresh investigation (after evidence collection) and a
    re-investigation restart (after evidence reuse and human guidance).
    """
    evidence_snapshot = case_file.model_copy(deep=True)
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_meta: dict[Future[CaseFile], tuple[InvestigationEventKind, str]] = {
            executor.submit(
                SuspectAnalyst(llm).run, evidence_snapshot.model_copy(deep=True)
            ): (
                InvestigationEventKind.SUSPECT_ANALYSIS_COMPLETED,
                _SPECIALIST_AGENT_NAMES[Specialist.SUSPECT_ANALYST],
            ),
            executor.submit(
                TimelineReconciler(llm).run, evidence_snapshot.model_copy(deep=True)
            ): (
                InvestigationEventKind.TIMELINE_RECONCILIATION_COMPLETED,
                _SPECIALIST_AGENT_NAMES[Specialist.TIMELINE_RECONCILER],
            ),
        }
        yield InvestigationEvent(
            kind=InvestigationEventKind.SUSPECT_ANALYSIS_STARTED, case_file=case_file
        )
        yield InvestigationEvent(
            kind=InvestigationEventKind.TIMELINE_RECONCILIATION_STARTED, case_file=case_file
        )

        for future in as_completed(future_meta):
            completed_kind, agent_name = future_meta[future]
            try:
                specialist_case_file = future.result()
            except LLMError as error:
                raise _StepFailed(agent_name, str(error)) from error
            if completed_kind is InvestigationEventKind.SUSPECT_ANALYSIS_COMPLETED:
                case_file.suspect_profiles = specialist_case_file.suspect_profiles
            else:
                case_file.timeline = specialist_case_file.timeline
            yield InvestigationEvent(kind=completed_kind, case_file=case_file)

    yield from _run_skeptic_review(case_file, llm)

    yield InvestigationEvent(
        kind=InvestigationEventKind.LEAD_DETECTIVE_STARTED, case_file=case_file
    )
    _run_agent(_LEAD_DETECTIVE_NAME, lambda: LeadDetective(llm).run(case_file))
    yield InvestigationEvent(
        kind=InvestigationEventKind.LEAD_DETECTIVE_COMPLETED, case_file=case_file
    )


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
        _run_agent(
            _SPECIALIST_AGENT_NAMES[specialist],
            # Bind the loop variable eagerly; a bare closure would see
            # whatever `specialist` is by the time the lambda runs.
            lambda specialist=specialist: _SPECIALIST_AGENTS[specialist](llm).run(case_file),
        )
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
    _run_agent(_SKEPTIC_NAME, lambda: Skeptic(llm).run(case_file))
    review = case_file.skeptic_reviews[-1]
    if review.outcome is SkepticReviewOutcome.APPROVED:
        yield InvestigationEvent(kind=InvestigationEventKind.SKEPTIC_REVIEW_APPROVED, case_file=case_file)
    return review
