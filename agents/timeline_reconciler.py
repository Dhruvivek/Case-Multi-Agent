"""Timeline Reconciler: orders supported events and exposes uncertainty."""

from __future__ import annotations

from agents._shared import (
    format_evidence_prompt,
    require_evidence_ids,
    validate_known_evidence_ids,
)
from case_file import (
    CaseFile,
    ClaimStatus,
    Timeline,
    TimelineEvent,
    TimelineIssue,
    TimelineIssueKind,
)
from llm_client import LLMClient

SYSTEM_PROMPT = (
    "You are the Timeline Reconciler on a fictional detective team. Using "
    "only the supplied evidence, build a chronological sequence of events "
    "and identify timeline gaps and contradictions. Every event and issue "
    "must cite the existing evidence IDs it relies on. Use a supported event "
    "only when the evidence establishes its time or relative order. When an "
    "event cannot be ordered, set its status to 'unknown' and its time to "
    "null; never invent a timestamp or ordering. Card or credential use "
    "establishes use of that credential, not the user's identity."
)

_EVENT_SCHEMA = {
    "type": "object",
    "required": ["statement", "time", "order", "status", "evidence_ids"],
    "properties": {
        "statement": {"type": "string"},
        "time": {"type": "string", "nullable": True},
        "order": {"type": "integer", "nullable": True},
        "status": {"enum": [status.value for status in ClaimStatus]},
        "evidence_ids": {"type": "array", "items": {"type": "string"}},
    },
}

_ISSUE_SCHEMA = {
    "type": "object",
    "required": ["kind", "statement", "evidence_ids"],
    "properties": {
        "kind": {"enum": [kind.value for kind in TimelineIssueKind]},
        "statement": {"type": "string"},
        "evidence_ids": {"type": "array", "items": {"type": "string"}},
    },
}

RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["events", "issues"],
    "properties": {
        "events": {"type": "array", "items": _EVENT_SCHEMA},
        "issues": {"type": "array", "items": _ISSUE_SCHEMA},
    },
}


class TimelineReconciler:
    """Reads evidence and writes only the case file's timeline section."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def run(self, case_file: CaseFile) -> CaseFile:
        response = self._llm.call_llm(
            prompt=format_evidence_prompt(case_file),
            system=SYSTEM_PROMPT,
            response_schema=RESPONSE_SCHEMA,
        )
        evidence_ids = {item.id for item in case_file.evidence}
        events = [
            self._parse_event(item, evidence_ids) for item in response["events"]
        ]
        timeline = Timeline(
            events=_ordered_events(events),
            issues=[
                self._parse_issue(item, evidence_ids) for item in response["issues"]
            ],
        )
        case_file.timeline = timeline
        return case_file

    @staticmethod
    def _parse_event(raw_event: dict, known_ids: set[str]) -> TimelineEvent:
        status = ClaimStatus(raw_event["status"])
        evidence_ids = validate_known_evidence_ids(
            raw_event["evidence_ids"], known_ids, "Timeline event"
        )
        require_evidence_ids(evidence_ids, "Every timeline event")
        time = raw_event["time"]
        order = raw_event["order"]
        if _unknown_event_has_position(status, time, order):
            raise ValueError(
                "An unknown timeline event cannot have an invented time or order."
            )
        if _supported_event_lacks_position(status, time, order):
            raise ValueError(
                "A supported timeline event must have an evidence-backed time and order."
            )
        return TimelineEvent(
            statement=raw_event["statement"],
            time=time,
            order=order,
            status=status,
            evidence_ids=evidence_ids,
        )

    @staticmethod
    def _parse_issue(raw_issue: dict, known_ids: set[str]) -> TimelineIssue:
        return TimelineIssue(
            kind=TimelineIssueKind(raw_issue["kind"]),
            statement=raw_issue["statement"],
            evidence_ids=_parse_issue_citations(raw_issue, known_ids),
        )


def _parse_issue_citations(raw_issue: dict, known_ids: set[str]) -> tuple[str, ...]:
    evidence_ids = validate_known_evidence_ids(
        raw_issue["evidence_ids"], known_ids, "Timeline issue"
    )
    require_evidence_ids(evidence_ids, "Every timeline issue")
    return evidence_ids


def _ordered_events(events: list[TimelineEvent]) -> list[TimelineEvent]:
    supported_orders = sorted(
        event.order for event in events if event.order is not None
    )
    expected_orders = list(range(1, len(supported_orders) + 1))
    if supported_orders != expected_orders:
        raise ValueError("Supported timeline event orders must be unique and consecutive.")
    return sorted(events, key=lambda event: (event.order is None, event.order or 0))


def _unknown_event_has_position(
    status: ClaimStatus, time: str | None, order: int | None
) -> bool:
    has_position = time is not None or order is not None
    return status is ClaimStatus.UNKNOWN and has_position


def _supported_event_lacks_position(
    status: ClaimStatus, time: str | None, order: int | None
) -> bool:
    lacks_position = not time or order is None
    return status is ClaimStatus.SUPPORTED and lacks_position
