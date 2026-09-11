from __future__ import annotations

import sys
import threading
import time
from types import ModuleType

import pytest

try:
    import gradio  # noqa: F401
except ModuleNotFoundError:
    sys.modules["gradio"] = ModuleType("gradio")

import app
from case_file import (
    CaseFile,
    ClaimStatus,
    Conclusion,
    Timeline,
    TimelineEvent,
    TimelineIssue,
    TimelineIssueKind,
    Verdict,
    VerdictReviewStatus,
)


class DisplayLLM:
    def call_llm(self, prompt: str, system: str, response_schema: dict) -> dict:
        if "Evidence Collector" in system:
            return {
                "evidence": [
                    {
                        "id": "C-2",
                        "statement": "The observatory door opened shortly before dawn.",
                        "classification": "observed_fact",
                    }
                ]
            }
        if "Lead Detective" in system:
            return {
                "conclusions": [
                    {
                        "rank": 1,
                        "suspect": "The Astronomer",
                        "explanation": "The available evidence does not establish their location.",
                        "evidence_ids": ["C-2"],
                    }
                ],
                "confidence": 35,
                "limitations": ["Who opened the door is not established."],
            }
        if "Skeptic" in system:
            return {"findings": []}
        if "Timeline Reconciler" in system:
            return {
                "events": [
                    {
                        "statement": "The observatory door opened.",
                        "time": "shortly before dawn",
                        "order": 1,
                        "status": "supported",
                        "evidence_ids": ["C-2"],
                    }
                ],
                "issues": [
                    {
                        "kind": "gap",
                        "statement": "The exact opening time is not recorded.",
                        "evidence_ids": ["C-2"],
                    }
                ],
            }
        return {
            "suspects": [
                {
                    "name": "The Astronomer",
                    "motive": [
                        {
                            "statement": "No motive is established.",
                            "status": "unknown",
                            "evidence_ids": [],
                        }
                    ],
                    "opportunity": [
                        {
                            "statement": "The evidence does not establish their location.",
                            "status": "unknown",
                            "evidence_ids": [],
                        }
                    ],
                }
            ]
        }


class OrderedDisplayLLM(DisplayLLM):
    def __init__(self, first_specialist: str) -> None:
        self._first_specialist = first_specialist
        self._rendezvous = threading.Barrier(2)
        self._first_finished = threading.Event()

    def call_llm(self, prompt: str, system: str, response_schema: dict) -> dict:
        if "Evidence Collector" not in system and "Skeptic" not in system:
            specialist = "timeline" if "Timeline Reconciler" in system else "suspect"
            self._rendezvous.wait(timeout=2)
            if specialist == self._first_specialist:
                self._first_finished.set()
            else:
                assert self._first_finished.wait(timeout=2)
                time.sleep(0.01)
        return super().call_llm(prompt, system, response_schema)


def test_render_timeline_displays_supported_uncertain_and_issue_entries() -> None:
    case_file = CaseFile(mystery_text="A lens vanished from an observatory.")
    case_file.timeline = Timeline(
        events=[
            TimelineEvent(
                statement="The observatory opened.",
                time="before dawn",
                order=1,
                status=ClaimStatus.SUPPORTED,
                evidence_ids=("C-2",),
            ),
            TimelineEvent(
                statement="The lens was removed.",
                time=None,
                order=None,
                status=ClaimStatus.UNKNOWN,
                evidence_ids=("C-2", "C-8"),
            ),
        ],
        issues=[
            TimelineIssue(
                kind=TimelineIssueKind.GAP,
                statement="The exact removal time is unknown.",
                evidence_ids=("C-2", "C-8"),
            )
        ],
    )

    markdown = app.render_timeline(case_file)

    assert "before dawn — The observatory opened. (C-2)" in markdown
    assert "_uncertain / unordered:_ The lens was removed. (C-2, C-8)" in markdown
    assert "Gap: The exact removal time is unknown. (C-2, C-8)" in markdown


def test_render_verdict_before_verdict_shows_placeholder() -> None:
    case_file = CaseFile(mystery_text="A lens vanished from an observatory.")

    assert app.render_verdict(case_file) == "_No verdict yet._"


def test_render_verdict_displays_ranking_citations_confidence_and_limitations() -> None:
    case_file = CaseFile(mystery_text="A lens vanished from an observatory.")
    case_file.verdict = Verdict(
        conclusions=(
            Conclusion(
                rank=1,
                suspect="The Astronomer",
                explanation="Present at the observatory when the lens went missing.",
                evidence_ids=("C-2",),
            ),
        ),
        confidence=55,
        limitations=("Who opened the door is not established.",),
    )

    markdown = app.render_verdict(case_file)

    assert "Confidence: 55/100" in markdown
    assert "1. **The Astronomer** — Present at the observatory when the lens went missing. (C-2)" in markdown
    assert "Who opened the door is not established." in markdown
    assert "proposal pending human review" in markdown


def _verdict_case_file() -> CaseFile:
    case_file = CaseFile(mystery_text="A lens vanished from an observatory.")
    case_file.verdict = Verdict(
        conclusions=(
            Conclusion(
                rank=1,
                suspect="The Astronomer",
                explanation="Present at the observatory when the lens went missing.",
                evidence_ids=("C-2",),
            ),
        ),
        confidence=55,
    )
    return case_file


def test_render_verdict_shows_accepted_decision() -> None:
    case_file = _verdict_case_file()
    case_file.accept_verdict()

    assert "Human decision: Accepted." in app.render_verdict(case_file)


def test_render_verdict_shows_rejected_decision() -> None:
    case_file = _verdict_case_file()
    case_file.reject_verdict()

    assert "Human decision: Rejected." in app.render_verdict(case_file)


def test_sync_review_controls_disabled_before_a_verdict_exists() -> None:
    case_file = CaseFile(mystery_text="A lens vanished from an observatory.")

    accept_update, reject_update = app.sync_review_controls(case_file)

    assert accept_update["interactive"] is False
    assert reject_update["interactive"] is False


def test_sync_review_controls_disabled_when_case_file_is_none() -> None:
    accept_update, reject_update = app.sync_review_controls(None)

    assert accept_update["interactive"] is False
    assert reject_update["interactive"] is False


def test_sync_review_controls_enabled_while_verdict_awaits_review() -> None:
    case_file = _verdict_case_file()

    accept_update, reject_update = app.sync_review_controls(case_file)

    assert accept_update["interactive"] is True
    assert reject_update["interactive"] is True


def test_sync_review_controls_disabled_once_verdict_is_decided() -> None:
    case_file = _verdict_case_file()
    case_file.accept_verdict()

    accept_update, reject_update = app.sync_review_controls(case_file)

    assert accept_update["interactive"] is False
    assert reject_update["interactive"] is False


def test_handle_accept_verdict_records_decision_and_disables_controls() -> None:
    case_file = _verdict_case_file()

    verdict_markdown, updated_case_file, accept_update, reject_update = (
        app.handle_accept_verdict(case_file)
    )

    assert updated_case_file.verdict.review_status is VerdictReviewStatus.ACCEPTED
    assert "Human decision: Accepted." in verdict_markdown
    assert accept_update["interactive"] is False
    assert reject_update["interactive"] is False


def test_handle_reject_verdict_records_decision_and_disables_controls() -> None:
    case_file = _verdict_case_file()

    verdict_markdown, updated_case_file, accept_update, reject_update = (
        app.handle_reject_verdict(case_file)
    )

    assert updated_case_file.verdict.review_status is VerdictReviewStatus.REJECTED
    assert "Human decision: Rejected." in verdict_markdown
    assert accept_update["interactive"] is False
    assert reject_update["interactive"] is False


def test_handle_accept_verdict_with_no_case_file_is_a_predictable_no_op() -> None:
    verdict_markdown, updated_case_file, accept_update, reject_update = (
        app.handle_accept_verdict(None)
    )

    assert verdict_markdown == "_No verdict yet._"
    assert updated_case_file is None
    assert accept_update["interactive"] is False
    assert reject_update["interactive"] is False


def test_repeated_accept_after_reject_does_not_flip_the_recorded_decision() -> None:
    case_file = _verdict_case_file()
    app.handle_reject_verdict(case_file)

    verdict_markdown, updated_case_file, accept_update, reject_update = (
        app.handle_accept_verdict(case_file)
    )

    assert updated_case_file.verdict.review_status is VerdictReviewStatus.REJECTED
    assert "Human decision: Rejected." in verdict_markdown
    assert accept_update["interactive"] is False
    assert reject_update["interactive"] is False


@pytest.mark.parametrize("first_specialist", ["timeline", "suspect"])
def test_complete_displayed_pipeline_includes_both_specialists_in_either_order(
    monkeypatch: pytest.MonkeyPatch,
    first_specialist: str,
) -> None:
    monkeypatch.setattr(
        app,
        "EnvLLMClient",
        lambda: OrderedDisplayLLM(first_specialist),
    )

    updates = list(app.run_investigation("A lens vanished from an observatory."))

    _, _, first_suspects, first_timeline, _, _, _ = updates[4]
    if first_specialist == "timeline":
        assert first_suspects == "_No suspect profiles yet._"
        assert "The observatory door opened" in first_timeline
    else:
        assert "The Astronomer" in first_suspects
        assert first_timeline == "_No timeline analysis yet._"

    transcript, evidence, suspects, timeline, skeptic, verdict, case_file = updates[-1]
    assert "Suspect Analyst finished" in transcript
    assert "Timeline Reconciler finished" in transcript
    assert "Skeptic approved" in transcript
    assert "Lead Detective finished" in transcript
    assert "C-2" in evidence
    assert "The Astronomer" in suspects
    assert "shortly before dawn — The observatory door opened. (C-2)" in timeline
    assert "Round 1: Approved" in skeptic
    assert "Confidence: 35/100" in verdict
    assert "1. **The Astronomer**" in verdict
    assert "Who opened the door is not established." in verdict
    assert "proposal pending human review" in verdict
    assert case_file is not None
    assert case_file.verdict.review_status is VerdictReviewStatus.AWAITING_REVIEW
