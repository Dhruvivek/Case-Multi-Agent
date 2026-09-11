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
    Timeline,
    TimelineEvent,
    TimelineIssue,
    TimelineIssueKind,
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
        if "Evidence Collector" not in system:
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

    _, _, first_suspects, first_timeline = updates[4]
    if first_specialist == "timeline":
        assert first_suspects == "_No suspect profiles yet._"
        assert "The observatory door opened" in first_timeline
    else:
        assert "The Astronomer" in first_suspects
        assert first_timeline == "_No timeline analysis yet._"

    transcript, evidence, suspects, timeline = updates[-1]
    assert "Suspect Analyst finished" in transcript
    assert "Timeline Reconciler finished" in transcript
    assert "C-2" in evidence
    assert "The Astronomer" in suspects
    assert "shortly before dawn — The observatory door opened. (C-2)" in timeline
