from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from agents.skeptic import Skeptic
from case_file import (
    CaseFile,
    Claim,
    ClaimStatus,
    EvidenceClassification,
    EvidenceItem,
    Specialist,
    SkepticFindingKind,
    SkepticReviewOutcome,
    SuspectProfile,
    Timeline,
    TimelineEvent,
    TimelineIssue,
    TimelineIssueKind,
)


@dataclass
class StubLLM:
    response: dict
    prompts: list[str] = field(default_factory=list)

    def call_llm(self, prompt: str, system: str, response_schema: dict) -> dict:
        self.prompts.append(prompt)
        return self.response


def _case_file_with_specialist_output() -> CaseFile:
    case_file = CaseFile(mystery_text="A necklace vanished from the study overnight.")
    case_file.evidence = [
        EvidenceItem(
            id="E-01",
            statement="The housekeeper's key card unlocked the study at midnight.",
            classification=EvidenceClassification.OBSERVED_FACT,
        ),
    ]
    case_file.suspect_profiles = [
        SuspectProfile(
            suspect="The Housekeeper",
            motive=(
                Claim(statement="Owed the victim money.", status=ClaimStatus.UNKNOWN),
            ),
            opportunity=(
                Claim(
                    statement="The key card used at midnight proves she was in the study.",
                    status=ClaimStatus.SUPPORTED,
                    evidence_ids=("E-01",),
                ),
            ),
        )
    ]
    case_file.timeline = Timeline(
        events=[
            TimelineEvent(
                statement="The study was unlocked.",
                time="midnight",
                order=1,
                status=ClaimStatus.SUPPORTED,
                evidence_ids=("E-01",),
            )
        ],
        issues=[
            TimelineIssue(
                kind=TimelineIssueKind.GAP,
                statement="Nobody accounts for the next hour.",
                evidence_ids=("E-01",),
            )
        ],
    )
    return case_file


def test_skeptic_includes_human_guidance_in_the_prompt_when_present() -> None:
    llm = StubLLM(response={"findings": []})
    case_file = _case_file_with_specialist_output()
    case_file.human_notes = ["Evidence E-01 is unavailable; do not rely on it."]

    Skeptic(llm).run(case_file)

    assert "Evidence E-01 is unavailable; do not rely on it." in llm.prompts[0]


def test_skeptic_prompt_omits_guidance_section_without_human_notes() -> None:
    llm = StubLLM(response={"findings": []})
    case_file = _case_file_with_specialist_output()

    Skeptic(llm).run(case_file)

    assert "human reviewer" not in llm.prompts[0].lower()


def test_skeptic_approves_when_no_findings() -> None:
    llm = StubLLM(response={"findings": []})
    case_file = _case_file_with_specialist_output()

    Skeptic(llm).run(case_file)

    assert len(case_file.skeptic_reviews) == 1
    review = case_file.skeptic_reviews[0]
    assert review.outcome is SkepticReviewOutcome.APPROVED
    assert review.findings == ()


def test_skeptic_records_findings_naming_specialist_and_claim() -> None:
    llm = StubLLM(
        response={
            "findings": [
                {
                    "specialist": "suspect_analyst",
                    "claim": "The key card used at midnight proves she was in the study.",
                    "kind": "unsupported_reasoning",
                    "explanation": "Card use establishes use of the card, not who held it.",
                }
            ]
        }
    )
    case_file = _case_file_with_specialist_output()

    Skeptic(llm).run(case_file)

    review = case_file.skeptic_reviews[0]
    assert review.outcome is SkepticReviewOutcome.REVISION_REQUESTED
    assert len(review.findings) == 1
    finding = review.findings[0]
    assert finding.specialist is Specialist.SUSPECT_ANALYST
    assert finding.claim == "The key card used at midnight proves she was in the study."
    assert finding.kind is SkepticFindingKind.UNSUPPORTED_REASONING


def test_skeptic_rejects_finding_that_cites_an_unknown_claim() -> None:
    llm = StubLLM(
        response={
            "findings": [
                {
                    "specialist": "suspect_analyst",
                    "claim": "A claim nobody made.",
                    "kind": "unsupported_reasoning",
                    "explanation": "This does not exist.",
                }
            ]
        }
    )
    case_file = _case_file_with_specialist_output()

    with pytest.raises(ValueError, match="not made by"):
        Skeptic(llm).run(case_file)


def test_skeptic_rejects_invalid_specialist_value() -> None:
    llm = StubLLM(
        response={
            "findings": [
                {
                    "specialist": "lead_detective",
                    "claim": "The key card used at midnight proves she was in the study.",
                    "kind": "unsupported_reasoning",
                    "explanation": "Not a real specialist.",
                }
            ]
        }
    )
    case_file = _case_file_with_specialist_output()

    with pytest.raises(ValueError, match="Unknown specialist"):
        Skeptic(llm).run(case_file)


def test_skeptic_only_updates_its_own_case_file_section() -> None:
    llm = StubLLM(
        response={
            "findings": [
                {
                    "specialist": "suspect_analyst",
                    "claim": "A claim nobody made.",
                    "kind": "unsupported_reasoning",
                    "explanation": "This does not exist.",
                }
            ]
        }
    )
    case_file = _case_file_with_specialist_output()
    original_suspects = list(case_file.suspect_profiles)
    original_timeline = case_file.timeline

    with pytest.raises(ValueError):
        Skeptic(llm).run(case_file)

    assert case_file.suspect_profiles == original_suspects
    assert case_file.timeline == original_timeline
    assert case_file.skeptic_reviews == []


def test_skeptic_appends_reviews_across_multiple_calls() -> None:
    llm = StubLLM(response={"findings": []})
    case_file = _case_file_with_specialist_output()

    Skeptic(llm).run(case_file)
    Skeptic(llm).run(case_file)

    assert len(case_file.skeptic_reviews) == 2
    assert all(
        review.outcome is SkepticReviewOutcome.APPROVED
        for review in case_file.skeptic_reviews
    )
