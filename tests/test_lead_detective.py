from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from agents.lead_detective import LeadDetective
from case_file import (
    CaseFile,
    Claim,
    ClaimStatus,
    EvidenceClassification,
    EvidenceItem,
    SkepticFinding,
    SkepticFindingKind,
    SkepticReview,
    SkepticReviewOutcome,
    Specialist,
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


def _case_file_with_approved_review() -> CaseFile:
    """A fully resolved case: every claim and event is supported, no open issues."""
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
                Claim(
                    statement="Owed the victim money, per a witnessed argument.",
                    status=ClaimStatus.SUPPORTED,
                    evidence_ids=("E-01",),
                ),
            ),
            opportunity=(
                Claim(
                    statement="The key card was used at midnight.",
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
        ]
    )
    case_file.skeptic_reviews = [SkepticReview(outcome=SkepticReviewOutcome.APPROVED)]
    return case_file


def _case_file_with_an_unknown_claim() -> CaseFile:
    case_file = _case_file_with_approved_review()
    case_file.suspect_profiles = [
        SuspectProfile(
            suspect="The Housekeeper",
            motive=(Claim(statement="Owed the victim money.", status=ClaimStatus.UNKNOWN),),
            opportunity=case_file.suspect_profiles[0].opportunity,
        )
    ]
    return case_file


def _case_file_with_a_timeline_issue() -> CaseFile:
    case_file = _case_file_with_approved_review()
    case_file.timeline = Timeline(
        events=case_file.timeline.events,
        issues=[
            TimelineIssue(
                kind=TimelineIssueKind.GAP,
                statement="Nobody accounts for the next hour.",
                evidence_ids=("E-01",),
            )
        ],
    )
    return case_file


def _valid_response(confidence: int = 70) -> dict:
    return {
        "conclusions": [
            {
                "rank": 1,
                "suspect": "The Housekeeper",
                "explanation": "The key card ties her to the study at the time of the theft.",
                "evidence_ids": ["E-01"],
            }
        ],
        "confidence": confidence,
        "limitations": ["Who physically held the key card is not established."],
    }


def test_lead_detective_includes_human_guidance_in_the_prompt_when_present() -> None:
    llm = StubLLM(response=_valid_response())
    case_file = _case_file_with_approved_review()
    case_file.human_notes = ["Evidence E-01 is unavailable; do not rely on it."]

    LeadDetective(llm).run(case_file)

    assert "Evidence E-01 is unavailable; do not rely on it." in llm.prompts[0]


def test_lead_detective_prompt_omits_guidance_section_without_human_notes() -> None:
    llm = StubLLM(response=_valid_response())
    case_file = _case_file_with_approved_review()

    LeadDetective(llm).run(case_file)

    assert "human reviewer" not in llm.prompts[0].lower()


def test_lead_detective_produces_a_ranked_evidence_cited_verdict() -> None:
    case_file = _case_file_with_approved_review()
    llm = StubLLM(response=_valid_response())

    LeadDetective(llm).run(case_file)

    verdict = case_file.verdict
    assert verdict is not None
    assert verdict.confidence == 70
    assert len(verdict.conclusions) == 1
    conclusion = verdict.conclusions[0]
    assert conclusion.rank == 1
    assert conclusion.suspect == "The Housekeeper"
    assert conclusion.evidence_ids == ("E-01",)
    assert verdict.limitations == ("Who physically held the key card is not established.",)


def test_lead_detective_orders_conclusions_by_rank_regardless_of_response_order() -> None:
    case_file = _case_file_with_approved_review()
    response = _valid_response()
    response["conclusions"] = [
        {
            "rank": 2,
            "suspect": "The Gardener",
            "explanation": "A weaker, unresolved alternative.",
            "evidence_ids": ["E-01"],
        },
        {
            "rank": 1,
            "suspect": "The Housekeeper",
            "explanation": "The strongest explanation.",
            "evidence_ids": ["E-01"],
        },
    ]
    llm = StubLLM(response=response)

    LeadDetective(llm).run(case_file)

    ranked_suspects = [conclusion.suspect for conclusion in case_file.verdict.conclusions]
    assert ranked_suspects == ["The Housekeeper", "The Gardener"]


def test_lead_detective_rejects_verdict_with_no_conclusions() -> None:
    case_file = _case_file_with_approved_review()
    llm = StubLLM(response={"conclusions": [], "confidence": 10, "limitations": []})

    with pytest.raises(ValueError, match="one or more ranked conclusions"):
        LeadDetective(llm).run(case_file)


def test_lead_detective_rejects_conclusion_citing_an_unknown_evidence_id() -> None:
    case_file = _case_file_with_approved_review()
    response = _valid_response()
    response["conclusions"][0]["evidence_ids"] = ["E-99"]
    llm = StubLLM(response=response)

    with pytest.raises(ValueError, match="not in the case file"):
        LeadDetective(llm).run(case_file)


def test_lead_detective_rejects_conclusion_with_no_citation() -> None:
    case_file = _case_file_with_approved_review()
    response = _valid_response()
    response["conclusions"][0]["evidence_ids"] = []
    llm = StubLLM(response=response)

    with pytest.raises(ValueError, match="must cite at least one evidence ID"):
        LeadDetective(llm).run(case_file)


def test_lead_detective_rejects_non_consecutive_ranks() -> None:
    case_file = _case_file_with_approved_review()
    response = _valid_response()
    response["conclusions"].append(
        {
            "rank": 3,
            "suspect": "The Gardener",
            "explanation": "No strong link established.",
            "evidence_ids": ["E-01"],
        }
    )
    llm = StubLLM(response=response)

    with pytest.raises(ValueError, match="consecutive"):
        LeadDetective(llm).run(case_file)


def test_lead_detective_rejects_duplicate_ranks() -> None:
    case_file = _case_file_with_approved_review()
    response = _valid_response()
    response["conclusions"].append(
        {
            "rank": 1,
            "suspect": "The Gardener",
            "explanation": "No strong link established.",
            "evidence_ids": ["E-01"],
        }
    )
    llm = StubLLM(response=response)

    with pytest.raises(ValueError, match="consecutive"):
        LeadDetective(llm).run(case_file)


@pytest.mark.parametrize("confidence", [-1, 101])
def test_lead_detective_rejects_confidence_outside_0_to_100(confidence: int) -> None:
    case_file = _case_file_with_approved_review()
    llm = StubLLM(response=_valid_response(confidence=confidence))

    with pytest.raises(ValueError):
        LeadDetective(llm).run(case_file)


def test_lead_detective_allows_empty_limitations_when_case_file_is_fully_resolved() -> None:
    case_file = _case_file_with_approved_review()
    response = _valid_response()
    response["limitations"] = []
    llm = StubLLM(response=response)

    LeadDetective(llm).run(case_file)

    assert case_file.verdict.limitations == ()


def test_lead_detective_requires_limitations_when_skeptic_review_is_exhausted() -> None:
    case_file = _case_file_with_approved_review()
    case_file.skeptic_reviews = [
        SkepticReview(
            outcome=SkepticReviewOutcome.EXHAUSTED,
            findings=(
                SkepticFinding(
                    specialist=Specialist.SUSPECT_ANALYST,
                    claim="The key card was used at midnight.",
                    kind=SkepticFindingKind.UNSUPPORTED_REASONING,
                    explanation="Card use does not establish who held it.",
                ),
            ),
        )
    ]
    response = _valid_response()
    response["limitations"] = []
    llm = StubLLM(response=response)

    with pytest.raises(ValueError, match="unresolved"):
        LeadDetective(llm).run(case_file)


def test_lead_detective_requires_limitations_when_a_claim_is_unknown() -> None:
    case_file = _case_file_with_an_unknown_claim()
    response = _valid_response()
    response["limitations"] = []
    llm = StubLLM(response=response)

    with pytest.raises(ValueError, match="unresolved"):
        LeadDetective(llm).run(case_file)


def test_lead_detective_requires_limitations_when_a_timeline_issue_is_open() -> None:
    case_file = _case_file_with_a_timeline_issue()
    response = _valid_response()
    response["limitations"] = []
    llm = StubLLM(response=response)

    with pytest.raises(ValueError, match="unresolved"):
        LeadDetective(llm).run(case_file)


def test_lead_detective_only_updates_its_own_case_file_section() -> None:
    case_file = _case_file_with_approved_review()
    response = _valid_response()
    response["conclusions"][0]["evidence_ids"] = ["E-99"]
    llm = StubLLM(response=response)
    original_suspects = list(case_file.suspect_profiles)
    original_timeline = case_file.timeline

    with pytest.raises(ValueError):
        LeadDetective(llm).run(case_file)

    assert case_file.suspect_profiles == original_suspects
    assert case_file.timeline == original_timeline
    assert case_file.verdict is None
