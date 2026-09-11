from __future__ import annotations

import pytest

pytest.importorskip("gradio")

from app import render_skeptic_reviews, render_suspect_profiles
from case_file import (
    CaseFile,
    Claim,
    ClaimStatus,
    SkepticFinding,
    SkepticFindingKind,
    SkepticReview,
    SkepticReviewOutcome,
    Specialist,
    SuspectProfile,
)


def test_render_suspect_profiles_displays_supported_and_unknown_claims() -> None:
    case_file = CaseFile(mystery_text="A necklace vanished from the study overnight.")
    case_file.suspect_profiles = [
        SuspectProfile(
            suspect="The Housekeeper",
            motive=(
                Claim(statement="Owed the victim money.", status=ClaimStatus.SUPPORTED, evidence_ids=("E-02",)),
            ),
            opportunity=(
                Claim(statement="No record places her near the study.", status=ClaimStatus.UNKNOWN),
            ),
        )
    ]

    markdown = render_suspect_profiles(case_file)

    assert "The Housekeeper" in markdown
    assert "Owed the victim money. (E-02)" in markdown
    assert "_unknown:_ No record places her near the study." in markdown


def test_render_suspect_profiles_before_analysis_shows_placeholder() -> None:
    case_file = CaseFile(mystery_text="A necklace vanished from the study overnight.")

    assert render_suspect_profiles(case_file) == "_No suspect profiles yet._"


def test_render_skeptic_reviews_before_review_shows_placeholder() -> None:
    case_file = CaseFile(mystery_text="A necklace vanished from the study overnight.")

    assert render_skeptic_reviews(case_file) == "_No Skeptic review yet._"


def test_render_skeptic_reviews_displays_rounds_and_findings() -> None:
    case_file = CaseFile(mystery_text="A necklace vanished from the study overnight.")
    case_file.skeptic_reviews = [
        SkepticReview(
            outcome=SkepticReviewOutcome.REVISION_REQUESTED,
            findings=(
                SkepticFinding(
                    specialist=Specialist.SUSPECT_ANALYST,
                    claim="The key card proves she was in the study.",
                    kind=SkepticFindingKind.UNSUPPORTED_REASONING,
                    explanation="Card use establishes use of the card, not who held it.",
                ),
            ),
        ),
        SkepticReview(outcome=SkepticReviewOutcome.APPROVED),
    ]

    markdown = render_skeptic_reviews(case_file)

    assert "Round 1: Revision Requested" in markdown
    assert "Suspect Analyst — unsupported reasoning" in markdown
    assert "The key card proves she was in the study." in markdown
    assert "Round 2: Approved" in markdown
    assert "No findings." in markdown
