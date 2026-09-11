from __future__ import annotations

import pytest

from case_file import (
    CaseFile,
    Claim,
    ClaimStatus,
    Conclusion,
    EvidenceItem,
    SkepticReview,
    SkepticReviewOutcome,
    Specialist,
    SuspectProfile,
    Timeline,
    TimelineEvent,
    Verdict,
    VerdictReviewError,
    VerdictReviewStatus,
)


def _verdict() -> Verdict:
    return Verdict(
        conclusions=(
            Conclusion(
                rank=1,
                suspect="The Curator",
                explanation="Present alone with the exhibit before it vanished.",
                evidence_ids=("G-1",),
            ),
        ),
        confidence=70,
        limitations=("The security footage gap is unexplained.",),
    )


def _case_file_with_verdict() -> CaseFile:
    case_file = CaseFile(mystery_text="A painting vanished from the gallery overnight.")
    case_file.verdict = _verdict()
    return case_file


def test_verdict_starts_awaiting_human_review() -> None:
    case_file = _case_file_with_verdict()

    assert case_file.verdict.review_status is VerdictReviewStatus.AWAITING_REVIEW


def test_accept_verdict_records_accepted_decision() -> None:
    case_file = _case_file_with_verdict()

    case_file.accept_verdict()

    assert case_file.verdict.review_status is VerdictReviewStatus.ACCEPTED


def test_reject_verdict_records_rejected_decision() -> None:
    case_file = _case_file_with_verdict()

    case_file.reject_verdict()

    assert case_file.verdict.review_status is VerdictReviewStatus.REJECTED


def test_accept_verdict_does_not_change_verdict_content_or_citations() -> None:
    case_file = _case_file_with_verdict()
    original = case_file.verdict

    case_file.accept_verdict()

    assert case_file.verdict.conclusions == original.conclusions
    assert case_file.verdict.confidence == original.confidence
    assert case_file.verdict.limitations == original.limitations


def test_accept_verdict_without_a_verdict_raises_predictable_error() -> None:
    case_file = CaseFile(mystery_text="A painting vanished from the gallery overnight.")

    with pytest.raises(VerdictReviewError):
        case_file.accept_verdict()


def test_reject_verdict_without_a_verdict_raises_predictable_error() -> None:
    case_file = CaseFile(mystery_text="A painting vanished from the gallery overnight.")

    with pytest.raises(VerdictReviewError):
        case_file.reject_verdict()


def test_repeated_accept_after_accept_raises_and_keeps_original_decision() -> None:
    case_file = _case_file_with_verdict()
    case_file.accept_verdict()

    with pytest.raises(VerdictReviewError):
        case_file.accept_verdict()

    assert case_file.verdict.review_status is VerdictReviewStatus.ACCEPTED


def test_reject_after_accept_raises_and_does_not_flip_the_decision() -> None:
    case_file = _case_file_with_verdict()
    case_file.accept_verdict()

    with pytest.raises(VerdictReviewError):
        case_file.reject_verdict()

    assert case_file.verdict.review_status is VerdictReviewStatus.ACCEPTED


def test_accept_after_reject_raises_and_does_not_flip_the_decision() -> None:
    case_file = _case_file_with_verdict()
    case_file.reject_verdict()

    with pytest.raises(VerdictReviewError):
        case_file.accept_verdict()

    assert case_file.verdict.review_status is VerdictReviewStatus.REJECTED


def test_repeated_reject_after_reject_raises_and_keeps_original_decision() -> None:
    case_file = _case_file_with_verdict()
    case_file.reject_verdict()

    with pytest.raises(VerdictReviewError):
        case_file.reject_verdict()

    assert case_file.verdict.review_status is VerdictReviewStatus.REJECTED


def test_request_reinvestigation_marks_the_verdict_reinvestigation_requested() -> None:
    case_file = _case_file_with_verdict()

    case_file.request_reinvestigation("Evidence G-1 is unavailable; do not rely on it.")

    assert case_file.verdict.review_status is VerdictReviewStatus.REINVESTIGATION_REQUESTED


def test_request_reinvestigation_does_not_change_the_recorded_verdict_content() -> None:
    case_file = _case_file_with_verdict()
    original = case_file.verdict

    case_file.request_reinvestigation("Evidence G-1 is unavailable; do not rely on it.")

    assert case_file.verdict.conclusions == original.conclusions
    assert case_file.verdict.confidence == original.confidence
    assert case_file.verdict.limitations == original.limitations


def test_request_reinvestigation_appends_the_note_to_human_notes() -> None:
    case_file = _case_file_with_verdict()

    case_file.request_reinvestigation("Evidence G-1 is unavailable; do not rely on it.")

    assert case_file.human_notes == ["Evidence G-1 is unavailable; do not rely on it."]


def test_request_reinvestigation_strips_surrounding_whitespace_from_the_note() -> None:
    case_file = _case_file_with_verdict()

    case_file.request_reinvestigation("  Reconsider the housekeeper's alibi.  ")

    assert case_file.human_notes == ["Reconsider the housekeeper's alibi."]


def test_request_reinvestigation_accumulates_notes_across_multiple_requests() -> None:
    case_file = _case_file_with_verdict()
    case_file.request_reinvestigation("First note.")
    case_file.verdict = _verdict()

    case_file.request_reinvestigation("Second note.")

    assert case_file.human_notes == ["First note.", "Second note."]


def test_request_reinvestigation_preserves_mystery_text_and_evidence() -> None:
    case_file = _case_file_with_verdict()
    case_file.evidence = [
        EvidenceItem(id="G-1", statement="The gallery alarm was silent.", classification="observed_fact")
    ]
    original_mystery_text = case_file.mystery_text
    original_evidence = case_file.evidence

    case_file.request_reinvestigation("Reconsider the housekeeper's alibi.")

    assert case_file.mystery_text == original_mystery_text
    assert case_file.evidence == original_evidence


def test_request_reinvestigation_resets_downstream_analysis_state() -> None:
    case_file = _case_file_with_verdict()
    case_file.suspect_profiles = [
        SuspectProfile(
            suspect="The Curator",
            motive=(Claim(statement="Owed money.", status=ClaimStatus.UNKNOWN),),
            opportunity=(Claim(statement="Was present.", status=ClaimStatus.UNKNOWN),),
        )
    ]
    case_file.timeline = Timeline(
        events=[
            TimelineEvent(
                statement="The exhibit was locked.",
                time="midnight",
                order=1,
                status=ClaimStatus.SUPPORTED,
                evidence_ids=("G-1",),
            )
        ]
    )
    case_file.skeptic_reviews = [SkepticReview(outcome=SkepticReviewOutcome.APPROVED)]
    case_file.revised_specialists = {Specialist.SUSPECT_ANALYST}

    case_file.request_reinvestigation("Reconsider the housekeeper's alibi.")

    assert case_file.suspect_profiles == []
    assert case_file.timeline.is_empty
    assert case_file.skeptic_reviews == []
    assert case_file.revised_specialists == set()


def test_request_reinvestigation_with_blank_note_raises_and_does_not_reset_state() -> None:
    case_file = _case_file_with_verdict()
    case_file.suspect_profiles = [
        SuspectProfile(
            suspect="The Curator",
            motive=(Claim(statement="Owed money.", status=ClaimStatus.UNKNOWN),),
            opportunity=(Claim(statement="Was present.", status=ClaimStatus.UNKNOWN),),
        )
    ]

    with pytest.raises(ValueError, match="guidance note"):
        case_file.request_reinvestigation("   ")

    assert case_file.suspect_profiles != []
    assert case_file.human_notes == []
    assert case_file.verdict.review_status is VerdictReviewStatus.AWAITING_REVIEW


def test_request_reinvestigation_without_a_verdict_raises_predictable_error() -> None:
    case_file = CaseFile(mystery_text="A painting vanished from the gallery overnight.")

    with pytest.raises(VerdictReviewError):
        case_file.request_reinvestigation("Reconsider the housekeeper's alibi.")


def test_request_reinvestigation_after_accept_raises_and_does_not_flip_the_decision() -> None:
    case_file = _case_file_with_verdict()
    case_file.accept_verdict()

    with pytest.raises(VerdictReviewError):
        case_file.request_reinvestigation("Reconsider the housekeeper's alibi.")

    assert case_file.verdict.review_status is VerdictReviewStatus.ACCEPTED


def test_accept_after_reinvestigation_requested_raises_and_does_not_flip_the_decision() -> None:
    case_file = _case_file_with_verdict()
    case_file.request_reinvestigation("Reconsider the housekeeper's alibi.")

    with pytest.raises(VerdictReviewError):
        case_file.accept_verdict()

    assert case_file.verdict.review_status is VerdictReviewStatus.REINVESTIGATION_REQUESTED
