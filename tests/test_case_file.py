from __future__ import annotations

import pytest

from case_file import (
    CaseFile,
    Conclusion,
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
