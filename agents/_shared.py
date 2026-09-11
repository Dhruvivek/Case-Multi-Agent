"""Small shared operations used at specialist-agent boundaries."""

from __future__ import annotations

from case_file import (
    CaseFile,
    Claim,
    ClaimStatus,
    Specialist,
    SkepticReviewOutcome,
    TimelineEvent,
    TimelineIssue,
)


def format_evidence_prompt(case_file: CaseFile) -> str:
    """Render the common mystery and collected-evidence prompt context."""
    evidence_lines = "\n".join(
        f"{item.id} ({item.classification.value}): {item.statement}"
        for item in case_file.evidence
    )
    return f"Mystery:\n{case_file.mystery_text}\n\nEvidence:\n{evidence_lines}"


def format_human_guidance(case_file: CaseFile) -> str | None:
    """Return prior human re-investigation guidance, if any was recorded.

    Returns `None` when no re-investigation has been requested yet.
    """
    if not case_file.human_notes:
        return None
    note_lines = "\n".join(f"- {note}" for note in case_file.human_notes)
    return (
        "A human reviewer requested re-investigation with this guidance. "
        "Follow it, including excluding any evidence it says is "
        "unavailable, while still citing only evidence IDs that exist in "
        "the case file:\n" + note_lines
    )


def format_review_feedback(case_file: CaseFile, specialist: Specialist) -> str | None:
    """Return prior-round review feedback for one specialist, if a revision was requested.

    Returns `None` when there is no pending revision-requested review, or
    when the latest review raised no finding against this specialist.
    """
    if not case_file.skeptic_reviews:
        return None
    latest_review = case_file.skeptic_reviews[-1]
    if latest_review.outcome is not SkepticReviewOutcome.REVISION_REQUESTED:
        return None
    relevant_findings = [
        finding for finding in latest_review.findings if finding.specialist is specialist
    ]
    if not relevant_findings:
        return None
    feedback_lines = "\n".join(
        f"- {finding.claim!r}: {finding.explanation}" for finding in relevant_findings
    )
    return (
        "A reviewer flagged these claims from your previous analysis. Revise "
        "each one to address the concern, downgrading it to unknown if the "
        "evidence truly does not support it:\n" + feedback_lines
    )


def validate_known_evidence_ids(
    raw_evidence_ids: list[str], known_ids: set[str], subject: str
) -> tuple[str, ...]:
    """Return citations after rejecting IDs outside the collected evidence."""
    evidence_ids = tuple(raw_evidence_ids)
    unknown_ids = [
        evidence_id for evidence_id in evidence_ids if evidence_id not in known_ids
    ]
    if unknown_ids:
        raise ValueError(f"{subject} cites evidence IDs not in the case file: {unknown_ids}")
    return evidence_ids


def require_evidence_ids(evidence_ids: tuple[str, ...], subject: str) -> None:
    """Reject a claim type whose contract requires at least one citation."""
    if not evidence_ids:
        raise ValueError(f"{subject} must cite at least one evidence ID.")


def format_claim_line(section: str, claim: Claim) -> str:
    if claim.status is ClaimStatus.UNKNOWN:
        return f"- {section} (unknown): {claim.statement}"
    citations = ", ".join(claim.evidence_ids)
    return f"- {section}: {claim.statement} ({citations})"


def format_event_line(event: TimelineEvent) -> str:
    if event.status is ClaimStatus.UNKNOWN:
        return f"- Event (unknown): {event.statement}"
    citations = ", ".join(event.evidence_ids)
    return f"- Event: {event.statement} ({citations})"


def format_issue_line(issue: TimelineIssue) -> str:
    citations = ", ".join(issue.evidence_ids)
    return f"- Issue ({issue.kind.value}): {issue.statement} ({citations})"


def format_specialist_claims_prompt(case_file: CaseFile) -> str:
    """Render the Suspect Analyst's and Timeline Reconciler's claims for review."""
    lines = ["Suspect Analyst claims:"]
    for profile in case_file.suspect_profiles:
        lines.append(f"Suspect: {profile.suspect}")
        lines.extend(format_claim_line("Motive", claim) for claim in profile.motive)
        lines.extend(format_claim_line("Opportunity", claim) for claim in profile.opportunity)

    lines.append("")
    lines.append("Timeline Reconciler claims:")
    lines.extend(format_event_line(event) for event in case_file.timeline.events)
    lines.extend(format_issue_line(issue) for issue in case_file.timeline.issues)
    return "\n".join(lines)
