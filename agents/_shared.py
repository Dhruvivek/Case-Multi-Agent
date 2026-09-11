"""Small shared operations used at specialist-agent boundaries."""

from __future__ import annotations

from case_file import CaseFile


def format_evidence_prompt(case_file: CaseFile) -> str:
    """Render the common mystery and collected-evidence prompt context."""
    evidence_lines = "\n".join(
        f"{item.id} ({item.classification.value}): {item.statement}"
        for item in case_file.evidence
    )
    return f"Mystery:\n{case_file.mystery_text}\n\nEvidence:\n{evidence_lines}"


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
