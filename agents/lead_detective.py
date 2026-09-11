"""Lead Detective: synthesizes the case into a ranked, evidence-cited verdict.

Reads `case_file.evidence`, `case_file.suspect_profiles`, `case_file.timeline`,
and `case_file.skeptic_reviews`. Writes only `case_file.verdict`. Runs only
after the Skeptic has approved the specialist claims or the one permitted
revision round has completed. Every conclusion must cite evidence IDs that
exist in the case file, and any unresolved Skeptic finding, unknown claim
or event, or open timeline issue must be named as a limitation rather than
hidden.
"""

from __future__ import annotations

from agents._shared import (
    format_evidence_prompt,
    format_human_guidance,
    format_specialist_claims_prompt,
    require_evidence_ids,
    validate_known_evidence_ids,
)
from case_file import CaseFile, ClaimStatus, Conclusion, SkepticReviewOutcome, Verdict
from llm_client import LLMClient

SYSTEM_PROMPT = (
    "You are the Lead Detective on a mystery investigation team. Read the "
    "collected evidence, the Suspect Analyst's and Timeline Reconciler's "
    "claims, and the Skeptic's review history below. Produce one or more "
    "ranked conclusions, ranked 1 (strongest) upward with no gaps or "
    "duplicate ranks, each naming a suspect and a concise explanation. "
    "Every conclusion must cite one or more evidence IDs that exist in the "
    "case file. Do not present a conclusion as certain when a Skeptic "
    "finding remains unresolved (a review round outcome of 'exhausted') or "
    "the claims behind it are marked unknown; instead lower your confidence "
    "and record the limitation. Assign an integer confidence from 0 (no "
    "support) to 100 (fully supported) that agrees with how certain your "
    "explanation actually is. List every material unresolved Skeptic "
    "finding and any other significant remaining uncertainty as a "
    "limitation instead of omitting it. If human re-investigation guidance "
    "is included below, follow it: do not cite evidence it says is "
    "unavailable in any conclusion, even if that evidence ID still exists "
    "in the case file."
)

_CONCLUSION_SCHEMA = {
    "type": "object",
    "required": ["rank", "suspect", "explanation", "evidence_ids"],
    "properties": {
        "rank": {"type": "integer"},
        "suspect": {"type": "string"},
        "explanation": {"type": "string"},
        "evidence_ids": {"type": "array", "items": {"type": "string"}},
    },
}

RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["conclusions", "confidence", "limitations"],
    "properties": {
        "conclusions": {"type": "array", "items": _CONCLUSION_SCHEMA},
        "confidence": {"type": "integer"},
        "limitations": {"type": "array", "items": {"type": "string"}},
    },
}


class LeadDetective:
    """Reads the full case file and writes only `case_file.verdict`."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def run(self, case_file: CaseFile) -> CaseFile:
        response = self._llm.call_llm(
            prompt=_build_prompt(case_file),
            system=SYSTEM_PROMPT,
            response_schema=RESPONSE_SCHEMA,
        )
        known_evidence_ids = {item.id for item in case_file.evidence}
        conclusions = _parse_conclusions(response["conclusions"], known_evidence_ids)
        limitations = tuple(response["limitations"])
        _require_limitations_when_uncertain(case_file, limitations)
        case_file.verdict = Verdict(
            conclusions=conclusions,
            confidence=response["confidence"],
            limitations=limitations,
        )
        return case_file


def _build_prompt(case_file: CaseFile) -> str:
    lines = [
        format_evidence_prompt(case_file),
        "",
        format_specialist_claims_prompt(case_file),
        "",
        "Skeptic review history:",
        _format_skeptic_reviews(case_file),
    ]
    guidance = format_human_guidance(case_file)
    if guidance:
        lines.extend(["", guidance])
    return "\n".join(lines)


def _format_skeptic_reviews(case_file: CaseFile) -> str:
    if not case_file.skeptic_reviews:
        return "No Skeptic review has run yet."
    lines: list[str] = []
    for round_number, review in enumerate(case_file.skeptic_reviews, start=1):
        lines.append(f"Round {round_number}: {review.outcome.value}")
        for finding in review.findings:
            lines.append(
                f"- {finding.specialist.value} — {finding.kind.value}: "
                f"{finding.claim!r} — {finding.explanation}"
            )
    return "\n".join(lines)


def _parse_conclusions(
    raw_conclusions: list[dict], known_evidence_ids: set[str]
) -> tuple[Conclusion, ...]:
    if not raw_conclusions:
        raise ValueError("A verdict must contain one or more ranked conclusions.")
    conclusions = []
    for raw_conclusion in raw_conclusions:
        evidence_ids = validate_known_evidence_ids(
            raw_conclusion["evidence_ids"], known_evidence_ids, "A verdict conclusion"
        )
        require_evidence_ids(evidence_ids, "Every verdict conclusion")
        conclusions.append(
            Conclusion(
                rank=raw_conclusion["rank"],
                suspect=raw_conclusion["suspect"],
                explanation=raw_conclusion["explanation"],
                evidence_ids=evidence_ids,
            )
        )
    _require_consecutive_ranks(conclusions)
    return tuple(sorted(conclusions, key=lambda conclusion: conclusion.rank))


def _require_consecutive_ranks(conclusions: list[Conclusion]) -> None:
    ranks = sorted(conclusion.rank for conclusion in conclusions)
    if ranks != list(range(1, len(conclusions) + 1)):
        raise ValueError(
            "Verdict conclusion ranks must be unique, start at 1, and be consecutive."
        )


def _require_limitations_when_uncertain(
    case_file: CaseFile, limitations: tuple[str, ...]
) -> None:
    if limitations or not _has_unresolved_uncertainty(case_file):
        return
    raise ValueError(
        "A verdict must state a limitation when the case file has unresolved "
        "Skeptic findings, unknown claims or events, or timeline issues."
    )


def _has_unresolved_uncertainty(case_file: CaseFile) -> bool:
    if (
        case_file.skeptic_reviews
        and case_file.skeptic_reviews[-1].outcome is SkepticReviewOutcome.EXHAUSTED
    ):
        return True
    if case_file.timeline.issues:
        return True
    claims = (
        claim
        for profile in case_file.suspect_profiles
        for claim in (*profile.motive, *profile.opportunity)
    )
    if any(claim.status is ClaimStatus.UNKNOWN for claim in claims):
        return True
    return any(event.status is ClaimStatus.UNKNOWN for event in case_file.timeline.events)
