"""Suspect Analyst: builds motive/opportunity profiles per suspect.

Reads `case_file.mystery_text` and `case_file.evidence`. Writes only
`case_file.suspect_profiles`. Every motive and opportunity claim must either
cite evidence IDs that exist in the case file, or be explicitly marked
unknown rather than presented as fact.
"""

from __future__ import annotations

from agents._shared import (
    format_evidence_prompt,
    require_evidence_ids,
    validate_known_evidence_ids,
)
from case_file import CaseFile, Claim, ClaimStatus, SuspectProfile
from llm_client import LLMClient

SYSTEM_PROMPT = (
    "You are the Suspect Analyst on a mystery investigation team. Using "
    "only the supplied evidence, identify the suspects it supports and "
    "build one profile per suspect. Separate motive analysis from "
    "opportunity analysis. Every claim must cite the evidence IDs it is "
    "based on. If the evidence does not support a motive or opportunity "
    "claim, mark that claim as unknown instead of presenting a guess as "
    "fact."
)

_CLAIM_SCHEMA = {
    "type": "object",
    "required": ["statement", "status", "evidence_ids"],
    "properties": {
        "statement": {"type": "string"},
        "status": {"enum": [status.value for status in ClaimStatus]},
        "evidence_ids": {"type": "array", "items": {"type": "string"}},
    },
}

RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["suspects"],
    "properties": {
        "suspects": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "motive", "opportunity"],
                "properties": {
                    "name": {"type": "string"},
                    "motive": {"type": "array", "items": _CLAIM_SCHEMA},
                    "opportunity": {"type": "array", "items": _CLAIM_SCHEMA},
                },
            },
        }
    },
}


class SuspectAnalyst:
    """Reads `case_file.evidence` and writes `case_file.suspect_profiles`."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def run(self, case_file: CaseFile) -> CaseFile:
        response = self._llm.call_llm(
            prompt=format_evidence_prompt(case_file),
            system=SYSTEM_PROMPT,
            response_schema=RESPONSE_SCHEMA,
        )
        known_evidence_ids = {item.id for item in case_file.evidence}
        profiles = [
            self._parse_profile(raw_suspect, known_evidence_ids)
            for raw_suspect in response["suspects"]
        ]
        case_file.suspect_profiles = profiles
        return case_file

    @classmethod
    def _parse_profile(cls, raw_suspect: dict, known_evidence_ids: set[str]) -> SuspectProfile:
        name = raw_suspect["name"]
        motive = cls._parse_claims(raw_suspect["motive"], known_evidence_ids, "motive")
        opportunity = cls._parse_claims(
            raw_suspect["opportunity"], known_evidence_ids, "opportunity"
        )
        if not motive:
            raise ValueError(f"Suspect {name!r} must have at least one motive claim.")
        if not opportunity:
            raise ValueError(f"Suspect {name!r} must have at least one opportunity claim.")
        return SuspectProfile(suspect=name, motive=motive, opportunity=opportunity)

    @staticmethod
    def _parse_claims(
        raw_claims: list[dict], known_evidence_ids: set[str], section: str
    ) -> tuple[Claim, ...]:
        claims: list[Claim] = []
        for raw_claim in raw_claims:
            try:
                status = ClaimStatus(raw_claim["status"])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid {section} claim status: {raw_claim['status']!r}"
                ) from exc

            evidence_ids = validate_known_evidence_ids(
                raw_claim["evidence_ids"], known_evidence_ids, f"{section} claim"
            )
            if status is ClaimStatus.SUPPORTED:
                require_evidence_ids(evidence_ids, f"A supported {section} claim")

            claims.append(
                Claim(
                    statement=raw_claim["statement"],
                    status=status,
                    evidence_ids=evidence_ids,
                )
            )
        return tuple(claims)
