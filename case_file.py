"""Shared investigation state passed through the agent pipeline."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class EvidenceClassification(str, Enum):
    """Whether an evidence item is a recorded observation or a drawn inference."""

    OBSERVED_FACT = "observed_fact"
    INFERENCE = "inference"


class EvidenceItem(BaseModel):
    """One ID-tagged fact or inference extracted from the mystery text."""

    id: str
    statement: str
    classification: EvidenceClassification


class ClaimStatus(str, Enum):
    """Whether a specialist claim is backed by cited evidence or unresolved."""

    SUPPORTED = "supported"
    UNKNOWN = "unknown"


class Claim(BaseModel):
    """A single motive or opportunity statement.

    A `SUPPORTED` claim must cite one or more evidence IDs that exist in the
    case file. An `UNKNOWN` claim marks the analysis as unresolved rather
    than presenting a guess as fact.
    """

    statement: str
    status: ClaimStatus
    evidence_ids: tuple[str, ...] = ()


class SuspectProfile(BaseModel):
    """The Suspect Analyst's motive/opportunity profile for one suspect."""

    suspect: str
    motive: tuple[Claim, ...]
    opportunity: tuple[Claim, ...]


class TimelineEvent(BaseModel):
    """An event placed in time, or explicitly left unordered."""

    statement: str
    time: str | None
    order: int | None
    status: ClaimStatus
    evidence_ids: tuple[str, ...]


class TimelineIssueKind(str, Enum):
    """A missing link or a conflict exposed while reconciling events."""

    GAP = "gap"
    CONTRADICTION = "contradiction"


class TimelineIssue(BaseModel):
    """An evidence-cited gap or contradiction in the timeline."""

    kind: TimelineIssueKind
    statement: str
    evidence_ids: tuple[str, ...]


class Timeline(BaseModel):
    """The Timeline Reconciler's ordered events and unresolved issues."""

    events: list[TimelineEvent] = Field(default_factory=list)
    issues: list[TimelineIssue] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.events and not self.issues


class Specialist(str, Enum):
    """A specialist agent a Skeptic finding or revision round targets."""

    SUSPECT_ANALYST = "suspect_analyst"
    TIMELINE_RECONCILER = "timeline_reconciler"


class SkepticFindingKind(str, Enum):
    """The kind of weakness a Skeptic finding identifies in a claim."""

    MISSING_CITATION = "missing_citation"
    NONEXISTENT_EVIDENCE_ID = "nonexistent_evidence_id"
    UNSUPPORTED_REASONING = "unsupported_reasoning"


class SkepticFinding(BaseModel):
    """One Skeptic challenge naming the affected specialist and claim."""

    specialist: Specialist
    claim: str
    kind: SkepticFindingKind
    explanation: str


class SkepticReviewOutcome(str, Enum):
    """The result of one Skeptic review round."""

    APPROVED = "approved"
    REVISION_REQUESTED = "revision_requested"
    EXHAUSTED = "exhausted"


class SkepticReview(BaseModel):
    """One inspectable Skeptic review round and any findings it raised."""

    outcome: SkepticReviewOutcome
    findings: tuple[SkepticFinding, ...] = ()


class Conclusion(BaseModel):
    """One ranked, evidence-cited explanation the Lead Detective proposes.

    Ranks start at 1 and must be unique and consecutive across a verdict's
    conclusions, with 1 the most strongly supported.
    """

    rank: int
    suspect: str
    explanation: str
    evidence_ids: tuple[str, ...]


class VerdictReviewStatus(str, Enum):
    """Whether a proposed verdict still awaits a human decision."""

    AWAITING_REVIEW = "awaiting_review"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    REINVESTIGATION_REQUESTED = "reinvestigation_requested"


class VerdictReviewError(ValueError):
    """Raised when a human review decision cannot be applied to the verdict."""


class Verdict(BaseModel):
    """The Lead Detective's ranked, evidence-cited proposal.

    This is a proposal only: storing it on the case file does not mean a
    human has reviewed or accepted it. `limitations` names any material
    unresolved Skeptic finding or notable uncertainty instead of hiding it.
    `review_status` starts `AWAITING_REVIEW` and is only ever changed by
    `CaseFile.accept_verdict`/`reject_verdict`/`request_reinvestigation`,
    which never touch the conclusions, confidence, or limitations recorded
    here.
    """

    conclusions: tuple[Conclusion, ...]
    confidence: int = Field(ge=0, le=100)
    limitations: tuple[str, ...] = ()
    review_status: VerdictReviewStatus = VerdictReviewStatus.AWAITING_REVIEW

    @property
    def is_awaiting_review(self) -> bool:
        return self.review_status is VerdictReviewStatus.AWAITING_REVIEW


class CaseFile(BaseModel):
    """The shared, structured state for one investigation.

    Each agent reads the fields it needs and writes only to its own section;
    see Architecture.md for the section-ownership contract.
    """

    mystery_text: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    suspect_profiles: list[SuspectProfile] = Field(default_factory=list)
    timeline: Timeline = Field(default_factory=Timeline)
    skeptic_reviews: list[SkepticReview] = Field(default_factory=list)
    revised_specialists: set[Specialist] = Field(default_factory=set)
    human_notes: list[str] = Field(default_factory=list)
    verdict: Verdict | None = None

    def accept_verdict(self) -> None:
        """Record a human Accept decision for the current verdict."""
        self._decide_verdict(VerdictReviewStatus.ACCEPTED)

    def reject_verdict(self) -> None:
        """Record a human Reject decision for the current verdict."""
        self._decide_verdict(VerdictReviewStatus.REJECTED)

    def request_reinvestigation(self, note: str) -> None:
        """Record a human re-investigation request and reset for a fresh pass.

        Preserves `mystery_text` and `evidence` so the Evidence Collector
        does not rerun. Appends `note` to `human_notes`, marks the current
        verdict's review status `REINVESTIGATION_REQUESTED` instead of
        clearing it, so the prior verdict and this request stay
        distinguishable, and resets `suspect_profiles`, `timeline`,
        `skeptic_reviews`, and `revised_specialists` so suspect analysis and
        timeline reconciliation can rerun clean, followed by a fresh Skeptic
        review and a new verdict.
        """
        stripped_note = note.strip()
        if not stripped_note:
            raise ValueError("A guidance note is required to request re-investigation.")
        self._decide_verdict(VerdictReviewStatus.REINVESTIGATION_REQUESTED)
        self.human_notes = [*self.human_notes, stripped_note]
        self.suspect_profiles = []
        self.timeline = Timeline()
        self.skeptic_reviews = []
        self.revised_specialists = set()

    def _decide_verdict(self, decision: VerdictReviewStatus) -> None:
        if self.verdict is None:
            raise VerdictReviewError("There is no verdict to review yet.")
        current_status = self.verdict.review_status
        if current_status is not VerdictReviewStatus.AWAITING_REVIEW:
            raise VerdictReviewError(
                "This verdict already has a recorded human decision "
                f"({current_status.value}); it cannot be changed."
            )
        self.verdict = self.verdict.model_copy(update={"review_status": decision})
