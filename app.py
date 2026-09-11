"""Gradio presentation layer.

Renders mystery input, streams pipeline progress, and displays the
resulting case file. Contains no provider calls or investigation
reasoning; all of that lives behind `orchestrator.stream_investigation`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import gradio as gr

from case_file import (
    CaseFile,
    Claim,
    ClaimStatus,
    Specialist,
    VerdictReviewError,
    VerdictReviewStatus,
)
from llm_client import EnvLLMClient
from orchestrator import InvestigationEvent, InvestigationEventKind, stream_investigation

SPECIALIST_LABELS = {
    Specialist.SUSPECT_ANALYST: "Suspect Analyst",
    Specialist.TIMELINE_RECONCILER: "Timeline Reconciler",
}

PROGRESS_LABELS = {
    InvestigationEventKind.EVIDENCE_COLLECTION_STARTED: (
        "🔎 Evidence Collector is reading the case..."
    ),
    InvestigationEventKind.EVIDENCE_COLLECTION_COMPLETED: "✅ Evidence Collector finished.",
    InvestigationEventKind.SUSPECT_ANALYSIS_STARTED: "🕵️ Suspect Analyst is building profiles...",
    InvestigationEventKind.SUSPECT_ANALYSIS_COMPLETED: "✅ Suspect Analyst finished.",
    InvestigationEventKind.TIMELINE_RECONCILIATION_STARTED: (
        "🕰️ Timeline Reconciler is ordering events..."
    ),
    InvestigationEventKind.TIMELINE_RECONCILIATION_COMPLETED: "✅ Timeline Reconciler finished.",
    InvestigationEventKind.SKEPTIC_REVIEW_STARTED: (
        "🧐 Skeptic is reviewing the specialists' claims..."
    ),
    InvestigationEventKind.SKEPTIC_REVIEW_APPROVED: "✅ Skeptic approved the specialist claims.",
    InvestigationEventKind.SKEPTIC_REVIEW_REVISION_REQUESTED: (
        "⚠️ Skeptic requested a revision."
    ),
    InvestigationEventKind.SKEPTIC_REVIEW_EXHAUSTED: (
        "❌ Skeptic review exhausted; unresolved findings remain."
    ),
    InvestigationEventKind.LEAD_DETECTIVE_STARTED: (
        "🧑‍💼 Lead Detective is drafting the verdict..."
    ),
    InvestigationEventKind.LEAD_DETECTIVE_COMPLETED: "✅ Lead Detective finished.",
}


def _progress_label(event: InvestigationEvent) -> str:
    if event.kind is InvestigationEventKind.SPECIALIST_REVISION_STARTED:
        name = SPECIALIST_LABELS[event.specialist]
        return f"🔁 {name} is revising with reviewer feedback..."
    if event.kind is InvestigationEventKind.SPECIALIST_REVISION_COMPLETED:
        name = SPECIALIST_LABELS[event.specialist]
        return f"✅ {name} revision finished."
    return PROGRESS_LABELS[event.kind]


def render_case_file(case_file: CaseFile) -> str:
    if not case_file.evidence:
        return "_No evidence collected yet._"
    lines = ["| ID | Statement | Classification |", "| --- | --- | --- |"]
    lines.extend(
        f"| {item.id} | {item.statement} | {item.classification.value} |"
        for item in case_file.evidence
    )
    return "\n".join(lines)


def _render_claim(claim: Claim) -> str:
    if claim.status is ClaimStatus.UNKNOWN:
        return f"  - _unknown:_ {claim.statement}"
    citations = ", ".join(claim.evidence_ids)
    return f"  - {claim.statement} ({citations})"


def render_suspect_profiles(case_file: CaseFile) -> str:
    if not case_file.suspect_profiles:
        return "_No suspect profiles yet._"
    lines: list[str] = []
    for profile in case_file.suspect_profiles:
        lines.append(f"**{profile.suspect}**")
        lines.append("- Motive")
        lines.extend(_render_claim(claim) for claim in profile.motive)
        lines.append("- Opportunity")
        lines.extend(_render_claim(claim) for claim in profile.opportunity)
    return "\n".join(lines)


def render_timeline(case_file: CaseFile) -> str:
    if case_file.timeline.is_empty:
        return "_No timeline analysis yet._"

    lines = ["**Events**"]
    for event in case_file.timeline.events:
        citations = ", ".join(event.evidence_ids)
        if event.status is ClaimStatus.UNKNOWN:
            lines.append(
                f"- _uncertain / unordered:_ {event.statement} ({citations})"
            )
        else:
            lines.append(f"- {event.time} — {event.statement} ({citations})")

    lines.append("**Gaps and contradictions**")
    for issue in case_file.timeline.issues:
        citations = ", ".join(issue.evidence_ids)
        lines.append(f"- {issue.kind.value.title()}: {issue.statement} ({citations})")
    return "\n".join(lines)


def render_skeptic_reviews(case_file: CaseFile) -> str:
    if not case_file.skeptic_reviews:
        return "_No Skeptic review yet._"
    lines: list[str] = []
    for round_number, review in enumerate(case_file.skeptic_reviews, start=1):
        outcome_label = review.outcome.value.replace("_", " ").title()
        lines.append(f"**Round {round_number}: {outcome_label}**")
        if not review.findings:
            lines.append("- No findings.")
            continue
        for finding in review.findings:
            specialist_name = SPECIALIST_LABELS[finding.specialist]
            kind_label = finding.kind.value.replace("_", " ")
            lines.append(
                f"- {specialist_name} — {kind_label}: {finding.claim!r} — {finding.explanation}"
            )
    return "\n".join(lines)


REVIEW_STATUS_MESSAGES = {
    VerdictReviewStatus.AWAITING_REVIEW: "_This verdict is a proposal pending human review._",
    VerdictReviewStatus.ACCEPTED: "_Human decision: Accepted._",
    VerdictReviewStatus.REJECTED: "_Human decision: Rejected._",
}


def render_verdict(case_file: CaseFile) -> str:
    if case_file.verdict is None:
        return "_No verdict yet._"
    verdict = case_file.verdict
    lines = [f"**Confidence: {verdict.confidence}/100**", "", "**Ranked conclusions**"]
    for conclusion in verdict.conclusions:
        citations = ", ".join(conclusion.evidence_ids)
        lines.append(
            f"{conclusion.rank}. **{conclusion.suspect}** — {conclusion.explanation} ({citations})"
        )

    lines.append("")
    lines.append("**Limitations**")
    if verdict.limitations:
        lines.extend(f"- {limitation}" for limitation in verdict.limitations)
    else:
        lines.append("- None noted.")

    lines.append("")
    lines.append(REVIEW_STATUS_MESSAGES[verdict.review_status])
    return "\n".join(lines)


def _verdict_awaiting_review(case_file: CaseFile | None) -> bool:
    return bool(case_file is not None and case_file.verdict is not None and case_file.verdict.is_awaiting_review)


def sync_review_controls(case_file: CaseFile | None) -> tuple[dict, dict]:
    """Enable Accept/Reject only while the current verdict awaits review."""
    interactive = _verdict_awaiting_review(case_file)
    return gr.update(interactive=interactive), gr.update(interactive=interactive)


def disable_review_controls() -> tuple[dict, dict]:
    """Disable Accept/Reject immediately when a new investigation starts.

    Without this, buttons left enabled by a prior verdict would stay
    clickable for the whole duration of a new run, before any new verdict
    exists to review.
    """
    return sync_review_controls(None)


def _handle_review_decision(
    case_file: CaseFile | None, decide: Callable[[CaseFile], None]
) -> tuple[str, CaseFile | None, dict, dict]:
    """Apply a human decision, ignoring one that no longer applies.

    A decision on a missing verdict, or a repeat decision on a verdict that
    already has one, raises `VerdictReviewError`; that is a predictable
    no-op here rather than a crash or a silently overwritten decision.
    """
    if case_file is not None:
        try:
            decide(case_file)
        except VerdictReviewError:
            pass
    verdict_markdown = render_verdict(case_file) if case_file else "_No verdict yet._"
    accept_update, reject_update = sync_review_controls(case_file)
    return verdict_markdown, case_file, accept_update, reject_update


def handle_accept_verdict(
    case_file: CaseFile | None,
) -> tuple[str, CaseFile | None, dict, dict]:
    return _handle_review_decision(case_file, CaseFile.accept_verdict)


def handle_reject_verdict(
    case_file: CaseFile | None,
) -> tuple[str, CaseFile | None, dict, dict]:
    return _handle_review_decision(case_file, CaseFile.reject_verdict)


def run_investigation(
    mystery_text: str,
) -> Iterator[tuple[str, str, str, str, str, str, CaseFile | None]]:
    llm = EnvLLMClient()
    transcript_lines: list[str] = []
    for event in stream_investigation(mystery_text, llm):
        if event.kind is InvestigationEventKind.VALIDATION_ERROR:
            yield event.message or "", "", "", "", "", "", None
            return
        transcript_lines.append(_progress_label(event))
        evidence_markdown = render_case_file(event.case_file) if event.case_file else ""
        suspects_markdown = render_suspect_profiles(event.case_file) if event.case_file else ""
        timeline_markdown = render_timeline(event.case_file) if event.case_file else ""
        skeptic_markdown = render_skeptic_reviews(event.case_file) if event.case_file else ""
        verdict_markdown = render_verdict(event.case_file) if event.case_file else ""
        yield (
            "\n".join(transcript_lines),
            evidence_markdown,
            suspects_markdown,
            timeline_markdown,
            skeptic_markdown,
            verdict_markdown,
            event.case_file,
        )


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="AI Mystery Detective Team") as interface:
        gr.Markdown("# AI Mystery Detective Team")
        mystery_input = gr.Textbox(
            label="Mystery text",
            placeholder="Paste a fictional mystery to investigate...",
            lines=10,
        )
        start_button = gr.Button("Start investigation")
        transcript = gr.Markdown(label="Investigation transcript")
        evidence_table = gr.Markdown(label="Collected evidence")
        suspect_profiles = gr.Markdown(label="Suspect profiles")
        timeline = gr.Markdown(label="Timeline analysis")
        skeptic_panel = gr.Markdown(label="Skeptic review")
        verdict_panel = gr.Markdown(label="Verdict")
        with gr.Row():
            accept_button = gr.Button("Accept", interactive=False)
            reject_button = gr.Button("Reject", interactive=False)
        case_file_state = gr.State(None)

        start_button.click(
            fn=disable_review_controls,
            outputs=[accept_button, reject_button],
        ).then(
            fn=run_investigation,
            inputs=mystery_input,
            outputs=[
                transcript,
                evidence_table,
                suspect_profiles,
                timeline,
                skeptic_panel,
                verdict_panel,
                case_file_state,
            ],
        ).then(
            fn=sync_review_controls,
            inputs=case_file_state,
            outputs=[accept_button, reject_button],
        )

        accept_button.click(
            fn=handle_accept_verdict,
            inputs=case_file_state,
            outputs=[verdict_panel, case_file_state, accept_button, reject_button],
        )
        reject_button.click(
            fn=handle_reject_verdict,
            inputs=case_file_state,
            outputs=[verdict_panel, case_file_state, accept_button, reject_button],
        )
    return interface


if __name__ == "__main__":
    build_interface().launch()
