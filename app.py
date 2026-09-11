"""Gradio presentation layer.

Renders mystery input, streams pipeline progress, and displays the
resulting case file. Contains no provider calls or investigation
reasoning; all of that lives behind `orchestrator.stream_investigation`.
"""

from __future__ import annotations

from collections.abc import Iterator

import gradio as gr

from case_file import CaseFile, Claim, ClaimStatus, Specialist
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
    lines.append("_This verdict is a proposal pending human review._")
    return "\n".join(lines)


def run_investigation(mystery_text: str) -> Iterator[tuple[str, str, str, str, str, str]]:
    llm = EnvLLMClient()
    transcript_lines: list[str] = []
    for event in stream_investigation(mystery_text, llm):
        if event.kind is InvestigationEventKind.VALIDATION_ERROR:
            yield event.message or "", "", "", "", "", ""
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

        start_button.click(
            fn=run_investigation,
            inputs=mystery_input,
            outputs=[
                transcript,
                evidence_table,
                suspect_profiles,
                timeline,
                skeptic_panel,
                verdict_panel,
            ],
        )
    return interface


if __name__ == "__main__":
    build_interface().launch()
