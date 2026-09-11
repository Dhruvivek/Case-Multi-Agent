"""Gradio presentation layer.

Renders mystery input, streams pipeline progress, and displays the
resulting case file. Contains no provider calls or investigation
reasoning; all of that lives behind `orchestrator.stream_investigation`.
"""

from __future__ import annotations

from collections.abc import Iterator

import gradio as gr

from case_file import CaseFile, Claim, ClaimStatus
from llm_client import EnvLLMClient
from orchestrator import InvestigationEventKind, stream_investigation

PROGRESS_LABELS = {
    InvestigationEventKind.EVIDENCE_COLLECTION_STARTED: "🔎 Evidence Collector is reading the case...",
    InvestigationEventKind.EVIDENCE_COLLECTION_COMPLETED: "✅ Evidence Collector finished.",
    InvestigationEventKind.SUSPECT_ANALYSIS_STARTED: "🕵️ Suspect Analyst is building profiles...",
    InvestigationEventKind.SUSPECT_ANALYSIS_COMPLETED: "✅ Suspect Analyst finished.",
}


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


def run_investigation(mystery_text: str) -> Iterator[tuple[str, str, str]]:
    llm = EnvLLMClient()
    transcript_lines: list[str] = []
    for event in stream_investigation(mystery_text, llm):
        if event.kind is InvestigationEventKind.VALIDATION_ERROR:
            yield event.message or "", "", ""
            return
        transcript_lines.append(PROGRESS_LABELS[event.kind])
        evidence_markdown = render_case_file(event.case_file) if event.case_file else ""
        suspects_markdown = render_suspect_profiles(event.case_file) if event.case_file else ""
        yield "\n".join(transcript_lines), evidence_markdown, suspects_markdown


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

        start_button.click(
            fn=run_investigation,
            inputs=mystery_input,
            outputs=[transcript, evidence_table, suspect_profiles],
        )
    return interface


if __name__ == "__main__":
    build_interface().launch()
