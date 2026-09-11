# Architecture

The intended application is a Python/Gradio multi-agent investigation demo.
The product requirements and detailed pipeline are in
`docs/specs-planned/2026-09-11-ai-mystery-detective-team-*.md`.

## Intended File Structure

```text
/
├── AGENTS.md                         # Agent working agreement
├── Architecture.md                    # This guide
├── CONTEXT.md                         # Domain glossary
├── app.py                             # Gradio presentation layer
├── case_file.py                       # Shared domain state and schemas
├── llm_client.py                      # Provider-agnostic LLM boundary
├── orchestrator.py                    # Pipeline and streaming coordination
├── agents/                            # One isolated responsibility per agent
│   ├── collector.py
│   ├── suspect_analyst.py
│   ├── timeline_reconciler.py
│   ├── skeptic.py
│   └── lead_detective.py
├── tests/                             # Unit tests and a deterministic E2E test
├── docs/
│   ├── adr/
│   ├── agents/
│   ├── project-brief/
│   ├── qna/
│   ├── reference-material/
│   ├── specs-implemented/
│   ├── specs-planned/
│   └── templates/
└── .agents/skills/                    # Focused vendored development skills
```

## Dependency Boundaries

- `app.py` presents input, progress, verdict, and human review controls. It
  does not contain provider calls or investigation reasoning.
- `orchestrator.py` owns ordering, parallel work, the bounded revision loop,
  and streaming events.
- Modules under `agents/` read and update only their assigned CaseFile section
  through their `run(case_file)` contract.
- `llm_client.py` is the only layer allowed to import a concrete LLM provider.
- Tests mock the LLM boundary rather than prompts or provider SDKs.

## Where New Code Belongs

- Add a reusable investigation data type to `case_file.py`.
- Add a specialist's reasoning and output schema under `agents/`.
- Add pipeline control flow or stream events to `orchestrator.py`.
- Add UI-only behavior to `app.py`.
- Add deterministic behavior tests in `tests/`, mirroring the public module
  under test.
