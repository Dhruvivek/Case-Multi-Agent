# Agent Guide

This is a beginner-friendly Python/Gradio demonstration of a multi-agent
detective workflow. The repository is the source of truth: read the durable
docs before changing behavior, make meaningful decisions explicit, and leave
the project clearer than you found it.

## Read First

- `docs/project-brief/PS.md` — assignment context and expected deliverable.
- `docs/specs-planned/2026-09-11-ai-mystery-detective-team-prd.md` — product
  requirements and behavior contracts.
- `docs/specs-planned/2026-09-11-ai-mystery-detective-team-design.md` —
  intended architecture.
- `Architecture.md` — code layout and dependency boundaries.
- `CONTEXT.md` — project-specific language.
- `docs/agents/code-quality.md` and `docs/agents/python-gradio-conventions.md`
  — implementation conventions.

## Working Rules

- Prefer small, demonstrable vertical slices over broad rewrites.
- Build product behavior test-first: failing behavior test, minimal
  implementation, then refactor.
- Keep `CONTEXT.md` a glossary, not an implementation spec.
- Record hard-to-reverse architecture decisions in `docs/adr/` using the
  provided template.
- Put active work in `docs/specs-planned/` and archive shipped or superseded
  specs in `docs/specs-implemented/`.
- Agent code must depend on the provider-agnostic LLM wrapper, never a
  provider SDK directly.
- Do not add CI/CD pipelines, GitHub workflows, or issue-tracker automation
  unless the user explicitly requests them.

## Vendored Skills

The focused, repo-local skill set in `.agents/skills/` is sourced from
VirtuNode-dev/Starter-Pack. Use the applicable skill when planning, designing,
implementing, testing, debugging, reviewing, researching, or handing off
work. It intentionally excludes the Starter Pack's GitHub-flow and CI/CD
material.

## Definition of Done

- Requested observable behavior works.
- Relevant tests pass, or the reason they cannot run is stated.
- Documentation reflects changed product language, architecture, or workflow.
- Every final detective claim remains traceable to evidence IDs.
