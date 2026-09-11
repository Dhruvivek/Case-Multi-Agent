<p align="center">
  <img src="assets/ai-mystery-detective-team-icon.png" width="180" alt="AI Mystery Detective Team icon — a fingerprint inside a magnifying glass" />
</p>

<h1 align="center">AI Mystery Detective Team</h1>

<p align="center">
  A beginner-friendly multi-agent AI project that turns fictional mysteries
  into evidence-based investigation proposals for human review.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-planning%20%26%20foundation-6366f1?style=for-the-badge" alt="Status: planning and foundation" />
  <img src="https://img.shields.io/badge/orchestration-Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Planned orchestration: Python" />
  <img src="https://img.shields.io/badge/interface-Gradio-F97316?style=for-the-badge" alt="Planned interface: Gradio" />
</p>

<p align="center">
  <a href="#the-idea">The idea</a> ·
  <a href="#planned-experience">Experience</a> ·
  <a href="#design-principles">Principles</a> ·
  <a href="#repository-guide">Docs</a>
</p>

> **Project status:** Planning and foundation complete — application
> implementation has not started yet.

Most AI assistants can give an answer; this project is about showing the work
behind one. A team of focused detective agents will extract evidence, inspect
suspects and timelines, challenge unsupported reasoning, and prepare a
cited verdict that a human must explicitly review.

## The idea

Complex mysteries invite confident guesses: clues conflict, timelines have
gaps, and a plausible story can be mistaken for a proven one. Instead of
asking one model to solve everything, this project separates the investigation
into accountable roles.

```text
Mystery text
    |
    v
Evidence Collector
    |
    +----------------------------+
    |                            |
    v                            v
Suspect Analyst          Timeline Reconciler
    |                            |
    +-------------+--------------+
                  v
          Skeptic / Challenger
                  |
          one bounded revision
                  v
            Lead Detective
                  |
                  v
       Human: Accept / Reject / Re-investigate
```

Every downstream claim is intended to cite the evidence IDs that support it.
The final verdict is a proposal, never an automatic decision.

## Planned experience

The future Gradio interface will let a user paste any fictional mystery and
follow the investigation as each role completes. It is designed to show:

- a structured, ID-tagged evidence list;
- motive and opportunity analysis for each suspect;
- timeline gaps and contradictions;
- a Skeptic's challenges to weak or uncited claims;
- a ranked final verdict with evidence citations and a confidence score; and
- human review controls to accept, reject, or request re-investigation with a
  note.

## Design principles

| Principle | What it means here |
| --- | --- |
| Evidence first | Nothing downstream analyzes the raw mystery before evidence is structured. |
| Clear roles | Each agent has one responsibility and an isolated part of the case file to update. |
| Productive skepticism | The Skeptic can request one revision per flagged agent, preventing endless loops. |
| Honest failure | Invalid LLM JSON gets one retry; a second failure is shown instead of being fabricated around. |
| Human authority | A verdict is reviewed by a person rather than self-certified by the system. |
| Provider flexibility | Agent code will depend on one LLM interface, keeping Gemini/GPT selection behind configuration. |

## Planned technology

- **Orchestration:** Python
- **Interface:** Gradio
- **Model access:** provider-agnostic LLM wrapper
- **Quality approach:** deterministic mocked unit tests plus an
  evidence-citation integration test

The exact provider SDK, dependency versions, and Gradio streaming mechanism
are intentionally deferred until implementation.

## Repository guide

| Location | Purpose |
| --- | --- |
| [PRD](docs/specs-planned/2026-09-11-ai-mystery-detective-team-prd.md) | User stories, behavioral contracts, and scope. |
| [Design](docs/specs-planned/2026-09-11-ai-mystery-detective-team-design.md) | Planned agent pipeline, state model, and test seams. |
| [Architecture](Architecture.md) | Intended code layout and module boundaries. |
| [Domain glossary](CONTEXT.md) | Precise meanings of project-specific terms. |
| [Agent guide](AGENTS.md) | Development rules and the focused local skill set. |
| [Documentation index](docs/README.md) | Durable project documentation structure. |

## Project status

This is currently a documentation-first foundation. There is no installable
package, runnable UI, configured LLM provider, or production deployment yet.
The next implementation slice is to establish the `CaseFile` model and LLM
wrapper, followed by the evidence collector and its deterministic tests.

## Contributing

Contributions should preserve the project's learning focus: keep agent
responsibilities small, cite evidence end-to-end, test behavior through public
boundaries, and document durable design decisions. See [AGENTS.md](AGENTS.md)
before starting work.

## License

No license has been selected yet. Do not assume permission to reuse or
redistribute this project until a license is added.
