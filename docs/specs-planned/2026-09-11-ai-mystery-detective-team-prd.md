# AI Mystery Detective Team — PRD

## Problem Statement

A single AI assistant analyzing a mystery can rush to a conclusion,
overlook evidence, repeat its own unverified assumptions, or state a
conclusion with confidence it hasn't earned. Someone using such an
assistant has no way to see *how* it got to an answer, no mechanism that
catches its own weak reasoning, and no natural point to sanity-check the
result before trusting it — especially when the evidence is complex or
the clues conflict with each other.

## Solution

Instead of one assistant doing everything, a team of specialized AI
"detective" agents each look at the mystery from one angle (evidence
extraction, suspect motive/opportunity, timeline consistency), a
dedicated Skeptic agent actively challenges any claim that isn't backed
by cited evidence, and a Lead Detective only produces a final verdict
once those challenges are resolved. The verdict, its evidence citations,
and a confidence score are shown to a human, who explicitly accepts,
rejects, or sends the case back for another pass — so the system never
hands over an unreviewed conclusion.

## User Stories

1. As a learner exploring multi-agent AI systems, I want to paste in a
   fictional mystery and watch specialized agents work on it, so that I
   can see multi-agent collaboration in action rather than reading about
   it abstractly.
2. As a user, I want the system to extract and structure the evidence
   from my mystery text first, so that every later step is grounded in
   an explicit, checkable list of facts rather than the raw prose.
3. As a user, I want a dedicated agent to analyze each suspect's motive
   and opportunity, so that I get a structured comparison instead of a
   vague narrative summary.
4. As a user, I want a dedicated agent to check the timeline for gaps or
   contradictions, so that inconsistencies in the mystery are caught
   instead of silently ignored.
5. As a user, I want a Skeptic agent to challenge any claim that isn't
   backed by cited evidence, so that the system doesn't quietly repeat a
   weak assumption as if it were fact.
6. As a user, I want the Skeptic to be able to send weak work back for
   one revision, so that flagged issues get a chance to be fixed before
   the final verdict is produced.
7. As a user, I want the Skeptic's revision loop to be capped, so that
   the system can't get stuck in an infinite back-and-forth.
8. As a user, I want the final verdict to cite the specific evidence IDs
   it's based on, so that I can trace every claim back to the source
   text instead of taking it on faith.
9. As a user, I want the final verdict to include a confidence score, so
   that I know how certain the system is rather than reading everything
   as equally sure.
10. As a user, I want to watch each detective's output appear live as it
    completes, so that I can follow the investigation's progress rather
    than waiting on a single opaque final answer.
11. As a user, I want to explicitly Accept, Reject, or request
    Re-investigation of the final verdict, so that I remain the final
    decision-maker rather than the system self-certifying its own
    conclusion.
12. As a user, I want to be able to request re-investigation with my own
    note attached, so that I can steer a second pass instead of starting
    over from scratch.
13. As a user, I want to paste in any fictional mystery I choose (no
    fixed built-in case), so that the tool is useful beyond a single
    canned demo.
14. As a beginner reading or extending this codebase, I want each
    agent's responsibility, inputs, and outputs to be simple and
    isolated, so that I can understand or modify one agent without
    having to understand the whole system.
15. As a developer maintaining this system, I want the LLM provider
    (Gemini, GPT, etc.) to be swappable behind one interface, so that
    changing providers doesn't require touching any agent's code.
16. As a developer, I want a malformed or invalid LLM response to
    trigger one retry and then a visible error, so that failures are
    surfaced honestly instead of the pipeline fabricating a result or
    hanging silently.

## Implementation Decisions

- **Shared state**: a single `CaseFile` data structure (evidence list,
  suspect profiles, timeline notes, skeptic flags, final verdict) is
  passed through the whole pipeline. Each agent reads what it needs and
  appends only to its own section — no agent mutates another agent's
  section.
- **Agent interface**: every agent exposes one operation, "run against a
  case file, return the updated case file," with a fixed system prompt
  and a required structured (JSON) output schema. Every claim an agent
  emits must include the evidence IDs it's grounded in.
- **Pipeline sequence**: Evidence Collector runs first (nothing else can
  run without its output) → Suspect Analyst and Timeline Reconciler run
  concurrently against the same evidence list → Skeptic reviews both
  outputs and may trigger exactly one revision round on the flagged
  agent(s) → Lead Detective synthesizes the final, possibly-revised
  outputs into the verdict.
- **LLM access**: one provider-agnostic call interface
  (`call_llm(prompt, system, response_schema)`); the concrete provider
  (Gemini or GPT) is selected via configuration, not hardcoded into any
  agent.
- **Retry policy**: an invalid/non-JSON LLM response gets exactly one
  retry with a stricter reformatting prompt; a second failure halts that
  step and surfaces a visible error rather than proceeding with
  fabricated output.
- **Skeptic loop cap**: the Skeptic may send flagged work back for
  revision exactly once per agent per run — no unbounded loops.
- **UI**: a Gradio interface with a mystery-text input, a live transcript
  panel that streams each agent's output as it completes, and a verdict
  panel with Accept / Reject / Re-investigate actions. Re-investigate
  restarts the pipeline from the parallel-analysis step with the human's
  note appended to the case file.
- **Human-in-the-loop is mandatory**: the pipeline's output is always
  presented as a proposal for human review, never auto-finalized.

## Testing Decisions

- Good tests here exercise each agent's *behavior contract* — given a
  case file in a known state, does the agent's output conform to the
  required schema and cite evidence IDs — not the internal prompt text
  or wording of any one LLM call.
- Each agent is unit-tested against a canned mystery fixture with the
  LLM call mocked, so tests are deterministic and don't depend on a live
  API.
- One end-to-end integration test runs the full pipeline against a fixed
  short mystery fixture and asserts: a verdict is produced, and every
  claim in the verdict cites at least one evidence ID.
- No prior art for these tests exists yet in this codebase — this is a
  new project — so these seams (the agent boundary and the full-pipeline
  boundary) are being established fresh here.

## Out of Scope

- Authentication, multi-user support, or persistence of past cases.
- Production-grade reliability/scaling — this is a learning/demo project.
- Any specific mystery content or dataset — the user always supplies
  their own text.
- Choosing between Gemini and GPT concretely — the interface is fixed by
  this PRD; the concrete provider wiring is an implementation-time
  choice behind that interface.

## Further Notes

This PRD builds directly on the design already discussed and recorded in
`docs/superpowers/specs/2026-09-11-ai-mystery-detective-team-design.md`
in this same directory — that document has the full architecture
diagram, file structure, and data-flow detail if deeper technical
context is needed alongside this PRD.
