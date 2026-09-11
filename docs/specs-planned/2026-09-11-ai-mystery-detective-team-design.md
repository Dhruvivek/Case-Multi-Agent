# AI Mystery Detective Team — Design

## Problem

A single AI assistant can rush to conclusions, overlook evidence, repeat
assumptions, or sound confident without enough proof. This project builds a
beginner-friendly multi-agent system where specialized AI "detectives"
analyze a fictional mystery, share findings through an orchestration
workflow, challenge weak assumptions, and produce an evidence-based verdict
for human review.

## Goals / Non-goals

- Goal: demonstrate multi-agent collaboration (specialization, sharing,
  challenge/verification, grounded conclusions) on a toy but realistic task.
- Goal: keep it simple enough for a beginner to read and extend.
- Non-goal: production-grade robustness, auth, persistence, or multi-user
  support. This is a demo/learning project.

## Architecture

```
Gradio UI (input textbox + live transcript + verdict panel)
        │
        ▼
Orchestrator (Python, sequential pipeline with one feedback loop)
        │
   1. Evidence Collector          (parses mystery → structured clue list)
        │
   ┌────┴────────┐
   ▼             ▼
2. Suspect        3. Timeline
   Analyst           Reconciler        (run in parallel on evidence)
   │                 │
   └────────┬────────┘
            ▼
4. Skeptic/Challenger   (flags unsupported claims → may bounce back once)
            ▼
5. Lead Detective        (final verdict, cites evidence IDs, confidence score)
            ▼
Human review (Accept / Reject / Re-investigate button in Gradio)
```

## Components

### LLM wrapper (`llm_client.py`)
- Single function: `call_llm(prompt, system, response_schema) -> dict`.
- Provider (Gemini or GPT) selected via env var / config, not hardcoded —
  agent code never talks to a provider SDK directly.
- On malformed/non-JSON output: one retry with a stricter reformatting
  prompt; if that also fails, raise a visible error (no silent failure).

### Shared state (`case_file.py`)
- Single `CaseFile` dataclass: evidence list, suspect profiles, timeline,
  skeptic flags, final verdict.
- Passed through the whole pipeline. Each agent reads the fields it needs
  and appends to its own section — agents never mutate another agent's
  section.

### Agents (`agents/`)
Each agent is a class with one method `run(case_file) -> case_file`, a
fixed system prompt, and a required JSON output schema. Every claim an
agent produces must cite the `evidence_ids` it is based on — this is the
mechanism that prevents ungrounded conclusions.

1. **Evidence Collector** (`collector.py`) — parses the raw mystery text
   into a structured, ID-tagged list of facts/clues, tagging each as
   observed fact vs. inference.
2. **Suspect Analyst** (`suspect_analyst.py`) — builds motive/opportunity
   profiles per suspect, citing evidence IDs.
3. **Timeline Reconciler** (`timeline_reconciler.py`) — checks temporal/
   logical consistency across clues, flags gaps or contradictions.
4. **Skeptic/Challenger** (`skeptic.py`) — reviews Analyst + Reconciler
   output, flags claims with weak or missing evidence citations. May send
   the case back to Analyst/Reconciler for one revision round.
5. **Lead Detective** (`lead_detective.py`) — synthesizes everything into a
   final ranked verdict with evidence citations and a confidence score.
   Only runs after the Skeptic has signed off (or after the one retry).

### Orchestrator (`orchestrator.py`)
- Runs the fixed sequence above.
- Analyst and Timeline Reconciler run concurrently (`asyncio.gather` or
  threads) since they're independent given the same evidence list.
- Skeptic gets exactly one feedback loop back to Analyst/Reconciler —
  prevents infinite loops.
- Yields/streams each agent's output as it completes, for the UI transcript.

### UI (`app.py`, Gradio)
- Textbox for pasting mystery text (no fixed/built-in mystery — user
  supplies their own).
- Live transcript pane showing each detective's turn as it happens
  (Gradio generator/streaming output).
- Final verdict panel: verdict text, confidence score, evidence citations,
  and human decision buttons (Accept / Reject / Request re-investigation).

## Data flow

1. User pastes mystery text into Gradio → orchestrator starts.
2. Collector runs first, produces the evidence list — nothing downstream
   runs without it.
3. Analyst + Reconciler run in parallel on the evidence list.
4. Skeptic reviews both outputs; if it flags unsupported claims, the
   flagged agent(s) re-run once with the Skeptic's feedback appended to
   their prompt.
5. Lead Detective synthesizes the (possibly revised) outputs into the
   final verdict.
6. UI displays the verdict; human clicks Accept / Reject / Re-investigate.
   Re-investigate restarts the pipeline from step 3 with the human's note
   appended to the case file.

## Error handling

- LLM call fails or returns invalid JSON → one retry with a stricter
  prompt, then a visible error in the transcript (pipeline halts, does not
  fabricate a result).
- Skeptic's feedback loop is capped at one retry per agent to avoid
  infinite loops.

## Testing

- Unit test each agent against a canned mystery fixture with a mocked
  `call_llm` (no real API calls in unit tests).
- One integration test running the full pipeline against a fixed short
  mystery fixture, asserting a verdict is produced and every claim in it
  cites at least one evidence ID.

## File structure

```
project/
  llm_client.py
  case_file.py
  agents/
    collector.py
    suspect_analyst.py
    timeline_reconciler.py
    skeptic.py
    lead_detective.py
  orchestrator.py
  app.py
  tests/
```

## Open decisions deferred to implementation

- Exact Gemini/GPT SDK call shape (wrapper interface is fixed above; the
  concrete provider call is an implementation detail behind it).
- Exact Gradio streaming mechanism (generator-based `gr.Blocks` update vs.
  polling) — pick whichever is simplest to implement correctly.
