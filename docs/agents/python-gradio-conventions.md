# Python and Gradio Conventions

Rules for the Python/Gradio application planned in this repository.

## Boundaries and Types

- Keep provider SDK imports and credentials inside `llm_client.py`.
- Use typed dataclasses or Pydantic models for `CaseFile` and structured agent
  outputs; validate LLM JSON before it enters shared state.
- Preserve evidence IDs end-to-end. Never replace cited IDs with uncited prose.
- Make agent operations explicit and side-effect-light: `run(case_file)` must
  return the updated case file or a visible error.

## Async and UI

- Run only the independent suspect and timeline analyses concurrently.
- Cap each Skeptic-requested revision at one per agent per investigation.
- Yield progress from the orchestrator; keep Gradio callbacks thin and focused
  on rendering state.
- Present verdicts as proposals that require Accept, Reject, or
  Re-investigate human action.

## Testing

- Mock `call_llm` at the LLM boundary; do not make network calls in unit tests.
- Test public behavior and schemas, including evidence citations and invalid
  response retry behavior.
- Keep fixtures short, readable, and fictional.
