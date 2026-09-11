# AI Mystery Detective Team Context

This project demonstrates how specialized AI agents can jointly analyze a
fictional mystery while keeping their reasoning inspectable and subject to a
human decision.

## Language

**Case file**:
The shared, structured state for one investigation: source mystery text,
evidence, analysis outputs, skeptic feedback, and the proposed verdict.
_Avoid_: session, database record

**Evidence**:
An ID-tagged fact or clearly labeled inference extracted from the supplied
mystery text and cited by downstream claims.
_Avoid_: clue text, source snippet

**Claim**:
A conclusion made by an agent that must cite one or more evidence IDs.
_Avoid_: fact, guess

**Revision round**:
The single permitted rerun of a flagged analyst after Skeptic feedback.
_Avoid_: retry loop

**Verdict**:
The Lead Detective's ranked, evidence-cited proposal with a confidence score;
it is not a decision until a human reviews it.
_Avoid_: final answer, resolution
