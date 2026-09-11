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

**Human decision**:
The Accept, Reject, or Request re-investigation outcome a person records
for a verdict that is awaiting human review. A verdict without one
remains a proposal.
_Avoid_: approval, final answer

**Guidance note**:
The non-empty note a human attaches when requesting re-investigation,
retained on the case file and available to every agent that reruns.
_Avoid_: prompt, comment

**Re-investigation**:
A restart of suspect analysis, timeline reconciliation, Skeptic review, and
verdict synthesis against the original mystery and evidence plus the
human's guidance note. The Evidence Collector does not rerun.
_Avoid_: retry, redo
