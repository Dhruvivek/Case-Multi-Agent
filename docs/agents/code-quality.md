# Code Quality

The stack-agnostic bar for "clean" code in this repo. This condenses the catalog used by the `/smell` skill down to what applies regardless of language or framework; see the skill itself for the full catalog and severity model. Stack-specific rules (TypeScript, React, a particular framework, etc.) belong in `docs/agents/stack-conventions.md`, not here.

## Before Opening a PR

Run `/smell` against the target branch on any non-trivial `feature`, `refactor`, or `bugfix` change. Treat any `BLOCKER` or `HIGH` finding as something to fix before requesting review, not after.

## Clean Code Baseline

- **Functions do one thing.** Small, one level of abstraction, no more than 3 arguments, no boolean flag arguments that make the function do two things (`CC.F3`, `CC.G30`, `CC.G34`).
- **Names reveal intent.** Long-lived or wide-scope names should be descriptive; short names are only fine in short scopes (`CC.N1`, `CC.N5`).
- **Duplication is the worst smell.** Extract shared logic rather than copy-adapt (`CC.G5`).
- **No magic numbers or dense one-liners.** Name constants and extract explanatory variables for anything non-obvious (`CC.G16`, `CC.G19`, `CC.G25`).
- **Encapsulate conditionals.** Extract compound booleans into a named predicate instead of inlining them (`CC.G28`).
- **Comments explain why, not what.** Delete obsolete, redundant, or commented-out code rather than leaving it (`CC.C2`, `CC.C3`, `CC.C5`).
- **No dead code.** Remove unused functions, branches, and symbols instead of leaving them "just in case" (`CC.F4`, `CC.G9`).

See `docs/agents/stack-conventions.md` for language- and framework-specific rules (type safety, rendering boundaries, data-fetching patterns, etc.), and the `/smell` skill's full catalog (`Clean Code`, `Gang of Four`, and any language-specific sections) for anything not covered here.
