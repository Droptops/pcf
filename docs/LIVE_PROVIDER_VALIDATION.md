# Live provider validation

PCF's `Live Provider Validation` workflow performs a deliberately bounded paid validation against OpenAI and direct Anthropic.

## Safety boundary

- Paid execution is allowed only from `main`.
- Repository permissions are read-only.
- Provider secrets are injected only into the credential check and the provider-specific paid steps; checkout, setup, dependency installation, and summary steps do not receive them.
- The workflow has fixed call counts and no user-configurable turn/repeat inputs.
- A run performs one one-call smoke per provider, then a 12-turn × 3-arm × 1-repeat memory-placement matrix per provider.
- Logs print only sanitized metadata and aggregate benchmark statistics; model answers and credential values are not printed by the summary step.

## Triggering

The workflow supports manual dispatch on `main` and a main-only push trigger tied to `.github/run/live-provider-validation.trigger`. Incrementing that trigger file requests one bounded run without modifying the secret-consuming workflow.

The initial matrix is intentionally small. Larger replicated experiments should be added only after the provider smoke and bounded matrix are healthy and their spend/quality behavior is understood.
