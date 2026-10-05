# Live provider validation

PCF's `Live Provider Validation` workflow performs a deliberately bounded paid validation against OpenAI and direct Anthropic.

## Safety boundary

- Paid execution is allowed only from `main`.
- Repository permissions are read-only.
- Provider secrets are injected only into the credential check and provider-specific paid steps; checkout, setup, dependency installation, and summary steps do not receive them.
- OpenAI and Anthropic run as independent jobs, so one provider cannot suppress evidence from the other.
- The workflow has fixed call counts and no user-configurable turn/repeat inputs.
- Each provider performs one one-call smoke, then a 24-turn × 3-arm × 3-repeat memory-placement matrix.
- Each paid session gets a fresh random nonce and the prompt never names the experimental arm, so the model is blind to whether it is running `front`, `tail`, or `placed`.
- Grading is deterministic from the scripted task answers; it is not a subjective human quality score.
- Logs print only sanitized metadata and aggregate benchmark statistics; model answers and credential values are not printed by summary steps.

## Anthropic workspace-scoped keys

Direct Anthropic runs use `PCF_ANTHROPIC_API_KEY` or `ANTHROPIC_API_KEY`. If the key is organization-level rather than scoped to one workspace, configure `PCF_ANTHROPIC_WORKSPACE_ID` (or `ANTHROPIC_WORKSPACE_ID`). The live wrapper sends it only as the `anthropic-workspace-id` header to `https://api.anthropic.com`.

Single-workspace keys continue to work without a workspace-ID setting. The workspace ID is never included in benchmark summaries.

## Triggering

The workflow supports manual dispatch on `main` and a main-only push trigger tied to `.github/run/live-provider-validation.trigger`. Incrementing that trigger file requests one bounded run without modifying the secret-consuming workflow.

The replicated 24-turn matrix is intended to measure whether the observed cache-economics and correctness effects survive longer sessions and run-to-run variation. It is still a bounded benchmark, not a universal provider-performance claim.
