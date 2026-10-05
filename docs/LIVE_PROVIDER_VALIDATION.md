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
- Logs print only sanitized metadata, per-repeat summaries, aggregate benchmark statistics, and sanitized miss coordinates; raw model answers and credential values are not printed by summary steps.

## Synthetic-safe replicated fixture

The replicated workflow uses `scripts/live_safe_memory_matrix.py` and identifies its fixture as `synthetic-safe-v1`. It keeps the legacy harness's memory-module change rates, answer domains, placement logic, deterministic grading, and billed-unit accounting while replacing real-looking customer/profile fields with explicitly synthetic benchmark data.

The original live fixture is retained unchanged for historical reproduction. During the first 24-turn OpenAI replication, the provider rejected one accumulated legacy prompt with a `400 invalid_prompt` policy-filter response. PCF does not retry or evade that rejection. The replicated workflow instead uses the explicitly synthetic-safe fixture for both providers so cross-provider results come from the same benchmark content.

If a provider rejects a synthetic-safe call, the runner fails closed and reports only sanitized location metadata: provider, experimental arm, turn, HTTP status when available, and provider error code. Prompt text, model output, and secrets are not included in that diagnostic.

### Format diagnostics

The historical `violation` metric is preserved for continuity but is now split into two explicit causes in `synthetic-safe-v1`:

- `copied_filler`: the answer repeats one of the benchmark-history filler phrases;
- `too_long`: the answer exceeds the existing 60 estimated-token response limit (`ceil(characters / 4) > 60`).

`format_violation` is the OR of those two flags. `contract_repaired` records cases where the raw answer violates the format contract but the bounded answer value can still be deterministically extracted. Raw format compliance is never rewritten into a pass simply because deterministic extraction succeeds.

### Sanitized miss telemetry

Every incorrect answer now emits only the following diagnostic fields in the workflow summary:

`repeat | arm | turn | question_type | expected | extracted | stale_trap | copied_filler | too_long | format_violation`

The diagnostic never includes the raw prompt or raw model response. This makes residual correctness errors attributable without broadening the data exposed by the live workflow.

### Anthropic exact-value contract

For the synthetic-safe Anthropic run, the stable system prefix now includes a provider-specific response contract requiring exactly one answer value and no prose. Count questions require digits only; plan, contact-channel, and language questions require only the configured value. The contract also tells the model not to repeat synthetic history.

This is a response-shaping change only. It does not alter MemoryPlacer, memory freshness, cache placement, task answers, or the cost accounting. The harness still records raw format adherence separately. A deterministic bounded extractor produces `contract_output` as fallback evidence, so the benchmark can distinguish a correct-but-verbose model response from a true value-selection error.

## Anthropic workspace-scoped keys

Direct Anthropic runs use `PCF_ANTHROPIC_API_KEY` or `ANTHROPIC_API_KEY`. If the key is organization-level rather than scoped to one workspace, configure `PCF_ANTHROPIC_WORKSPACE_ID` (or `ANTHROPIC_WORKSPACE_ID`). The live wrapper sends it only as the `anthropic-workspace-id` header to `https://api.anthropic.com`.

Single-workspace keys continue to work without a workspace-ID setting. The workspace ID is never included in benchmark summaries.

## Triggering

The workflow supports manual dispatch on `main` and a main-only push trigger tied to `.github/run/live-provider-validation.trigger`. Incrementing that trigger file requests one bounded run without modifying the secret-consuming workflow.

The replicated 24-turn matrix is intended to measure whether the observed cache-economics and correctness effects survive longer sessions and run-to-run variation. It is still a bounded synthetic benchmark, not a universal provider-performance claim.
