# Release evidence

`v0.3.0rc1` and stable releases are blocked until a real four-arm production pilot finishes. Do not copy the
example into `production-pilot.json` and change `pending` to `pass` by hand.

After the preregistered pilot ends:

1. Analyze provider usage with the actual billed prices and conversation as the sampling unit.
2. Complete the arm-hidden review with at least two independent reviewers and adjudicate disagreements.
3. Keep raw prompts and sensitive review material in the approved private evidence store. Create a sanitized,
   content-free release bundle under `release/evidence/`: `pilot.jsonl` with request/conversation IDs, arm,
   usage/outcome fields but no prompts; `blind-review.json` with the frozen review summary; and `analysis.json`
   with the frozen gate results.
4. Hash those exact sanitized files into `release/production-pilot.json`. The checker resolves every hash, verifies
   arm counts/failures from the bound pilot log, recomputes the cost/quality/blind-quality/latency/failure
   estimates from those rows, requires the blind-review summary and gate analysis to match, and then applies the
   preregistered numeric limits.
5. Obtain distinct pilot-owner and independent-quality-reviewer approvals. The tag job also uses the protected
   `production-release` GitHub environment; configure required reviewers on that environment before releasing.
6. Run `python scripts/check_release_gate.py release/production-pilot.json --artifact-dir release/evidence`.

The tag workflow also requires protected `main`, an exact tag/project-version match, the full test suite, a built
wheel and sdist, and this evidence check. A missing, pending, synthetic or malformed evidence file blocks release.
