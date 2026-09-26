# Release evidence

`v0.3.0rc1` and stable releases are blocked until a real four-arm production pilot finishes. Do not copy the
example into `production-pilot.json` and change `pending` to `pass` by hand.

After the preregistered pilot ends:

1. Analyze provider usage with the actual billed prices and conversation as the sampling unit.
2. Complete the arm-hidden review with at least two independent reviewers and adjudicate disagreements.
3. Keep raw prompts and reviews in the approved private evidence store. Record SHA-256 identities here; do not add
   private prompts, customer records, reviewer exports or internal links to this public repository.
4. Add `release/production-pilot.json` using the example shape. Record the frozen thresholds and observed values in
   each gate, then obtain distinct pilot-owner and independent-quality-reviewer approvals.
5. Run `python scripts/check_release_gate.py release/production-pilot.json`.

The tag workflow also requires protected `main`, an exact tag/project-version match, the full test suite, a built
wheel and sdist, and this evidence check. A missing, pending, synthetic or malformed evidence file blocks release.
