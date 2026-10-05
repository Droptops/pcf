# Bounded deterministic synthesis

PCF can synthesize a restricted deterministic expression from adjudicated trace examples without generating or executing source code.

## Grammar

The v1 synthesizer can infer:

- direct input projection
- `strip`
- `lower`
- `upper`
- `strip -> lower`
- `strip -> upper`
- stable object construction
- fixed-length array construction
- constants only when explicitly enabled

Anything outside that grammar fails closed.

## Anti-overfitting rules

- Training and held-out validation are separate inputs.
- A synthesized expression must exactly replay every training example.
- The resulting artifact must still pass the deterministic pipeline's independent held-out validation gate.
- If two equally simple expressions fit the same examples, synthesis fails as ambiguous instead of picking one.
- Constant-output synthesis is disabled by default.

## Security boundary

The synthesizer emits only the existing declarative deterministic IR. It has no path to `eval`, `exec`, imports, callbacks, filesystem access, networking, or arbitrary Python execution.

The artifact records `pcf.deterministic-synthesis.v1` as a dependency so the synthesis implementation is part of artifact identity and provenance.

## Privacy boundary

Synthesis evidence records counts, grammar version, input-path count and expression node count only. Raw training/validation examples are not embedded in registry metadata.
