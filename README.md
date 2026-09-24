# PCF 0.2

PCF is a reference implementation for portable context structure, provider-native prompt compilation, cache accounting, and conservative model routing.

## Status

This is an offline-tested prototype. The simulated engine can know cache warmth before execution. OpenAI and Anthropic adapters report provider warmth as `unknown` until a real response supplies usage. `compat_key` groups computation identity; it does not authorize KV-byte transfer, and no cross-family KV transport is implemented.

Calibration is a versioned, held-out empirical gate. A fitted scorer is not automatically eligible to route. Validation requires unique contexts disjoint from fitting data, a tail slice, error bounds, and quality/sample thresholds on both all selected examples and selected tail examples. Zero selected tail examples does not qualify a candidate. These are operational checks, not a guarantee of future quality.

```bash
python -m pip install -e '.[test]'
python -m pytest -q
ruff check .
python examples/demo.py
python scripts/live_smoke.py       # offline request-shape check
python scripts/live_smoke.py --run # paid calls only when keys are set
```

Schemas live under `pcf.schemas`. Hashes use RFC 8785 JCS and tagged SHA-256 domains. PCF 0.2 is a breaking wire/hash change from 0.1; see `CHANGELOG.md`.

External documents remain data, tool IDs/arguments are preserved, cache inspection is read-only, costs include write premiums, and unknown provider models require an explicit profile.

Cache identity and cumulative token counts use the same rendered input. Simulated billing counts canonical native JSON; provider billing remains an estimate until usage is returned. The revised accounting changes compiler cache keys and candidate fingerprints: old cache metadata and validation records must not be reused.

The Jev transport requires a direct HTTPS endpoint and rejects redirects. CI runs offline tests, lint, examples, a wheel installation check, and a private-session-link check on tracked text and new commit messages. The metadata check does not remove links from existing Git history or GitHub PR descriptions.

See `spec/SPEC.md` and `spec/REMEDIATION.md`.

License: MIT.
