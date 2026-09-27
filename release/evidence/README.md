# Sanitized release evidence bundle

This directory intentionally contains no production prompts, customer records, raw reviewer comments, credentials,
or private-system links.

A release remains blocked until these three files are added from the completed, frozen production pilot:

- `pilot.jsonl` — one content-free row per provider request, including conversation/request identity, assigned arm,
  provider usage buckets, success/failure and adjudicated quality fields needed by the release analysis.
- `blind-review.json` — the exact sanitized `blind_review` summary recorded in `production-pilot.json`.
- `analysis.json` — an object whose `gates` value is the exact frozen gate table recorded in
  `production-pilot.json`.

`scripts/check_release_gate.py` hashes these exact files and rejects a release when the hashes, arm counts,
failures, blind-review summary, gate analysis, or preregistered numeric thresholds do not match the release record.

Sensitive source evidence remains in the approved private evidence system. The sanitized bundle is the public,
content-free cryptographic binding between that review process and the release claim.
