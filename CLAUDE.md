# Working in this repository

- The repository is public. Never put private session links (claude.ai/code/session_..., chat links) in tracked
  files, commit messages, or PR descriptions, and do not add a `Claude-Session:` trailer to commits. CI enforces
  this for tracked text and new commit messages; run `PCF_BASE_SHA=$(git merge-base HEAD origin/main) python
  scripts/check_public_metadata.py` before pushing.
- Before pushing run the CI checks locally: `ruff check .`, `python -m pytest -q`, `python examples/demo.py`,
  `python scripts/live_smoke.py`.
- Live scripts (`scripts/live_smoke.py --run`, `scripts/live_memory_placement.py --run`) make paid API calls; run
  them only when asked. Commit raw results of runs quoted in docs under `results/`.
- Portable segment hashes and the PCF wire format are compatibility surfaces: say in CHANGELOG.md when a change
  affects what documents are accepted or how cache keys are computed.
