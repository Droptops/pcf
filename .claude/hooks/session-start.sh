#!/bin/bash
# Install the Python package with its test tooling (ruff, pytest) and live SDKs (anthropic, openai) so CI checks
# and live scripts work in Claude Code on the web. Idempotent: pip skips what is already installed.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"
python -m pip install --quiet --disable-pip-version-check --root-user-action=ignore -e '.[test,live]'

# Put pip's script directory first so the pinned ruff wins over an older copy in ~/.local/bin.
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  echo "export PATH=\"$(python -c 'import sysconfig; print(sysconfig.get_path("scripts"))'):\$PATH\"" >> "$CLAUDE_ENV_FILE"
fi
