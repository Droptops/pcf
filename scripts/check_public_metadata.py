#!/usr/bin/env python3
"""Reject private session links in tracked text and new commit messages.

PCF_BASE_SHA sets the exclusive commit boundary in CI. Without it, check HEAD.
Historical commits before that boundary are outside this prevention check.
Locations are reported without printing the matching private URL into CI logs.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

SESSION_LINK = re.compile(
    r"https?://(?:claude\.ai/(?:code/session_|chat/)|chatgpt\.com/(?:c/|g/[^/\s]+/c/)|chat\.openai\.com/c/)",
    re.IGNORECASE,
)


def git(*args):
    return subprocess.check_output(["git", *args]).decode("utf-8")


def locations(label, text):
    return [f"{label}:{i}: private session link" for i, line in enumerate(text.splitlines(), 1)
            if SESSION_LINK.search(line)]


def main():
    findings = []
    root = Path(git("rev-parse", "--show-toplevel").strip())
    for name in git("ls-files", "--full-name", "-z").split("\0"):
        if not name:
            continue
        path = root / name
        if not path.is_file() or path.is_symlink():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        findings.extend(locations(name, text))
    base = os.environ.get("PCF_BASE_SHA", "")
    if base and not re.fullmatch(r"[0-9a-fA-F]{40}", base):
        raise ValueError("PCF_BASE_SHA must be a complete Git commit SHA")
    commits = git("rev-list", f"{base}..HEAD").splitlines() if base and set(base) != {"0"} else ["HEAD"]
    for commit in commits:
        findings.extend(locations(f"commit {commit[:12]}", git("show", "-s", "--format=%B", commit)))
    if findings:
        print("\n".join(findings))
        return 1
    print("Tracked text and new commit messages contain no private session links.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
