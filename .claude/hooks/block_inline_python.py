#!/usr/bin/env python3
"""
PreToolUse hook to warn (not block) when inline Python saves to paper-critical paths.

Only warns if:
1. Code is inline Python (heredoc)
2. AND output goes to paper/figures/ or similar critical paths

Does NOT block exploratory/debugging code.
"""

import sys
import json
import re


def main():
    hook_input = json.load(sys.stdin)

    if hook_input.get("tool_name") != "Bash":
        return 0

    command = hook_input.get("tool_input", {}).get("command", "")

    # Only care about Python heredocs
    if not re.search(r'python3?\s*<<|<<.*\n.*import\s', command, re.DOTALL):
        return 0

    # Check if output goes to critical paths
    critical_patterns = [
        r'paper/figures/',
        r'paper/.*\.pdf',
        r'paper/.*\.tex',
        r'fig_.*\.pdf',
        r'fig_.*\.png',
    ]

    for pattern in critical_patterns:
        if re.search(pattern, command):
            # Warn but don't block (exit 0, print to stdout)
            print(f"\n⚠️  REPRODUCIBILITY WARNING: Inline Python saving to paper artifact.\n"
                  f"Consider saving this code to scripts/analysis/<name>.py for reproducibility.\n"
                  f"(This is a warning, not a block - proceed if this is truly temporary.)\n")
            return 0

    # For non-critical paths, allow silently
    return 0


if __name__ == "__main__":
    sys.exit(main())
