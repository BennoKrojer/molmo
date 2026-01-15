#!/usr/bin/env python3
"""
PreToolUse hook to block inline Python execution via Bash heredocs.

This prevents running Python code inline without saving to a file first.
Code should be saved to scripts/ directory for reproducibility.

Exit codes:
  0 = allow (print any warnings to stdout)
  2 = block (print error message to stderr)
"""

import sys
import json
import re


def main():
    # Read hook input from stdin
    hook_input = json.load(sys.stdin)

    # Only process Bash tool calls
    if hook_input.get("tool_name") != "Bash":
        return 0

    command = hook_input.get("tool_input", {}).get("command", "")

    # Detect Python heredoc patterns
    heredoc_patterns = [
        r'python3?\s*<<',           # python3 << EOF
        r'<<\s*[\'"]?EOF[\'"]?\s*\n.*python',  # heredoc containing python
        r'<<\s*[\'"]?EOF[\'"]?\s*\n\s*(import|from|def |class )',  # heredoc with Python keywords
    ]

    for pattern in heredoc_patterns:
        if re.search(pattern, command, re.IGNORECASE | re.DOTALL):
            # Check if it's a significant amount of code (>10 lines)
            heredoc_match = re.search(r'<<\s*[\'"]?(\w+)[\'"]?\s*\n(.*?)\n\1', command, re.DOTALL)
            if heredoc_match:
                code_content = heredoc_match.group(2)
                line_count = len(code_content.strip().split('\n'))

                if line_count > 10:
                    # Block with error message
                    print(f"BLOCKED: Inline Python code ({line_count} lines) detected in Bash heredoc.\n"
                          f"For reproducibility, save code to a script file first:\n"
                          f"  1. Write to scripts/analysis/<descriptive_name>.py\n"
                          f"  2. Run: python scripts/analysis/<name>.py\n"
                          f"  3. Commit the script to git\n"
                          f"\nThis ensures all code is version-controlled and reproducible.",
                          file=sys.stderr)
                    return 2
                else:
                    # Allow but warn for small snippets
                    print(f"WARNING: Small inline Python ({line_count} lines). "
                          f"Consider saving to a file if this generates outputs.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
