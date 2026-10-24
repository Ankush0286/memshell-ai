"""Glue for exposing a Shell to an LLM agent as a single tool."""
from __future__ import annotations

from typing import Any, Dict

from .shell import Shell

TOOL_NAME = "shell"

INSTRUCTIONS = """\
You have a `shell` tool: a small, safe, in-memory Unix-like shell over a virtual \
filesystem. Every document you need lives there; start with `ls` or `tree`.

Reading:   ls, tree, cat [-n], head, tail, sed -n 'N,Mp', grep, find, wc, sort, uniq
Writing:   mkdir -p, touch, cp, mv, rm, echo, printf, tee, and redirects (> and >>)
Editing:   edit PATH --old 'exact text' --new 'replacement'   (add --all for every match)

Conventions:
- Create or overwrite a file with a quoted here-document, exactly like this:
    cat > /notes/todo.md << 'EOF'
    file content, written literally ($, quotes and backslashes are safe)
    EOF
  The terminator line (EOF) must be alone on its line.
- Change an existing file with `edit`. If it reports that the old text was not found or \
matched several places, read the file again and use more precise text.
- Single quotes keep text literal. Variables ($VAR), $(...), backticks, subshells and \
background jobs are NOT available.
- Big outputs are truncated: prefer head, tail, sed -n, grep.
- Run `help` for the full command list. Errors are explicit; read them and adjust.
"""


def tool_definition(name: str = TOOL_NAME) -> Dict[str, Any]:
    """Tool schema in the Anthropic Messages API format."""
    return {
        "name": name,
        "description": (
            "Run a shell command in a sandboxed in-memory filesystem. Supports ls, cat, grep, "
            "find, edit, redirects and here-documents. Not a real shell: see the instructions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command(s) to run. May span multiple lines (e.g. a here-document).",
                }
            },
            "required": ["command"],
        },
    }


def handle_tool_call(shell: Shell, tool_input: Dict[str, Any]) -> str:
    """Run one tool call and return the text for the tool_result block."""
    command = tool_input.get("command") if isinstance(tool_input, dict) else None
    if not isinstance(command, str) or not command.strip():
        return "memshell: the 'command' field must be a non-empty string\n[exit status 2]"
    return shell.execute(command) or "(no output)"
