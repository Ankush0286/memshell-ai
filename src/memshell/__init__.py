"""memshell: an in-memory filesystem with a shell-style interface for AI agents."""
from .agent_tool import INSTRUCTIONS, TOOL_NAME, handle_tool_call, tool_definition
from .shell import Result, Shell
from .vfs import FsError, VirtualFS

__all__ = [
    "Shell", "VirtualFS", "FsError", "Result",
    "tool_definition", "handle_tool_call", "INSTRUCTIONS", "TOOL_NAME",
]
__version__ = "0.1.0"
