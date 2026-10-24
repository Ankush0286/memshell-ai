"""The Shell: parses a command string and runs it against a VirtualFS."""
from __future__ import annotations

import fnmatch
import logging
import posixpath
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from .commands import COMMANDS, IO, UsageError
from .parser import Command, ParseError, Pipeline, Word, parse
from .vfs import FsError, VirtualFS

log = logging.getLogger("memshell.shell")


@dataclass
class Result:
    stdout: str
    stderr: str
    exit_code: int
    output: str  # stdout and stderr interleaved in the order they were produced


def _has_magic(seg: str) -> bool:
    return any(c in seg for c in "*?[")


class Shell:
    """One shell = one agent session (it has its own current directory).
    Several shells may share one VirtualFS."""

    def __init__(
        self,
        vfs: Optional[VirtualFS] = None,
        *,
        cwd: str = "/",
        max_output_chars: int = 20_000,
        **vfs_kwargs,
    ):
        self.vfs = vfs if vfs is not None else VirtualFS(**vfs_kwargs)
        self.cwd = cwd
        self.max_output_chars = max_output_chars
        self.commands: Dict[str, Tuple[Callable, str]] = dict(COMMANDS)

    # ------------------------------------------------------------ extension
    def register(self, name: str, fn: Callable, usage: str) -> None:
        """Add a custom command: ``fn(shell, args, stdin, io) -> exit_code``."""
        self.commands[name] = (fn, usage)

    # --------------------------------------------------------------- helpers
    def path(self, arg: str) -> str:
        return self.vfs.normalize(arg, self.cwd)

    def _glob(self, pattern: str) -> List[str]:
        absolute = pattern.startswith("/")
        segs = [s for s in pattern.split("/") if s]
        bases = [("/" if absolute else "", "/" if absolute else self.cwd)]
        for k, seg in enumerate(segs):
            last = k == len(segs) - 1
            nxt = []
            for disp, real in bases:
                if _has_magic(seg):
                    if not self.vfs.isdir(real):
                        continue
                    for e in self.vfs.listdir(real):
                        if e.name.startswith(".") and not seg.startswith("."):
                            continue
                        if fnmatch.fnmatchcase(e.name, seg) and (last or e.is_dir):
                            nxt.append((self._dj(disp, e.name), e.path))
                else:
                    target = self.vfs.normalize(seg, real)
                    if self.vfs.exists(target) and (last or self.vfs.isdir(target)):
                        nxt.append((self._dj(disp, seg), target))
            bases = nxt
        return [d for d, _ in bases]

    @staticmethod
    def _dj(disp: str, name: str) -> str:
        if disp == "":
            return name
        return disp + name if disp.endswith("/") else disp + "/" + name

    def _expand(self, word: Word) -> List[str]:
        if word.glob is None:
            return [word.value]
        return self._glob(word.glob) or [word.value]

    # ------------------------------------------------------------- execution
    def run(self, command: str) -> Result:
        try:
            steps = parse(command)
        except ParseError as e:
            msg = f"memshell: {e}\n"
            return Result("", msg, 2, msg)
        chunks: List[Tuple[int, str]] = []
        code = 0
        with self.vfs.lock:
            for step in steps:
                if step.connector == "&&" and code != 0:
                    continue
                if step.connector == "||" and code == 0:
                    continue
                c, code = self._run_pipeline(step.pipeline)
                chunks.extend(c)
        return Result(
            "".join(t for fd, t in chunks if fd == 1),
            "".join(t for fd, t in chunks if fd == 2),
            code,
            "".join(t for _, t in chunks),
        )

    def execute(self, command: str) -> str:
        """Run a command and return agent-ready text: output, truncation
        notice if needed, and the exit status when it is non-zero."""
        r = self.run(command)
        text = r.output
        if len(text) > self.max_output_chars:
            omitted = len(text) - self.max_output_chars
            text = text[: self.max_output_chars] + (
                f"\n[output truncated: {omitted} more characters omitted. "
                "Narrow the command: use head, tail, grep, or a smaller path.]\n"
            )
        if r.exit_code != 0:
            if text and not text.endswith("\n"):
                text += "\n"
            text += f"[exit status {r.exit_code}]\n"
        return text

    def _run_pipeline(self, pl: Pipeline):
        chunks: List[Tuple[int, str]] = []
        piped: Optional[str] = None
        code = 0
        for idx, cmd in enumerate(pl.commands):
            delivered, code = self._run_command(cmd, piped)
            if idx == len(pl.commands) - 1:
                chunks.extend(delivered)
            else:
                piped = "".join(t for fd, t in delivered if fd == 1)
                chunks.extend((fd, t) for fd, t in delivered if fd == 2)
        return chunks, code

    def _run_command(self, cmd: Command, piped: Optional[str]):
        argv: List[str] = []
        for w in cmd.argv:
            argv.extend(self._expand(w))

        sinks: Dict[int, tuple] = {1: ("term", 1), 2: ("term", 2)}
        stdin = cmd.stdin_text if cmd.stdin_text is not None else (piped or "")

        for r in cmd.redirects:
            label = r.target.value if r.target else ""
            try:
                if r.kind == "dup":
                    sinks[r.fd] = sinks[r.dup]  # type: ignore[index]
                    continue
                targets = self._expand(r.target)  # type: ignore[arg-type]
                if len(targets) != 1:
                    return [(2, f"memshell: {label}: ambiguous redirect\n")], 1
                t = targets[0]
                if r.kind == "in":
                    if t == "/dev/null":
                        stdin = ""
                    else:
                        stdin = self.vfs.read_bytes(self.path(t)).decode("utf-8")
                elif t == "/dev/null":
                    sinks[r.fd] = ("null",)
                else:
                    p = self.path(t)
                    if r.kind == "out" or not self.vfs.exists(p):
                        self.vfs.write_bytes(p, b"")  # open/truncate/create now
                    sinks[r.fd] = ("file", p, t)
            except FsError as e:
                return [(2, f"memshell: {label}: {e.reason}\n")], 1
            except UnicodeDecodeError:
                return [(2, f"memshell: {label}: binary file (not valid UTF-8)\n")], 1

        if not argv:
            return [], 0

        name = argv[0]
        io = IO()
        entry = self.commands.get(name)
        if entry is None:
            io.err(f"memshell: {name}: command not found (run 'help' to list commands)\n")
            code = 127
        else:
            try:
                code = entry[0](self, argv[1:], stdin, io) or 0
            except UsageError as e:
                io.err(f"{name}: {e}\n")
                code = 2
            except Exception as e:  # a bug in a command must not kill the agent loop
                log.exception("internal error in %s", name)
                io.err(f"memshell: internal error in '{name}': {type(e).__name__}: {e}\n")
                code = 1

        delivered: List[Tuple[int, str]] = []
        for fd, text in io.chunks:
            sink = sinks[fd]
            if sink[0] == "term":
                delivered.append((sink[1], text))
            elif sink[0] == "file":
                try:
                    self.vfs.write_bytes(sink[1], text.encode("utf-8"), append=True)
                except FsError as e:
                    delivered.append((2, f"memshell: {sink[2]}: {e.reason}\n"))
                    code = 1
        return delivered, code
