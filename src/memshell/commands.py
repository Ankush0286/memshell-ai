"""Built-in commands. Each one imitates its real counterpart closely enough
that models use it correctly, and fails loudly on anything unsupported.

A command is ``fn(shell, args, stdin, io) -> exit_code``.
"""
from __future__ import annotations

import fnmatch
import posixpath
import re
from typing import Callable, Dict, List, Optional, Tuple

from .vfs import ENOENT, ENOTDIR, FsError


class UsageError(Exception):
    """Bad options/arguments; reported as ``<cmd>: <message>`` with exit code 2."""


class IO:
    """Collects a command's output as an ordered list of (fd, text) chunks."""

    def __init__(self) -> None:
        self.chunks: List[Tuple[int, str]] = []

    def out(self, text: str) -> None:
        if text:
            self.chunks.append((1, text))

    def err(self, text: str) -> None:
        if text:
            self.chunks.append((2, text))


COMMANDS: Dict[str, Tuple[Callable, str]] = {}


def command(name: str, usage: str):
    def deco(fn):
        COMMANDS[name] = (fn, usage)
        return fn

    return deco


# ------------------------------------------------------------------ helpers
def parse_opts(cmd, args, bools="", values="", long=None):
    """Tiny getopt: returns (opts, positionals). Supports -abc, -n5, -n 5,
    --long, --long=value and '--'."""
    long = long or {}
    opts: Dict[str, object] = {}
    pos: List[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--":
            pos.extend(args[i + 1:])
            break
        if a.startswith("--"):
            name, eq, val = a[2:].partition("=")
            if name not in long:
                raise UsageError(f"unrecognized option '--{name}'")
            ch = long[name]
            if ch in values:
                if not eq:
                    i += 1
                    if i >= len(args):
                        raise UsageError(f"option '--{name}' requires an argument")
                    val = args[i]
                opts[ch] = val
            else:
                opts[ch] = True
        elif a.startswith("-") and len(a) > 1:
            j = 1
            while j < len(a):
                ch = a[j]
                if ch in bools:
                    opts[ch] = True
                    j += 1
                elif ch in values:
                    rest = a[j + 1:]
                    if rest:
                        opts[ch] = rest
                    else:
                        i += 1
                        if i >= len(args):
                            raise UsageError(f"option requires an argument -- '{ch}'")
                        opts[ch] = args[i]
                    break
                else:
                    raise UsageError(f"invalid option -- '{ch}'")
        else:
            pos.append(a)
        i += 1
    return opts, pos


def _fail(io: IO, cmd: str, arg: str, e: FsError) -> None:
    io.err(f"{cmd}: {arg}: {e.reason}\n")


def _read_text(sh, io: IO, cmd: str, arg: str) -> Optional[str]:
    try:
        data = sh.vfs.read_bytes(sh.path(arg))
    except FsError as e:
        _fail(io, cmd, arg, e)
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        io.err(f"{cmd}: {arg}: binary file (not valid UTF-8)\n")
        return None


def _write_text(sh, io: IO, cmd: str, arg: str, text: str, append: bool = False) -> bool:
    try:
        sh.vfs.write_bytes(sh.path(arg), text.encode("utf-8"), append=append)
        return True
    except FsError as e:
        _fail(io, cmd, arg, e)
        return False


def _lines(text: str) -> List[str]:
    if text == "":
        return []
    parts = text.split("\n")
    if parts[-1] == "":
        parts.pop()
    return parts


def _join_label(label: str, name: str) -> str:
    if label.endswith("/"):
        return label + name
    return label + "/" + name


def _int(cmd: str, val, what: str = "number") -> int:
    try:
        n = int(val)
        if n < 0:
            raise ValueError
        return n
    except (TypeError, ValueError):
        raise UsageError(f"invalid {what}: '{val}'")


def _norm_count_args(args: List[str]) -> List[str]:
    """Turn the legacy `head -5` into `head -n 5`."""
    out = []
    for a in args:
        if re.fullmatch(r"-\d+", a):
            out += ["-n", a[1:]]
        else:
            out.append(a)
    return out


_ESC = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"', "a": "\a"}


def _unescape(s: str) -> str:
    return re.sub(r"\\(.)", lambda m: _ESC.get(m.group(1), m.group(0)), s)


# ----------------------------------------------------------------- listing
@command("ls", "ls [-a] [-l] [-R] [-F] [-1] [-d] [PATH...]   list directory contents")
def ls(sh, args, stdin, io):
    o, paths = parse_opts("ls", args, bools="alRF1hd")
    paths = paths or ["."]
    code = 0

    def fmt(e, name=None):
        name = name if name is not None else e.name
        if "F" in o and e.is_dir:
            name += "/"
        if "l" in o:
            mode = "drwxr-xr-x" if e.is_dir else "-rw-r--r--"
            return f"{mode} {e.size:>8} {name}"
        return name

    files, dirs = [], []
    for a in paths:
        try:
            p = sh.path(a)
            st = sh.vfs.stat(p)
        except FsError as e:
            io.err(f"ls: cannot access '{a}': {e.reason}\n")
            code = 2
            continue
        (dirs if st.is_dir and "d" not in o else files).append((a, p, st))

    for a, p, st in files:
        io.out(fmt(st, a) + "\n")

    show_headers = len(paths) > 1 or "R" in o
    first = not files

    def list_dir(label, p, header):
        nonlocal first
        entries = sh.vfs.listdir(p)
        if "a" not in o:
            entries = [e for e in entries if not e.name.startswith(".")]
        if not first:
            io.out("\n")
        first = False
        if header:
            io.out(f"{label}:\n")
        for e in entries:
            io.out(fmt(e) + "\n")
        if "R" in o:
            for e in entries:
                if e.is_dir:
                    list_dir(_join_label(label, e.name), e.path, True)

    for a, p, st in dirs:
        list_dir(a, p, show_headers)
    return code


@command("tree", "tree [-a] [-d] [-L LEVELS] [PATH]   show the directory tree")
def tree(sh, args, stdin, io):
    o, pos = parse_opts("tree", args, bools="ad", values="L")
    maxd = _int("tree", o["L"], "level") if "L" in o else None
    a = pos[0] if pos else "."
    try:
        p = sh.path(a)
        if not sh.vfs.isdir(p):
            raise FsError(ENOTDIR if sh.vfs.isfile(p) else ENOENT, p)
    except FsError as e:
        _fail(io, "tree", a, e)
        return 2
    counts = [0, 0]
    io.out(a + "\n")

    def rec(path, prefix, depth):
        entries = sh.vfs.listdir(path)
        if "a" not in o:
            entries = [e for e in entries if not e.name.startswith(".")]
        if "d" in o:
            entries = [e for e in entries if e.is_dir]
        for k, e in enumerate(entries):
            last = k == len(entries) - 1
            io.out(prefix + ("└── " if last else "├── ") + e.name + "\n")
            if e.is_dir:
                counts[0] += 1
                if maxd is None or depth < maxd:
                    rec(e.path, prefix + ("    " if last else "│   "), depth + 1)
            else:
                counts[1] += 1

    rec(p, "", 1)
    d, f = counts
    summary = f"{d} director{'y' if d == 1 else 'ies'}"
    if "d" not in o:
        summary += f", {f} file{'' if f == 1 else 's'}"
    io.out("\n" + summary + "\n")
    return 0


@command("find", "find [PATH...] [-name GLOB] [-iname GLOB] [-type f|d] [-maxdepth N] [-mindepth N]")
def find(sh, args, stdin, io):
    starts: List[str] = []
    i = 0
    while i < len(args) and not args[i].startswith("-"):
        starts.append(args[i])
        i += 1
    starts = starts or ["."]
    name = iname = typ = None
    maxd: Optional[int] = None
    mind = 0
    while i < len(args):
        t = args[i]
        if t == "-print":
            i += 1
            continue
        if t in ("-name", "-iname", "-type", "-maxdepth", "-mindepth"):
            if i + 1 >= len(args):
                raise UsageError(f"missing argument to '{t}'")
            v = args[i + 1]
            if t == "-name":
                name = v
            elif t == "-iname":
                iname = v.lower()
            elif t == "-type":
                if v not in ("f", "d"):
                    raise UsageError("-type supports only 'f' and 'd'")
                typ = v
            elif t == "-maxdepth":
                maxd = _int("find", v)
            else:
                mind = _int("find", v)
            i += 2
            continue
        raise UsageError(
            f"unsupported expression '{t}' (supported: -name -iname -type -maxdepth -mindepth)"
        )

    code = 0

    def matches(nm, is_dir):
        if typ == "f" and is_dir:
            return False
        if typ == "d" and not is_dir:
            return False
        if name is not None and not fnmatch.fnmatchcase(nm, name):
            return False
        if iname is not None and not fnmatch.fnmatchcase(nm.lower(), iname):
            return False
        return True

    def rec(label, path, depth):
        for e in sh.vfs.listdir(path):
            lab = _join_label(label, e.name)
            if depth >= mind and matches(e.name, e.is_dir):
                io.out(lab + "\n")
            if e.is_dir and (maxd is None or depth < maxd):
                rec(lab, e.path, depth + 1)

    for a in starts:
        try:
            p = sh.path(a)
            st = sh.vfs.stat(p)
        except FsError as e:
            io.err(f"find: '{a}': {e.reason}\n")
            code = 1
            continue
        if mind == 0 and matches(posixpath.basename(p) or "/", st.is_dir):
            io.out(a + "\n")
        if st.is_dir and (maxd is None or maxd > 0):
            rec(a, p, 1)
    return code


# ----------------------------------------------------------------- reading
@command("cat", "cat [-n] [FILE...]   print files (use '-' or no FILE for stdin)")
def cat(sh, args, stdin, io):
    o, files = parse_opts("cat", args, bools="n")
    files = files or ["-"]
    code, n = 0, 0
    for f in files:
        text = stdin if f == "-" else _read_text(sh, io, "cat", f)
        if text is None:
            code = 1
            continue
        if "n" in o:
            for line in text.splitlines(True):
                n += 1
                io.out(f"{n:>6}\t{line}")
        else:
            io.out(text)
    return code


@command("head", "head [-n N] [FILE...]   first N lines (default 10)")
def head(sh, args, stdin, io):
    return _head_tail(sh, args, stdin, io, "head")


@command("tail", "tail [-n N | -n +K] [FILE...]   last N lines (default 10), or from line K")
def tail(sh, args, stdin, io):
    return _head_tail(sh, args, stdin, io, "tail")


def _head_tail(sh, args, stdin, io, cmd):
    o, files = parse_opts(cmd, _norm_count_args(args), bools="q", values="n", long={"lines": "n"})
    raw = str(o.get("n", "10"))
    plus = cmd == "tail" and raw.startswith("+")
    n = _int(cmd, raw[1:] if plus else raw, "number of lines")
    files = files or ["-"]
    code = 0
    for k, f in enumerate(files):
        text = stdin if f == "-" else _read_text(sh, io, cmd, f)
        if text is None:
            code = 1
            continue
        if len(files) > 1 and "q" not in o:
            io.out(("\n" if k else "") + f"==> {'standard input' if f == '-' else f} <==\n")
        lines = text.splitlines(True)
        if cmd == "head":
            sel = lines[:n]
        elif plus:
            sel = lines[max(n - 1, 0):]
        else:
            sel = lines[-n:] if n > 0 else []
        io.out("".join(sel))
        if sel and not sel[-1].endswith("\n"):
            io.out("\n")
    return code


@command("sed", "sed -n 'N,Mp' [FILE...]   print a line range ('N,$p' = to the end). To change files use `edit`.")
def sed(sh, args, stdin, io):
    o, pos = parse_opts("sed", args, bools="n")
    hint = (
        "only  sed -n 'N,Mp' [FILE...]  is supported "
        "(to modify a file use: edit PATH --old TEXT --new TEXT)"
    )
    if "n" not in o or not pos:
        raise UsageError(hint)
    m = re.fullmatch(r"(\d+)(?:,(\d+|\$))?p", pos[0].strip())
    if not m:
        raise UsageError(hint)
    start = max(int(m.group(1)), 1)
    end = m.group(2)
    stop = start if end is None else (None if end == "$" else int(end))
    code = 0
    for f in pos[1:] or ["-"]:
        text = stdin if f == "-" else _read_text(sh, io, "sed", f)
        if text is None:
            code = 1
            continue
        lines = text.splitlines(True)
        sel = lines[start - 1:stop]
        io.out("".join(sel))
        if sel and not sel[-1].endswith("\n"):
            io.out("\n")
    return code


@command("wc", "wc [-l] [-w] [-c] [FILE...]   count lines, words, bytes")
def wc(sh, args, stdin, io):
    o, files = parse_opts("wc", args, bools="lwc")
    show = [k for k in "lwc" if k in o] or ["l", "w", "c"]
    rows = []
    code = 0
    for f in files or ["-"]:
        text = stdin if f == "-" else _read_text(sh, io, "wc", f)
        if text is None:
            code = 1
            continue
        counts = {"l": text.count("\n"), "w": len(text.split()), "c": len(text.encode("utf-8"))}
        rows.append(([counts[k] for k in show], "" if f == "-" else f))
    if len(rows) > 1:
        rows.append(([sum(r[0][j] for r in rows) for j in range(len(show))], "total"))
    width = max([len(str(v)) for r in rows for v in r[0]] or [1])
    for vals, label in rows:
        io.out(" ".join(str(v).rjust(width) for v in vals) + (" " + label if label else "") + "\n")
    return code


def _bre_to_py(p: str) -> str:
    """Translate a POSIX basic regex (grep default) to Python syntax."""
    out, i = [], 0
    while i < len(p):
        c = p[i]
        if c == "\\" and i + 1 < len(p):
            nxt = p[i + 1]
            out.append(nxt if nxt in "|(){}+?" else "\\" + nxt)
            i += 2
        elif c in "|(){}+?":
            out.append("\\" + c)
            i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


@command(
    "grep",
    "grep [-i] [-v] [-n] [-c] [-l] [-F] [-E] [-w] [-r] [-H|-h] [-A N] [-B N] [-C N] PATTERN [FILE...]",
)
def grep(sh, args, stdin, io):
    o, pos = parse_opts(
        "grep",
        args,
        bools="ivnclFEwrRHh",
        values="eABC",
        long={
            "ignore-case": "i", "invert-match": "v", "line-number": "n", "count": "c",
            "files-with-matches": "l", "fixed-strings": "F", "extended-regexp": "E",
            "word-regexp": "w", "recursive": "r", "with-filename": "H", "no-filename": "h",
        },
    )
    if "e" in o:
        pattern, files = str(o["e"]), pos
    else:
        if not pos:
            raise UsageError("missing PATTERN")
        pattern, files = pos[0], pos[1:]
    after = _int("grep", o.get("A", o.get("C", 0)), "context length")
    before = _int("grep", o.get("B", o.get("C", 0)), "context length")
    if "F" in o:
        expr = re.escape(pattern)
    elif "E" in o:
        expr = pattern
    else:
        expr = _bre_to_py(pattern)
    if "w" in o:
        expr = f"(?<!\\w)(?:{expr})(?!\\w)"
    try:
        rx = re.compile(expr, re.IGNORECASE if "i" in o else 0)
    except re.error as e:
        io.err(f"grep: invalid regular expression: {e}\n")
        return 2

    recursive = "r" in o or "R" in o
    defaulted = False
    if not files:
        if recursive:
            files, defaulted = ["."], True
        else:
            files = ["-"]
    errors = False
    targets: List[Tuple[str, Optional[str], bool]] = []  # (label, path, is_stdin)
    for a in files:
        if a == "-":
            targets.append(("(standard input)", None, True))
            continue
        try:
            p = sh.path(a)
            st = sh.vfs.stat(p)
        except FsError as e:
            io.err(f"grep: {a}: {e.reason}\n")
            errors = True
            continue
        if st.is_dir:
            if not recursive:
                io.err(f"grep: {a}: Is a directory\n")
                errors = True
                continue
            for path, is_dir in sh.vfs.walk(p):
                if not is_dir:
                    rel = posixpath.relpath(path, p)
                    targets.append((rel if defaulted else _join_label(a, rel), path, False))
        else:
            targets.append((a, p, False))

    show_name = ("H" in o) or (("h" not in o) and (len(files) > 1 or recursive))
    matched_any = False
    printed_group = False
    for label, path, is_stdin in targets:
        if is_stdin:
            text = stdin
        else:
            try:
                data = sh.vfs.read_bytes(path)
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                if not recursive:
                    io.err(f"grep: {label}: binary file (not valid UTF-8)\n")
                    errors = True
                continue
            except FsError as e:
                io.err(f"grep: {label}: {e.reason}\n")
                errors = True
                continue
        lines = _lines(text)
        hits = [i for i, l in enumerate(lines) if bool(rx.search(l)) != ("v" in o)]
        if hits:
            matched_any = True
        if "c" in o:
            io.out((f"{label}:" if show_name else "") + f"{len(hits)}\n")
            continue
        if "l" in o:
            if hits:
                io.out(label + "\n")
            continue

        def emit(j, sep):
            pre = f"{label}{sep}" if show_name else ""
            if "n" in o:
                pre += f"{j + 1}{sep}"
            io.out(pre + lines[j] + "\n")

        if not hits:
            continue
        if after == 0 and before == 0:
            for j in hits:
                emit(j, ":")
            continue
        hitset = set(hits)
        ranges: List[List[int]] = []
        for j in hits:
            lo, hi = max(0, j - before), min(len(lines) - 1, j + after)
            if ranges and lo <= ranges[-1][1] + 1:
                ranges[-1][1] = max(ranges[-1][1], hi)
            else:
                ranges.append([lo, hi])
        for lo, hi in ranges:
            if printed_group:
                io.out("--\n")
            printed_group = True
            for j in range(lo, hi + 1):
                emit(j, ":" if j in hitset else "-")
    if errors:
        return 2
    return 0 if matched_any else 1


@command("sort", "sort [-r] [-n] [-u] [FILE...]")
def sort_(sh, args, stdin, io):
    o, files = parse_opts("sort", args, bools="rnu")
    lines: List[str] = []
    code = 0
    for f in files or ["-"]:
        text = stdin if f == "-" else _read_text(sh, io, "sort", f)
        if text is None:
            code = 2
            continue
        lines += _lines(text)

    def key(s):
        if "n" in o:
            m = re.match(r"\s*(-?\d+(?:\.\d+)?)", s)
            return (float(m.group(1)) if m else 0.0, s)
        return s

    lines.sort(key=key, reverse="r" in o)
    if "u" in o:
        seen, uniq = set(), []
        for l in lines:
            if l not in seen:
                seen.add(l)
                uniq.append(l)
        lines = uniq
    io.out("".join(l + "\n" for l in lines))
    return code


@command("uniq", "uniq [-c] [FILE]   collapse adjacent duplicate lines")
def uniq(sh, args, stdin, io):
    o, files = parse_opts("uniq", args, bools="c")
    f = files[0] if files else "-"
    text = stdin if f == "-" else _read_text(sh, io, "uniq", f)
    if text is None:
        return 1
    prev, count = None, 0

    def flush():
        if prev is not None:
            io.out((f"{count:>7} " if "c" in o else "") + prev + "\n")

    for l in _lines(text):
        if l == prev:
            count += 1
        else:
            flush()
            prev, count = l, 1
    flush()
    return 0


# ----------------------------------------------------------------- writing
@command("echo", "echo [-n] [-e] [TEXT...]")
def echo(sh, args, stdin, io):
    nl, esc, i = True, False, 0
    while i < len(args) and re.fullmatch(r"-[neE]+", args[i]):
        for ch in args[i][1:]:
            if ch == "n":
                nl = False
            elif ch == "e":
                esc = True
            else:
                esc = False
        i += 1
    text = " ".join(args[i:])
    if esc:
        text = _unescape(text)
    io.out(text + ("\n" if nl else ""))
    return 0


@command("printf", "printf FORMAT [ARG...]   supports %s %d %% and \\n \\t \\\\")
def printf(sh, args, stdin, io):
    if not args:
        raise UsageError("usage: printf FORMAT [ARG...]")
    fmt, vals = args[0], list(args[1:])
    spec = re.compile(r"%(.)", re.S)
    out: List[str] = []
    while True:
        used = 0

        def sub(m):
            nonlocal used
            c = m.group(1)
            if c == "%":
                return "%"
            if c not in "sd":
                raise UsageError(f"unsupported format %{c} (only %s %d %%)")
            v = vals.pop(0) if vals else None
            used += 1
            if c == "d":
                try:
                    return str(int(v)) if v is not None else "0"
                except ValueError:
                    raise UsageError(f"invalid number '{v}'")
            return v if v is not None else ""

        out.append(_unescape(spec.sub(sub, fmt)))
        if not vals or used == 0:
            break
    io.out("".join(out))
    return 0


@command("tee", "tee [-a] FILE...   copy stdin to files and to stdout")
def tee(sh, args, stdin, io):
    o, files = parse_opts("tee", args, bools="a")
    code = 0
    for f in files:
        if not _write_text(sh, io, "tee", f, stdin, append="a" in o):
            code = 1
    io.out(stdin)
    return code


@command("edit", "edit [--all] PATH --old TEXT --new TEXT   replace exact text; fails if TEXT is missing or ambiguous")
def edit(sh, args, stdin, io):
    o, pos = parse_opts(
        "edit", args, bools="a", values="on", long={"all": "a", "old": "o", "new": "n"}
    )
    if len(pos) != 1 or "o" not in o or "n" not in o:
        raise UsageError("usage: edit [--all] PATH --old TEXT --new TEXT")
    path, old, new = pos[0], str(o["o"]), str(o["n"])
    if old == "":
        io.err("edit: --old must not be empty (to create or overwrite a file use a redirect)\n")
        return 1
    text = _read_text(sh, io, "edit", path)
    if text is None:
        return 1
    count = text.count(old)
    if count == 0:
        hint = ""
        if old.strip() and old.strip() in text:
            hint = " (found when ignoring leading/trailing whitespace: check exact whitespace and newlines)"
        io.err(f"edit: {path}: old text not found{hint}\n")
        return 1
    if count > 1 and "a" not in o:
        io.err(
            f"edit: {path}: old text matches {count} places; include more surrounding "
            "text to make it unique, or pass --all to replace every match\n"
        )
        return 1
    line = text.count("\n", 0, text.find(old)) + 1
    replaced = count if "a" in o else 1
    new_text = text.replace(old, new) if "a" in o else text.replace(old, new, 1)
    if not _write_text(sh, io, "edit", path, new_text):
        return 1
    io.out(f"edited {path}: replaced {replaced} occurrence{'s' if replaced != 1 else ''} (first at line {line})\n")
    return 0


# --------------------------------------------------------- files and folders
@command("mkdir", "mkdir [-p] DIR...")
def mkdir(sh, args, stdin, io):
    o, dirs = parse_opts("mkdir", args, bools="p")
    if not dirs:
        raise UsageError("missing operand")
    code = 0
    for a in dirs:
        try:
            sh.vfs.mkdir(sh.path(a), parents="p" in o)
        except FsError as e:
            io.err(f"mkdir: cannot create directory '{a}': {e.reason}\n")
            code = 1
    return code


@command("touch", "touch FILE...   create empty files (existing files are left alone)")
def touch(sh, args, stdin, io):
    o, files = parse_opts("touch", args)
    if not files:
        raise UsageError("missing file operand")
    code = 0
    for a in files:
        try:
            p = sh.path(a)
            if not sh.vfs.exists(p):
                sh.vfs.write_bytes(p, b"")
        except FsError as e:
            io.err(f"touch: cannot touch '{a}': {e.reason}\n")
            code = 1
    return code


@command("rm", "rm [-r] [-f] PATH...")
def rm(sh, args, stdin, io):
    o, paths = parse_opts("rm", args, bools="rRfd")
    if not paths:
        raise UsageError("missing operand")
    recursive = "r" in o or "R" in o
    code = 0
    for a in paths:
        try:
            p = sh.path(a)
            if not sh.vfs.exists(p):
                if "f" in o:
                    continue
                raise FsError(ENOENT, p)
            if sh.vfs.isdir(p) and not recursive:
                if "d" in o and not sh.vfs.listdir(p):
                    sh.vfs.remove(p)
                    continue
                raise FsError("Is a directory (use -r to remove directories)", p)
            sh.vfs.remove(p, recursive=recursive)
        except FsError as e:
            io.err(f"rm: cannot remove '{a}': {e.reason}\n")
            code = 1
    return code


@command("rmdir", "rmdir DIR...   remove empty directories")
def rmdir(sh, args, stdin, io):
    o, dirs = parse_opts("rmdir", args)
    if not dirs:
        raise UsageError("missing operand")
    code = 0
    for a in dirs:
        try:
            p = sh.path(a)
            if not sh.vfs.isdir(p):
                raise FsError(ENOTDIR if sh.vfs.isfile(p) else ENOENT, p)
            sh.vfs.remove(p, recursive=False)
        except FsError as e:
            io.err(f"rmdir: failed to remove '{a}': {e.reason}\n")
            code = 1
    return code


def _mv_cp(sh, args, io, cmd):
    o, pos = parse_opts(cmd, args, bools="rRfa" if cmd == "cp" else "f")
    if len(pos) < 2:
        raise UsageError("missing file operand (need SOURCE... DEST)")
    *srcs, dst = pos
    try:
        d = sh.path(dst)
    except FsError as e:
        io.err(f"{cmd}: cannot access '{dst}': {e.reason}\n")
        return 1
    if len(srcs) > 1 and not sh.vfs.isdir(d):
        io.err(f"{cmd}: target '{dst}' is not a directory\n")
        return 1
    code = 0
    for a in srcs:
        try:
            s = sh.path(a)
            if cmd == "cp":
                sh.vfs.copy(s, d, recursive=("r" in o or "R" in o or "a" in o))
            else:
                sh.vfs.move(s, d)
        except FsError as e:
            verb = "copy" if cmd == "cp" else "move"
            io.err(f"{cmd}: cannot {verb} '{a}' to '{dst}': {e.reason}\n")
            code = 1
    return code


@command("mv", "mv SOURCE... DEST   move or rename")
def mv(sh, args, stdin, io):
    return _mv_cp(sh, args, io, "mv")


@command("cp", "cp [-r] SOURCE... DEST   copy files (-r for directories)")
def cp(sh, args, stdin, io):
    return _mv_cp(sh, args, io, "cp")


# ------------------------------------------------------------- shell state
@command("pwd", "pwd   print the current directory")
def pwd(sh, args, stdin, io):
    io.out(sh.cwd + "\n")
    return 0


@command("cd", "cd [DIR]   change directory (no argument = /)")
def cd(sh, args, stdin, io):
    if len(args) > 1:
        raise UsageError("too many arguments")
    a = args[0] if args else "/"
    try:
        p = sh.path(a)
        if not sh.vfs.isdir(p):
            raise FsError(ENOTDIR if sh.vfs.isfile(p) else ENOENT, p)
    except FsError as e:
        _fail(io, "cd", a, e)
        return 1
    sh.cwd = p
    return 0


@command("true", "true")
def true_(sh, args, stdin, io):
    return 0


@command("false", "false")
def false_(sh, args, stdin, io):
    return 1


@command("help", "help [COMMAND]   list commands, or show one command's usage")
def help_(sh, args, stdin, io):
    if args:
        entry = sh.commands.get(args[0])
        if not entry:
            io.err(f"help: no such command: {args[0]}\n")
            return 1
        io.out(entry[1] + "\n")
        return 0
    io.out("Available commands:\n")
    for name in sorted(sh.commands):
        io.out(f"  {sh.commands[name][1]}\n")
    io.out(
        "\nShell features: pipes (|), && || ;, redirects (> >> < 2> 2>&1), here-documents (<< 'EOF'), "
        "globs (* ? [..]), quotes.\nNot available: $VAR, $(...), backticks, subshells, background jobs.\n"
    )
    return 0
