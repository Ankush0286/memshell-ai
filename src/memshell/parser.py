"""Tokenizer and parser for the memshell command language.

Supported syntax
    words with 'single', "double" quotes and \\escapes (multi-line quotes OK)
    pipes                 a | b
    sequencing            a ; b     a && b     a || b     (and newlines)
    redirects             > f   >> f   < f   2> f   2>> f   2>&1   >&2
    here-documents        << 'EOF'   <<EOF   <<-EOF   (body is ALWAYS literal)
    globs                 * ? [abc]  (expanded by the shell, not here)
    comments              # at the start of a word

Deliberately unsupported (rejected with a clear error, never guessed):
    $VAR / ${VAR} / $(cmd) / `cmd`, subshells ( ), background &, &>
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


class ParseError(Exception):
    pass


_META = set(" \t\n|&;<>()")
_OPERATORS = ["<<-", "<<", ">>", ">&", "&&", "||", ">", "<", "|", ";", "&"]

_DOLLAR_MSG = (
    "unsupported: variable or command expansion ('$'). "
    "Variables and $(...) are not available; use single quotes for a literal $, e.g. 'cost: $5'"
)
_BACKTICK_MSG = "unsupported: command substitution (backticks)"


@dataclass
class Tok:
    kind: str  # 'word' | 'op'
    value: str
    start: int
    end: int
    pattern: Optional[str] = None  # glob pattern if the word has unquoted wildcards
    plain_digit: bool = False  # unquoted all-digit word (possible fd prefix)
    fd: Optional[int] = None
    body: Optional[str] = None  # heredoc body, for '<<' tokens


@dataclass(frozen=True)
class Word:
    value: str
    glob: Optional[str] = None


@dataclass
class Redirect:
    fd: int
    kind: str  # 'in' | 'out' | 'append' | 'dup'
    target: Optional[Word] = None
    dup: Optional[int] = None


@dataclass
class Command:
    argv: List[Word] = field(default_factory=list)
    redirects: List[Redirect] = field(default_factory=list)
    stdin_text: Optional[str] = None

    def is_empty(self) -> bool:
        return not (self.argv or self.redirects or self.stdin_text is not None)


@dataclass
class Pipeline:
    commands: List[Command]


@dataclass
class Step:
    pipeline: Pipeline
    connector: str  # ';' | '&&' | '||'  (how it joins the previous step)


# ---------------------------------------------------------------- tokenizer
def _check_dollar(src: str, i: int) -> None:
    nxt = src[i + 1:i + 2]
    if nxt and (nxt.isalnum() or nxt in "_{(?$!@*#-"):
        raise ParseError(_DOLLAR_MSG)


def _lex_word(src: str, i: int):
    n = len(src)
    buf: List[str] = []
    pbuf: List[str] = []
    quoted = False
    magic = False

    def lit(ch: str) -> None:
        buf.append(ch)
        pbuf.append("[" + ch + "]" if ch in "*?[" else ch)

    while i < n:
        c = src[i]
        if c in _META:
            break
        if c == "'":
            j = src.find("'", i + 1)
            if j < 0:
                raise ParseError("unterminated single quote")
            for ch in src[i + 1:j]:
                lit(ch)
            quoted = True
            i = j + 1
        elif c == '"':
            quoted = True
            i += 1
            while True:
                if i >= n:
                    raise ParseError("unterminated double quote")
                d = src[i]
                if d == '"':
                    i += 1
                    break
                if d == "\\" and i + 1 < n and src[i + 1] in '"\\$`\n':
                    if src[i + 1] != "\n":
                        lit(src[i + 1])
                    i += 2
                elif d == "`":
                    raise ParseError(_BACKTICK_MSG)
                elif d == "$":
                    _check_dollar(src, i)
                    lit("$")
                    i += 1
                else:
                    lit(d)
                    i += 1
        elif c == "\\":
            if i + 1 >= n:
                lit("\\")
                i += 1
            elif src[i + 1] == "\n":
                i += 2
            else:
                lit(src[i + 1])
                quoted = True
                i += 2
        elif c == "`":
            raise ParseError(_BACKTICK_MSG)
        elif c == "$":
            _check_dollar(src, i)
            buf.append("$")
            pbuf.append("$")
            i += 1
        else:
            if c in "*?[":
                magic = True
            buf.append(c)
            pbuf.append(c)
            i += 1
    return "".join(buf), ("".join(pbuf) if magic else None), quoted, i


def _read_heredocs(src: str, i: int, toks: List[Tok], pending: List[int]) -> int:
    for idx in pending:
        op = toks[idx]
        if idx + 1 >= len(toks) or toks[idx + 1].kind != "word":
            raise ParseError("syntax error: '<<' must be followed by a delimiter word")
        delim = toks[idx + 1].value
        strip = op.value == "<<-"
        lines: List[str] = []
        while True:
            if i >= len(src):
                raise ParseError(
                    f"here-document is missing its terminator: add a line containing only '{delim}'"
                )
            j = src.find("\n", i)
            line = src[i:j] if j >= 0 else src[i:]
            i = j + 1 if j >= 0 else len(src)
            if strip:
                line = line.lstrip("\t")
            if line.rstrip("\r") == delim:
                break
            lines.append(line)
        op.body = "".join(l + "\n" for l in lines)
    return i


def tokenize(src: str) -> List[Tok]:
    src = src.replace("\r\n", "\n")
    toks: List[Tok] = []
    pending: List[int] = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c in " \t":
            i += 1
            continue
        if c == "\\" and src[i + 1:i + 2] == "\n":
            i += 2
            continue
        if c == "\n":
            toks.append(Tok("op", "\n", i, i + 1))
            i += 1
            if pending:
                i = _read_heredocs(src, i, toks, pending)
                pending = []
            continue
        if c == "#" and (i == 0 or src[i - 1] in " \t\n;|&"):
            while i < n and src[i] != "\n":
                i += 1
            continue
        if c in "()":
            raise ParseError("unsupported: subshells and grouping ('(' ')')")
        if c in "|&;<>":
            if src.startswith("&>", i):
                raise ParseError("unsupported: '&>' (write '> file 2>&1' instead)")
            op = next(o for o in _OPERATORS if src.startswith(o, i))
            if op == "&":
                raise ParseError("unsupported: background jobs ('&')")
            tok = Tok("op", op, i, i + len(op))
            if (
                op in (">", ">>", ">&", "<")
                and toks
                and toks[-1].kind == "word"
                and toks[-1].plain_digit
                and toks[-1].end == i
            ):
                tok.fd = int(toks.pop().value)
            toks.append(tok)
            if op in ("<<", "<<-"):
                pending.append(len(toks) - 1)
            i += len(op)
            continue
        start = i
        value, pattern, quoted, i = _lex_word(src, i)
        if i == start:
            raise ParseError(f"unexpected character {src[i]!r}")
        toks.append(
            Tok("word", value, start, i, pattern=pattern, plain_digit=(value.isdigit() and not quoted))
        )
    if pending:
        raise ParseError("here-document is missing its body and terminator line")
    return toks


# ------------------------------------------------------------------- parser
def parse(src: str) -> List[Step]:
    toks = tokenize(src)
    steps: List[Step] = []
    cmds: List[Command] = []
    cur = Command()
    conn = ";"
    after_pipe = False
    need_cmd = False  # after && or ||
    i = 0

    def unexpected(v: str) -> ParseError:
        return ParseError(f"syntax error near unexpected token '{'newline' if v == chr(10) else v}'")

    while i < len(toks):
        t = toks[i]
        if t.kind == "word":
            cur.argv.append(Word(t.value, t.pattern))
            i += 1
            continue
        v = t.value
        if v in (">", ">>", "<", ">&"):
            fd = t.fd if t.fd is not None else (0 if v == "<" else 1)
            i += 1
            if i >= len(toks) or toks[i].kind != "word":
                raise ParseError(f"syntax error: expected a file name after '{v}'")
            target = toks[i]
            if v == "<":
                if fd != 0:
                    raise ParseError("unsupported file descriptor for '<'")
                cur.redirects.append(Redirect(0, "in", Word(target.value, target.pattern)))
            elif v == ">&":
                if fd not in (1, 2) or target.value not in ("1", "2"):
                    raise ParseError("unsupported redirect: only 2>&1 and 1>&2 are supported")
                cur.redirects.append(Redirect(fd, "dup", dup=int(target.value)))
            else:
                if fd not in (1, 2):
                    raise ParseError("unsupported file descriptor: only 1 (stdout) and 2 (stderr)")
                cur.redirects.append(
                    Redirect(fd, "out" if v == ">" else "append", Word(target.value, target.pattern))
                )
            i += 1
            continue
        if v in ("<<", "<<-"):
            cur.stdin_text = t.body
            i += 2  # skip the delimiter word
            continue
        if v == "|":
            if cur.is_empty():
                raise unexpected(v)
            cmds.append(cur)
            cur = Command()
            after_pipe = True
            i += 1
            continue
        if v == "\n" and cur.is_empty():
            i += 1  # blank line, or continuation after '|', '&&', '||'
            continue
        # ';'  '&&'  '||'  newline  -> end of a pipeline
        if cur.is_empty():
            raise unexpected(v)
        cmds.append(cur)
        cur = Command()
        steps.append(Step(Pipeline(cmds), conn))
        cmds = []
        after_pipe = False
        need_cmd = v in ("&&", "||")
        conn = v if need_cmd else ";"
        i += 1

    if not cur.is_empty():
        cmds.append(cur)
        after_pipe = False
        need_cmd = False
    if after_pipe:
        raise ParseError("syntax error: unexpected end of input after '|'")
    if need_cmd:
        raise ParseError(f"syntax error: unexpected end of input after '{conn}'")
    if cmds:
        steps.append(Step(Pipeline(cmds), conn))
    return steps
