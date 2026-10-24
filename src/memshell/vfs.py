"""In-memory virtual filesystem.

A thin, hardened wrapper around fsspec's MemoryFileSystem. It adds:

* a virtual root "/" that paths can never escape (``..`` clamps at the root)
* size limits (per file, total, and file count)
* POSIX-style error reasons ("No such file or directory", ...)
* an audit trail of every change
* a lock so several shells can safely share one filesystem
"""
from __future__ import annotations

import logging
import os
import posixpath
import threading
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterator, List, Tuple

from fsspec.implementations.memory import MemoryFileSystem

log = logging.getLogger("memshell.vfs")

ENOENT = "No such file or directory"
EEXIST = "File exists"
EISDIR = "Is a directory"
ENOTDIR = "Not a directory"
ENOTEMPTY = "Directory not empty"


class FsError(Exception):
    """A filesystem failure with a shell-style reason string."""

    def __init__(self, reason: str, path: str | None = None):
        super().__init__(reason)
        self.reason = reason
        self.path = path


class _IsolatedMemoryFS(MemoryFileSystem):
    """fsspec's MemoryFileSystem keeps ONE store shared by every instance
    (it is a class attribute). Give each instance its own, so two agents
    or two tests never see each other's files by accident."""

    cachable = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.store = {}
        self.pseudo_dirs = [""]


@dataclass(frozen=True)
class Entry:
    name: str
    path: str
    is_dir: bool
    size: int


class VirtualFS:
    def __init__(
        self,
        *,
        max_file_size: int = 1_000_000,
        max_total_size: int = 50_000_000,
        max_files: int = 10_000,
        audit_size: int = 1000,
    ):
        self._fs = _IsolatedMemoryFS()
        self.max_file_size = max_file_size
        self.max_total_size = max_total_size
        self.max_files = max_files
        self.lock = threading.RLock()
        self.audit: deque = deque(maxlen=audit_size)

    # ------------------------------------------------------------------ paths
    @staticmethod
    def normalize(path: str, cwd: str = "/") -> str:
        """Resolve ``path`` against ``cwd`` into a clean absolute path.
        ``..`` can never climb above the virtual root."""
        if path == "":
            raise FsError(ENOENT, path)
        if "\x00" in path:
            raise FsError("Invalid argument", path)
        if len(path) > 4096:
            raise FsError("File name too long", path)
        joined = path if path.startswith("/") else posixpath.join(cwd, path)
        return posixpath.normpath("/" + joined.lstrip("/"))

    # ---------------------------------------------------------------- queries
    def isdir(self, p: str) -> bool:
        return p == "/" or bool(self._fs.isdir(p))

    def isfile(self, p: str) -> bool:
        return p != "/" and bool(self._fs.isfile(p))

    def exists(self, p: str) -> bool:
        return self.isdir(p) or self.isfile(p)

    def stat(self, p: str) -> Entry:
        if p == "/":
            return Entry("/", "/", True, 0)
        if self.isdir(p):
            return Entry(posixpath.basename(p), p, True, 0)
        if self.isfile(p):
            return Entry(posixpath.basename(p), p, False, int(self._fs.info(p)["size"]))
        raise FsError(ENOENT, p)

    def listdir(self, p: str) -> List[Entry]:
        if not self.isdir(p):
            raise FsError(ENOTDIR if self.isfile(p) else ENOENT, p)
        out = []
        for d in self._fs.ls(p, detail=True):
            full = "/" + d["name"].lstrip("/")
            out.append(
                Entry(
                    posixpath.basename(full),
                    full,
                    d["type"] == "directory",
                    int(d.get("size") or 0),
                )
            )
        out.sort(key=lambda e: e.name)
        return out

    def walk(self, p: str) -> Iterator[Tuple[str, bool]]:
        """Yield (path, is_dir) for every descendant of directory ``p``."""
        for e in self.listdir(p):
            yield e.path, e.is_dir
            if e.is_dir:
                yield from self.walk(e.path)

    def total_size(self) -> int:
        return sum(f.size for f in self._fs.store.values())

    def file_count(self) -> int:
        return len(self._fs.store)

    # ------------------------------------------------------------------ reads
    def read_bytes(self, p: str) -> bytes:
        if self.isdir(p):
            raise FsError(EISDIR, p)
        if not self.isfile(p):
            raise FsError(ENOENT, p)
        return bytes(self._fs.cat_file(p))

    # ----------------------------------------------------------------- writes
    def _record(self, op: str, path: str, detail: str = "") -> None:
        self.audit.append((op, path, detail))
        log.info("%s %s %s", op, path, detail)

    def _check_ancestors(self, p: str) -> None:
        parent = posixpath.dirname(p)
        while parent not in ("", "/"):
            if self.isfile(parent):
                raise FsError(ENOTDIR, p)
            parent = posixpath.dirname(parent)

    def write_bytes(self, p: str, data: bytes, append: bool = False) -> None:
        with self.lock:
            if p == "/" or self.isdir(p):
                raise FsError(EISDIR, p)
            self._check_ancestors(p)
            if not self.isdir(posixpath.dirname(p)):
                raise FsError(ENOENT, p)
            is_new = not self.isfile(p)
            existing = b"" if is_new else self.read_bytes(p)
            final = existing + data if append else data
            if len(final) > self.max_file_size:
                raise FsError(f"File too large (limit {self.max_file_size} bytes)", p)
            if self.total_size() - len(existing) + len(final) > self.max_total_size:
                raise FsError("No space left on virtual filesystem", p)
            if is_new and self.file_count() >= self.max_files:
                raise FsError(f"Too many files (limit {self.max_files})", p)
            self._fs.pipe_file(p, final)
            self._record("append" if append else "write", p, f"{len(final)} bytes")

    def mkdir(self, p: str, parents: bool = False) -> None:
        with self.lock:
            if self.exists(p):
                if parents and self.isdir(p):
                    return
                raise FsError(EEXIST, p)
            self._check_ancestors(p)
            if not parents and not self.isdir(posixpath.dirname(p)):
                raise FsError(ENOENT, p)
            self._fs.makedirs(p, exist_ok=True)
            self._record("mkdir", p)

    def remove(self, p: str, recursive: bool = False) -> None:
        with self.lock:
            if p == "/":
                raise FsError("Refusing to remove the root directory", p)
            if self.isfile(p):
                self._fs.rm_file(p)
                self._record("rm", p)
                return
            if not self.isdir(p):
                raise FsError(ENOENT, p)
            descendants = list(self.walk(p))
            if descendants and not recursive:
                raise FsError(ENOTEMPTY, p)
            for path, is_dir in reversed(descendants):  # deepest first
                if is_dir:
                    self._fs.rmdir(path)
                else:
                    self._fs.rm_file(path)
            self._fs.rmdir(p)
            self._record("rm", p, "recursive" if descendants else "")

    def _dest(self, src: str, dst: str) -> str:
        return posixpath.join(dst, posixpath.basename(src)) if self.isdir(dst) else dst

    def copy(self, src: str, dst: str, recursive: bool = False) -> None:
        """POSIX-style cp: if ``dst`` is an existing directory, copy into it."""
        with self.lock:
            if not self.exists(src):
                raise FsError(ENOENT, src)
            self._copy_to(src, self._dest(src, dst), recursive)

    def _copy_to(self, src: str, final: str, recursive: bool) -> None:
        if final == src:
            raise FsError("Source and destination are the same", src)
        if self.isfile(src):
            if self.isdir(final):
                raise FsError(EISDIR, final)
            self.write_bytes(final, self.read_bytes(src))
            return
        if not recursive:
            raise FsError("Is a directory (use -r to copy directories)", src)
        if final.startswith(src.rstrip("/") + "/"):
            raise FsError("Cannot copy a directory into itself", src)
        if self.isfile(final):
            raise FsError(ENOTDIR, final)
        items = list(self.walk(src))
        needed = sum(self.stat(p).size for p, d in items if not d)
        if self.total_size() + needed > self.max_total_size:
            raise FsError("No space left on virtual filesystem", final)
        if not self.isdir(final):
            self.mkdir(final)
        for path, is_dir in items:
            target = final + path[len(src.rstrip("/")):]
            if is_dir:
                self.mkdir(target, parents=True)
            else:
                self.write_bytes(target, self.read_bytes(path))

    def move(self, src: str, dst: str) -> None:
        """POSIX-style mv: if ``dst`` is an existing directory, move into it."""
        with self.lock:
            if src == "/":
                raise FsError("Refusing to move the root directory", src)
            if not self.exists(src):
                raise FsError(ENOENT, src)
            final = self._dest(src, dst)
            if final == src:
                raise FsError("Source and destination are the same", src)
            if self.isdir(src):
                if final.startswith(src.rstrip("/") + "/"):
                    raise FsError("Cannot move a directory into itself", src)
                if self.isfile(final):
                    raise FsError(ENOTDIR, final)
                if self.isdir(final) and self.listdir(final):
                    raise FsError(ENOTEMPTY, final)
                self._copy_to(src, final, recursive=True)
                self.remove(src, recursive=True)
                return
            if self.isdir(final):
                raise FsError(EISDIR, final)
            data = self.read_bytes(src)
            self._fs.rm_file(src)  # free the space first, restore on failure
            try:
                self.write_bytes(final, data)
            except FsError:
                self._fs.pipe_file(src, data)
                raise
            self._record("mv", src, f"-> {final}")

    # ------------------------------------------------- snapshots / persistence
    def export(self) -> Dict[str, object]:
        """Plain-data snapshot: {'dirs': [...], 'files': {path: bytes}}."""
        with self.lock:
            dirs, files = [], {}
            for path, is_dir in self.walk("/"):
                if is_dir:
                    dirs.append(path)
                else:
                    files[path] = self.read_bytes(path)
            return {"dirs": dirs, "files": files}

    def load(self, snapshot: Dict[str, object], replace: bool = True) -> None:
        with self.lock:
            if replace:
                for e in self.listdir("/"):
                    self.remove(e.path, recursive=True)
            for d in snapshot.get("dirs", []):  # type: ignore[union-attr]
                self.mkdir(d, parents=True)
            for path, data in snapshot.get("files", {}).items():  # type: ignore[union-attr]
                self.mkdir(posixpath.dirname(path), parents=True)
                self.write_bytes(path, data)

    def load_from_dir(self, directory: str, dest: str = "/") -> int:
        """Seed the virtual filesystem from a real folder (read-only on the
        real side; symlinks are skipped). Returns the number of files loaded."""
        count = 0
        root = os.path.abspath(directory)
        for base, dirnames, filenames in os.walk(root, followlinks=False):
            rel = os.path.relpath(base, root)
            vbase = self.normalize(dest if rel == "." else posixpath.join(dest, rel.replace(os.sep, "/")))
            self.mkdir(vbase, parents=True)
            for d in dirnames:
                if not os.path.islink(os.path.join(base, d)):
                    self.mkdir(posixpath.join(vbase, d), parents=True)
            for f in filenames:
                full = os.path.join(base, f)
                if os.path.islink(full):
                    continue
                with open(full, "rb") as fh:
                    self.write_bytes(posixpath.join(vbase, f), fh.read())
                count += 1
        return count

    def dump_to_dir(self, directory: str) -> int:
        """Write the whole virtual filesystem out to a real folder."""
        snap = self.export()
        os.makedirs(directory, exist_ok=True)
        for d in snap["dirs"]:  # type: ignore[union-attr]
            os.makedirs(os.path.join(directory, d.lstrip("/")), exist_ok=True)
        for path, data in snap["files"].items():  # type: ignore[union-attr]
            target = os.path.join(directory, path.lstrip("/"))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "wb") as fh:
                fh.write(data)
        return len(snap["files"])  # type: ignore[arg-type]
