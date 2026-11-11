"""Read-only pid-file helpers, importable without pulling in torch.

``melba.utils`` imports the device helpers (and torch) at module level, which is exactly
what a status probe must avoid, so the pid-file logic lives in this top-level module:
``scripts/run_queue.py`` needs it to know whether the job it waits for is still alive and
``scripts/status.py`` needs it without importing torch, the GPU stack or the GPU at all.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Sequence

__all__ = [
    "LAUNCHER_PROCESS_NAMES",
    "pid_alive",
    "process_name",
    "read_pid_file",
]

#: images that may legitimately own a grid/queue pid file (pids get recycled)
LAUNCHER_PROCESS_NAMES: tuple[str, ...] = (
    "python",
    "python3",
    "py",
    "powershell",
    "pwsh",
)

#: pid files are written by several tools: bare ("4242"), "pid 4242" and with a UTF-8 BOM
_PID_PATTERN = re.compile(r"\d+")

_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def _normalise(name: str) -> str:
    base = name.strip().strip('"').lower().replace("\\", "/").rsplit("/", 1)[-1]
    if base.endswith(".exe"):
        base = base[: -len(".exe")]
    return base


def _windows_image_name(pid: int) -> str | None:  # pragma: no cover - Windows only
    """Image name via ``OpenProcess``/``QueryFullProcessImageName`` (no subprocess)."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL

    handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return None
    try:
        size = wintypes.DWORD(32768)
        buffer = ctypes.create_unicode_buffer(size.value)
        if not kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            return None
        return _normalise(buffer.value) or None
    finally:
        kernel32.CloseHandle(handle)


def process_name(pid: int) -> str | None:
    """Image name of ``pid`` (``None`` when it does not exist or cannot be read)."""
    try:
        value = int(pid)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    if sys.platform.startswith("win"):  # pragma: no cover - platform branch
        return _windows_image_name(value)
    try:  # pragma: no cover - platform branch
        return _normalise(Path(f"/proc/{value}/comm").read_text(encoding="utf-8"))
    except OSError:
        return None


def pid_alive(pid: int, *, names: Sequence[str] | None = None) -> bool:
    """True when a process with ``pid`` exists; never signals or waits for it.

    ``names`` restricts the answer to a set of image names, which is what a pid file
    needs: Windows recycles pids quickly, so a stale ``logs/x_grid.pid`` can point at an
    unrelated process (a browser, say) and liveness alone would lie.
    """
    image = process_name(pid)
    if image is None:
        return False
    if names is None:
        return True
    return image in {_normalise(name) for name in names}


def read_pid_file(path: str | Path) -> int:
    """The pid stored in ``path`` (``0`` when it is missing or holds no pid).

    The files are not written by one tool: some hold a bare number, some ``pid 4242``, and some
    carry a UTF-8 BOM (a sidestepping ``open(..., "w", encoding="utf-8-sig")`` anywhere upstream).
    ``utf-8-sig`` strips the BOM and the first integer in the file is the pid, so all three forms
    work - a pid file that parses to 0 would otherwise disable a queue's wait guard silently.
    """
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except OSError:
        return 0
    match = _PID_PATTERN.search(text)
    if match is None:
        return 0
    value = int(match.group(0))
    return value if value > 0 else 0
