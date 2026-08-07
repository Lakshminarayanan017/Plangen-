"""
console.py — terminal output that survives every platform.

Windows consoles default to cp1252, which cannot encode the glyphs our
reports use (>= / -> / mid-dot / em dash). Without this, a CLI that runs
perfectly on Linux dies with UnicodeEncodeError on the user's machine at the
moment it tries to print its results — the worst possible time.
"""

from __future__ import annotations

import sys


def force_utf8_console() -> None:
    """Reconfigure stdout/stderr to UTF-8 (no-op where unsupported)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
