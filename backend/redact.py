"""Keep host filesystem paths out of messages that reach the client.

Exception text from loaders and file I/O often carries absolute paths
(`FileNotFoundError: [Errno 2] ... '/abs/path/to/file.csv'`). Job errors and
400 details go to the browser verbatim, so they are passed through
`scrub_paths` first: every absolute path is cut down to its last component.
"""

from __future__ import annotations

import re

# A POSIX absolute path ("/a/b/c.csv", "~/a/b") or a Windows one ("C:\\a\\b").
# The lookbehind keeps URL paths ("https://host/a/b") and relative ones intact.
_ABS_PATH = re.compile(
    r"(?<![\w:/.~-])(?:~?/|[A-Za-z]:[\\/])(?:[^\s'\"<>:,;()\[\]{}]+[\\/])*"
    r"(?P<name>[^\s'\"<>:,;()\[\]{}\\/]*)"
)


def scrub_paths(message: str) -> str:
    """Replace each absolute path in `message` with `.../<basename>`."""
    return _ABS_PATH.sub(lambda m: f".../{m.group('name')}", message)


def safe_detail(exc: BaseException) -> str:
    """`str(exc)` with host paths removed, for client-visible error details."""
    return scrub_paths(str(exc))
