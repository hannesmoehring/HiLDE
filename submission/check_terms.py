"""Grep the branch for every term in anonymization-terms.txt.

    python submission/check_terms.py

Scans git-tracked files plus untracked, non-ignored ones. Matching is
case-insensitive and whole-word, so a short term does not hit longer words. Files that
cannot be decoded as UTF-8 (PDFs, images) are listed separately: term
replacement on anonymous.4open.science does not reach them, so look at each one.
Exits 1 if any term is found.
"""

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TERMS_FILE = HERE / "anonymization-terms.txt"
# These files list the terms on purpose; the 4open term list masks them.
SKIP = {"submission/anonymization-terms.txt", "submission/ANONYMIZATION.md"}


def load_terms() -> list[str]:
    lines = TERMS_FILE.read_text(encoding="utf-8").splitlines()
    return [t.strip() for t in lines if t.strip() and not t.startswith("#")]


def candidate_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    return sorted({f for f in out.splitlines() if f and f not in SKIP and (ROOT / f).is_file()})


def main() -> int:
    terms = load_terms()
    pattern = re.compile(
        r"(?<!\w)(" + "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True)) + r")(?!\w)",
        re.IGNORECASE,
    )
    hits, binaries = [], []
    for rel in candidate_files():
        try:
            text = (ROOT / rel).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            binaries.append(rel)
            continue
        for no, line in enumerate(text.splitlines(), 1):
            for m in pattern.finditer(line):
                hits.append(f"{rel}:{no}: {m.group(0)!r}  {line.strip()[:120]}")

    for h in hits:
        print(h)
    if binaries:
        print(f"\n{len(binaries)} binary file(s), check by hand:")
        for b in binaries:
            print(f"  {b}")
    print(f"\n{len(terms)} terms, {len(hits)} hit(s).")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
