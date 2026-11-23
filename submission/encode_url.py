#!/usr/bin/env python3
"""Encode the app URL for submission/index.html (XOR with a short key, then base64).

Usage:
    python submission/encode_url.py https://real.url           # print encoded string
    python submission/encode_url.py https://real.url --write   # also update index.html
    python submission/encode_url.py --decode <encoded>         # decode an encoded string

Encoding always round-trips the result through decode before printing or writing.
This is obfuscation, not encryption: it only keeps the URL out of plain sight.
"""

import argparse
import base64
import re
import sys
from pathlib import Path

# Must match KEY in index.html.
KEY = b"hilde"
INDEX = Path(__file__).resolve().parent / "index.html"
CONSTANT_RE = re.compile(r'(const ENC = ")([A-Za-z0-9+/=]*)(";)')


def _xor(data: bytes) -> bytes:
    return bytes(b ^ KEY[i % len(KEY)] for i, b in enumerate(data))


def encode(url: str) -> str:
    return base64.b64encode(_xor(url.encode("utf-8"))).decode("ascii")


def decode(encoded: str) -> str:
    return _xor(base64.b64decode(encoded)).decode("utf-8")


def write_index(encoded: str) -> None:
    html = INDEX.read_text(encoding="utf-8")
    new_html, count = CONSTANT_RE.subn(lambda m: m.group(1) + encoded + m.group(3), html)
    if count != 1:
        sys.exit(f"error: expected exactly one 'const ENC = \"...\";' in {INDEX}, found {count}")
    INDEX.write_text(new_html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url", nargs="?", help="URL to encode, e.g. https://REPLACE-ME.example")
    parser.add_argument("--write", action="store_true", help="rewrite the ENC constant in index.html")
    parser.add_argument("--decode", metavar="ENCODED", help="decode an encoded string and print the URL")
    args = parser.parse_args()

    if args.decode:
        print(decode(args.decode))
        return
    if not args.url:
        parser.error("a URL is required unless --decode is given")
    if not args.url.startswith("https://"):
        print("warning: URL does not start with https://", file=sys.stderr)

    encoded = encode(args.url)
    if decode(encoded) != args.url:
        sys.exit("error: round-trip check failed")
    print(encoded)
    if args.write:
        write_index(encoded)
        print(f"updated {INDEX}", file=sys.stderr)


if __name__ == "__main__":
    main()
