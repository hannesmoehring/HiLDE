"""Driver: run every re-derivation and print a digest.

uv run python -m src_research.rederive
"""

from __future__ import annotations

from src_research.rederive import (
    REDERIVED_DIRNAME,
    h1a,
    h2b,
    internal_external,
    verdicts,
)


def main() -> None:
    print(
        f"Re-deriving corrected aggregates into outputs/experiments/<run>/{REDERIVED_DIRNAME}/"
    )
    print("Originals are read-only; no experiment is rerun.\n")

    print("1. Corrected h1a_summary.csv (B2 — MRRE direction)")
    h1a.main()
    print("\n2. RQ1-S / H2b summary (H12d — duplicated control, mixed samples)")
    h2b.main()
    print(
        "\n3. internal-vs-external (H12e — the join, and the number that was never persisted)"
    )
    internal_external.main()
    print("\n4. RQ2 verdicts (H12c — honest n)")
    verdicts.main()

    print(f"\nDone. Each run directory now carries {REDERIVED_DIRNAME}/DELTAS.md.")


if __name__ == "__main__":
    main()
