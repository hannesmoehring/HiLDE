# Regression checks

Standalone checks retained from the 2026-08 pre-freeze code review.

- `02_zadu_equivalence.py` / `03_zadu_chunked_and_degenerate.py` — prove
  `src/evaluation/neighbor_metrics.py` is numerically equivalent to ZADU,
  including the chunked path and degenerate inputs. The thesis's
  faithfulness numbers rest on this equivalence.
- `05_h1a_replicate_collapse.py` — live check that replicate seeds reach the
  reducers (distinct replicates) and that the H1a Wilcoxon runs on unpooled n.
- `09_mrre_direction_inverted.py` — live check that MRRE direction is
  declared correctly (both terms are 1 - error similarities, higher better).

Run from /tmp so no CWD-relative dataset path resolves into the repo:

    cd /tmp && PYTHONPATH=<repo> <repo>/.venv/bin/python <repo>/scripts/checks/<script>.py
