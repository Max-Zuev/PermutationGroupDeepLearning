# coding=utf-8
"""Cut fixed-size prefixes of pair CSVs (the paper's subsampled training sets).

The pair builders write their rows shuffled, so the first N rows of a pair file
are a uniform random subsample of it. The data-scaling runs (Table 2) train on
prefixes of the |G| <= 64 training file, and the r <= 16 Table 1 runs on a
300k/10k prefix of the full-universe train/eval files. This script makes those
files explicitly instead of a `head` in a job script.

Each --copy takes SOURCE OUT ROWS: the header plus the first ROWS data rows of
SOURCE are written to OUT (via OUT.partial, renamed when complete). If SOURCE
has fewer than ROWS rows, all of it is copied and the shortfall is reported.

  python data/data_pairs/prefix_pairs.py \\
      --copy pairs_union_full/train_le64_masked.csv pairs_union_subsets/train_100000.csv 100000 \\
      --copy ...
"""

from __future__ import annotations

import argparse
import os


def copy_prefix(source: str, out: str, rows: int) -> int:
    """Write the header and first `rows` data rows of source to out; return rows written."""
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    partial = out + ".partial"
    written = 0
    with open(source, encoding="utf-8") as src, open(partial, "w", encoding="utf-8") as dst:
        dst.write(next(src))                  # header
        for line in src:
            if written == rows:
                break
            dst.write(line)
            written += 1
    os.replace(partial, out)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--copy", nargs=3, action="append", required=True,
                    metavar=("SOURCE", "OUT", "ROWS"), help="repeatable")
    args = ap.parse_args()
    for source, out, rows in args.copy:
        rows = int(rows)
        written = copy_prefix(source, out, rows)
        note = "" if written == rows else f"  (SHORT: source has only {written} rows)"
        print(f"{out}: {written} rows from {source}{note}", flush=True)


if __name__ == "__main__":
    main()
