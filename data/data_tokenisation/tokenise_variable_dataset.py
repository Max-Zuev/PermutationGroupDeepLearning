# coding=utf-8
"""Tokenisation worker: tokenise exactly the groups a plan file assigns to it.

Work is assigned centrally by ``data/submit.py tokenise``, which writes a plan
file (``worker<TAB>group`` rows) before the array job starts. A worker reads its
own rows and nothing else: it never lists the orbit store or looks at other
workers, so every group is processed by exactly one worker.

For each assigned group it writes ``OUT_DIR/{group}.csv`` (format: see
tokenise_variable.py). The CSV is written under a ``.partial`` name and renamed
only when complete, so a group has a CSV if and only if it finished; anything
that failed is simply planned again on the next submit. The worker also writes
its own report file: one row per assigned group with its status and the counts
of what the tokeniser wrote and dropped (see tokenise_variable.COUNTS).

Usage (under Sage):
  sage -python tokenise_variable_dataset.py --plan PLAN.tsv --worker K \
      --orbits-dir DIR --out-dir DIR --report REPORT.tsv \
      --max-r 16 --largest-degree 500 --part-length 128 --max-per-class 200
"""

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenise_variable import COUNTS, tokenise_group_variable  # noqa: E402


def assigned_groups(plan_path, worker):
    """Group ids assigned to `worker` in the plan file, in plan order."""
    with open(plan_path, encoding="utf-8") as f:
        next(f)                               # header: worker, group
        rows = (line.split() for line in f if line.strip())
        return [group for w, group in rows if int(w) == worker]


def tokenise_one(group, orbits_dir, out_dir, settings):
    """Tokenise one group; return (status, counts). counts is {} unless it ran."""
    src = os.path.join(orbits_dir, f"{group}.sobj")
    if not os.path.exists(src):
        return "MISSING_INPUT", {}
    out = os.path.join(out_dir, f"{group}.csv")
    partial = out + ".partial"
    try:
        counts = tokenise_group_variable(src, partial, **settings)
    except Exception as e:                    # noqa: BLE001 -- report and continue
        print(f"{group}: EXCEPTION {e}")
        if os.path.exists(partial):
            os.remove(partial)
        return f"EXC:{type(e).__name__}", {}
    # a group with no rows keeps its header-only CSV: done, contributes nothing
    os.replace(partial, out)
    return ("OK" if counts["rows"] else "NO_ROWS"), counts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True, help="plan file written by data/submit.py")
    ap.add_argument("--worker", type=int, required=True, help="this worker's index in the plan")
    ap.add_argument("--orbits-dir", required=True, help="input: {group}.sobj orbit files")
    ap.add_argument("--out-dir", required=True, help="output: {group}.csv tokenised files")
    ap.add_argument("--report", required=True, help="output: this worker's group/status report")
    ap.add_argument("--max-r", type=int, required=True,
                    help="skip signatures with more than this many elements")
    ap.add_argument("--largest-degree", type=int, required=True,
                    help="n of S_n numbering the transposition tokens; "
                         "signatures of higher permutation degree are skipped")
    ap.add_argument("--part-length", type=int, required=True,
                    help="tokens per element block (order token + word + padding)")
    ap.add_argument("--max-per-class", type=int, required=True,
                    help="keep at most this many stored vectors per class")
    args = ap.parse_args()
    settings = dict(max_r=args.max_r, largest_degree=args.largest_degree,
                    part_length=args.part_length,
                    max_vectors_per_class=args.max_per_class)

    groups = assigned_groups(args.plan, args.worker)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"worker {args.worker}: {len(groups)} groups assigned by {args.plan}")

    outcomes = Counter()
    with open(args.report, "w", encoding="utf-8") as report:
        report.write("\t".join(("group", "status") + COUNTS) + "\n")
        for group in groups:
            status, counts = tokenise_one(group, args.orbits_dir, args.out_dir, settings)
            print(f"{group}: {status} " + " ".join(f"{k}={v}" for k, v in counts.items()),
                  flush=True)
            report.write("\t".join([group, status] + [str(counts.get(k, "")) for k in COUNTS])
                         + "\n")
            report.flush()
            outcomes[status] += 1

    print(f"worker {args.worker} done: "
          + ", ".join(f"{k} {v}" for k, v in sorted(outcomes.items())))


if __name__ == "__main__":
    main()
