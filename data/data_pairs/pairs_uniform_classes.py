# coding=utf-8
"""Class-uniform test pairs: every eligible class contributes the same number.

The random sampler (pairs_variable.py) draws positives uniformly over classes
but negatives uniformly over signatures, so a class in a crowded signature is
under-represented among negatives. For the test sets and the error analysis
every class should carry equal weight, so this builder goes class by class.

For every class of a group of order > --min-order it emits

  --per-class-pos  positive pairs: two distinct stored vectors of the class
                   (a class with a single stored vector gives none), and
  --per-class-neg  negative pairs: a vector of the class against a vector of
                   another class of the same signature and order range, that
                   class drawn uniformly each time (a class whose signature
                   has no other such class gives none).

With --target-pairs N the classes are visited in a seeded random order and
the builder stops as soon as exactly N pairs have been emitted, skipping any
class whose pairs would overshoot: a size-matched census over a uniform
sample of classes (the paper's census-full test set, matched to the
|G| > 64 census). Without it every class is visited, in sorted order.

The positive/negative balance is whatever the class structure gives, not
1:1. Rows are shuffled before writing. Output columns match pairs_variable.py:
sig, text1, text2, target, id1, id2.

    python data/data_pairs/pairs_uniform_classes.py \\
        --tok-dir DIR --out FILE.csv --min-order 64 --seed 11
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pairs_variable import index_rows, read_text  # noqa: E402

csv.field_size_limit(10 ** 9)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tok-dir", required=True, help="tokenised group CSVs")
    p.add_argument("--out", required=True, help="output CSV")
    p.add_argument("--min-order", type=int, default=64,
                   help="keep classes of groups with order strictly above this")
    p.add_argument("--per-class-pos", type=int, default=4)
    p.add_argument("--per-class-neg", type=int, default=4)
    p.add_argument("--target-pairs", type=int, default=0,
                   help="if > 0, sample classes uniformly and stop at exactly this many pairs")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = random.Random(args.seed)

    id_map, sig_index = index_rows(args.tok_dir)

    def order(full_id):
        return int(full_id.rsplit("#", 1)[-1].split("_")[0])

    keep = {c: rows for c, rows in id_map.items() if order(c) > args.min_order}
    cls_sig = {}                              # class -> its signature
    sig_classes = defaultdict(set)            # signature -> eligible classes
    for sig, entries in sig_index.items():
        for full_id, _, _ in entries:
            if full_id in keep:
                cls_sig[full_id] = sig
                sig_classes[sig].add(full_id)
    print(f"{len(id_map)} classes indexed, {len(keep)} above order "
          f"{args.min_order} in {len(sig_classes)} signatures", flush=True)

    order_list = sorted(keep)
    if args.target_pairs:
        rng.shuffle(order_list)               # uniform class sample
    remaining, taken = args.target_pairs, 0
    rows_out, no_pos, no_neg = [], 0, 0
    for c in order_list:
        sig = cls_sig[c]
        rows = keep[c]
        can_pos = len(rows) >= 2
        mates = sorted(sig_classes[sig] - {c})
        if args.target_pairs:
            yld = (args.per_class_pos if can_pos else 0) \
                + (args.per_class_neg if mates else 0)
            if yld == 0 or yld > remaining:
                continue                      # keep scanning for a class that fits
            remaining -= yld
            taken += 1
        if can_pos:
            for _ in range(args.per_class_pos):
                (p1, o1), (p2, o2) = rng.sample(rows, 2)
                rows_out.append([sig, read_text(p1, o1), read_text(p2, o2), 1, c, c])
        else:
            no_pos += 1
        if mates:
            for _ in range(args.per_class_neg):
                other = rng.choice(mates)
                (p1, o1) = rng.choice(rows)
                (p2, o2) = rng.choice(keep[other])
                rows_out.append([sig, read_text(p1, o1), read_text(p2, o2), 0, c, other])
        else:
            no_neg += 1
        if args.target_pairs and remaining == 0:
            break
    if args.target_pairs:
        print(f"sampled {taken} classes for target {args.target_pairs} "
              f"({remaining} unfilled)", flush=True)
    rng.shuffle(rows_out)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sig", "text1", "text2", "target", "id1", "id2"])
        w.writerows(rows_out)
    pos = sum(1 for r in rows_out if r[3] == 1)
    print(f"wrote {len(rows_out)} pairs ({pos} pos / {len(rows_out) - pos} neg) "
          f"to {args.out}; {no_pos} single-vector classes without positives, "
          f"{no_neg} lone-signature classes without negatives", flush=True)


if __name__ == "__main__":
    main()
