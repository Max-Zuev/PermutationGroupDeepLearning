# coding=utf-8
"""Split pair files by group order: train on small groups, evaluate on large ones.

For the extrapolation experiment (the paper's Table 2) the model must never
see a test group during training. From a train/eval pair of files built by
pairs_variable.py this writes

  train_le{T}_masked.csv   training pairs whose two groups both have order <= T
  eval_gt{T}_masked.csv    evaluation pairs whose two groups both have order > T

(T = --threshold, 64 in the paper). A pair with one group on each side of T
is dropped from both. Group orders are read off the class ids
('{sig}#{k}#{order}_{index}').

It also reports token coverage: how many token ids of the evaluation file
never occur in the training file. Groups of different orders can share
transposition tokens, so "unseen group" does not mean "unseen tokens"; this
number says how far the evaluation really is from the training vocabulary.

    python data/data_pairs/order_split.py --train-in DIR/train.csv \\
        --eval-in DIR/eval.csv --threshold 64 --out-dir DIR
"""

from __future__ import annotations

import argparse
import csv
import os

csv.field_size_limit(10 ** 9)


def orders_of(row):
    o1 = int(row["id1"].rsplit("#", 1)[-1].split("_")[0])
    o2 = int(row["id2"].rsplit("#", 1)[-1].split("_")[0])
    return o1, o2


def tokens_of(row):
    for field in ("text1", "text2"):
        for t in row[field].split(","):
            t = int(t)
            if t >= 0:                        # -1 is padding
                yield t


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-in", required=True)
    p.add_argument("--eval-in", required=True)
    p.add_argument("--threshold", type=int, default=64)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    th = args.threshold

    train_out = os.path.join(args.out_dir, f"train_le{th}_masked.csv")
    eval_out = os.path.join(args.out_dir, f"eval_gt{th}_masked.csv")

    train_tokens = set()
    kept = dropped = 0
    with open(args.train_in, newline="", encoding="utf-8") as fin, \
         open(train_out, "w", newline="", encoding="utf-8") as fout:
        r = csv.DictReader(fin)
        w = csv.DictWriter(fout, fieldnames=r.fieldnames)
        w.writeheader()
        for row in r:
            o1, o2 = orders_of(row)
            if max(o1, o2) <= th:
                w.writerow(row)
                kept += 1
                train_tokens.update(tokens_of(row))
            else:
                dropped += 1
    print(f"train: kept {kept} pairs (max order <= {th}), dropped {dropped}; "
          f"{len(train_tokens)} distinct tokens")

    ek = ed = pos = 0
    eval_tokens = set()
    with open(args.eval_in, newline="", encoding="utf-8") as fin, \
         open(eval_out, "w", newline="", encoding="utf-8") as fout:
        r = csv.DictReader(fin)
        w = csv.DictWriter(fout, fieldnames=r.fieldnames)
        w.writeheader()
        for row in r:
            o1, o2 = orders_of(row)
            if min(o1, o2) > th:              # both groups above the threshold
                w.writerow(row)
                ek += 1
                pos += (row["target"] == "1")
                eval_tokens.update(tokens_of(row))
            else:
                ed += 1
    unseen = eval_tokens - train_tokens
    print(f"eval:  kept {ek} pairs (both orders > {th}; {pos} pos / {ek - pos} neg), "
          f"dropped {ed}")
    print(f"token coverage: eval uses {len(eval_tokens)} distinct tokens, "
          f"{len(unseen)} ({100 * len(unseen) / max(len(eval_tokens), 1):.1f}%) "
          f"never seen in training")
    print(f"wrote {train_out}\n      {eval_out}")


if __name__ == "__main__":
    main()
