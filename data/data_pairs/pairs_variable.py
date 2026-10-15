# coding=utf-8
"""Random pairs of generating vectors for training and validation.

Input: a directory of tokenised group CSVs (one {order}_{index}.csv per group,
columns id, sig, r, text; see data/data_tokenisation/tokenise_variable.py).
Output: train.csv and eval.csv in --out-dir, one pair per row:

    sig, text1, text2, target, id1, id2

where id1/id2 are class ids with the group appended ('{sig}#{k}#{order}_{index}'),
and target is 1 if the two vectors are in the same topological equivalence
class, 0 otherwise.

Sampling (seeded by --seed; train and eval are drawn one after the other from
the same generator, so a file is reproducible from its seed):

  positive pair   a class drawn uniformly among the classes with at least two
                  stored vectors, then two distinct vectors of that class;
  negative pair   a signature drawn uniformly among the signatures with at
                  least two classes, then two vectors of that signature from
                  different classes. The two classes may belong to different
                  groups (a "cross-group" pair) or to the same group.

Half of each file is positive. Rows are shuffled before writing: the model's
training loader only shuffles within a window of the file, so the file must
not be ordered by class or signature.

Nothing de-duplicates: train and eval are independent draws from the same
pool, so a class, a vector or even a pair can occur in both.

--sig-len N restricts the pool to signatures of exactly N entries (the paper's
r = 4 arm). Texts are read from the group CSVs by byte offset, so the CSVs
are never held in memory.

    python data/data_pairs/pairs_variable.py --tok-dir DIR --out-dir DIR \\
        --pairs 2000000 --eval-pairs 60000 --seed 0 [--sig-len 4]
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from collections import defaultdict

csv.field_size_limit(10 ** 9)


def index_rows(tok_dir, sig_len=0):
    """Index every stored vector by class and by signature, without loading texts.

    Returns (id_map, sig_index):
      id_map[class id]    -> list of (csv path, byte offset) of the class's vectors
      sig_index[sig]      -> list of (class id, csv path, byte offset) of every vector
                             with that signature, across all groups
    Class ids are the CSV's '{sig}#{k}' with '#{group}' appended. With sig_len > 0,
    only signatures of exactly that many entries are indexed.
    """
    id_map = defaultdict(list)
    sig_index = defaultdict(list)
    for name in sorted(os.listdir(tok_dir)):
        if not name.endswith(".csv"):
            continue
        group = os.path.splitext(name)[0]
        path = os.path.join(tok_dir, name)
        with open(path, encoding="utf-8") as f:
            f.readline()                      # header
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                cls_id, sig = line.split(",", 2)[:2]
                if sig_len and sig.count("_") + 1 != sig_len:
                    continue
                full_id = f"{cls_id}#{group}"
                id_map[full_id].append((path, offset))
                sig_index[sig].append((full_id, path, offset))
    return id_map, sig_index


_open = {}
_MAX_OPEN = 256                               # bounded pool of open CSVs (thousands of groups)


def read_text(path, offset):
    """The text column of the row at byte offset `offset` of the group CSV `path`."""
    f = _open.pop(path, None)
    if f is None:
        if len(_open) >= _MAX_OPEN:
            _open.pop(next(iter(_open))).close()   # evict the least recently used
        f = open(path, encoding="utf-8")
    _open[path] = f                           # re-insert as most recently used
    f.seek(offset)
    row = next(csv.reader([f.readline()]))
    return row[3]                             # id, sig, r, text


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tok-dir", required=True, help="tokenised group CSVs")
    p.add_argument("--out-dir", required=True, help="where train.csv and eval.csv go")
    p.add_argument("--pairs", type=int, default=500000, help="training pairs")
    p.add_argument("--eval-pairs", type=int, default=30000, help="validation pairs")
    p.add_argument("--sig-len", type=int, default=0,
                   help="use only signatures with exactly N entries (0 = all)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    id_map, sig_index = index_rows(args.tok_dir, args.sig_len)
    pos_classes = [c for c, rows in id_map.items() if len(rows) > 1]
    multi_sigs = [s for s, e in sig_index.items() if len({fid for fid, _, _ in e}) > 1]
    print(f"indexed: {sum(len(v) for v in id_map.values())} rows, "
          f"{len(id_map)} classes ({len(pos_classes)} multi-row), "
          f"{len(sig_index)} signatures ({len(multi_sigs)} multi-class)")
    if not pos_classes or not multi_sigs:
        raise SystemExit("insufficient structure for pair generation")

    def gen(n_pairs):
        rows = []
        for _ in range(n_pairs // 2):                             # positives
            c = rng.choice(pos_classes)
            (p1, o1), (p2, o2) = rng.sample(id_map[c], 2)
            rows.append([c.rsplit("#", 2)[0], read_text(p1, o1), read_text(p2, o2), 1, c, c])
        for _ in range(n_pairs - n_pairs // 2):                   # negatives
            s = rng.choice(multi_sigs)
            while True:
                (c1, p1, o1), (c2, p2, o2) = rng.sample(sig_index[s], 2)
                if c1 != c2:
                    break
            rows.append([s, read_text(p1, o1), read_text(p2, o2), 0, c1, c2])
        rng.shuffle(rows)
        return rows

    os.makedirs(args.out_dir, exist_ok=True)
    header = ["sig", "text1", "text2", "target", "id1", "id2"]
    for split, n in (("train", args.pairs), ("eval", args.eval_pairs)):
        rows = gen(n)
        with open(os.path.join(args.out_dir, f"{split}.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        pos = sum(1 for r in rows if r[3] == 1)
        print(f"{split}: {len(rows)} pairs ({pos} pos / {len(rows) - pos} neg)")


if __name__ == "__main__":
    main()
