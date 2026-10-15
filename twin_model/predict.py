# coding=utf-8
"""Dump a checkpoint's per-pair probabilities on an eval CSV.

Output CSV: row, id1, id2, target, prob (probability of "same class"). The row
is the pair's 0-based index in the eval CSV, so downstream analysis can join
probabilities back to the input rows positionally: eval batches are bucketed by
number of generators, so output order != file order, and (id1, id2) alone is
not unique per pair (the same class pair can be sampled with different
vectors). Feeds build_census_release.py.

  python predict.py --checkpoint .../best.pt --out conf_idx.csv
"""

import argparse
import csv

import torch

from src.interface import Interface
from src.model import Model
from src.utils.loader import load_data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--eval-file", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu",
                            weights_only=False)
    margs = checkpoint["args"]
    margs.cuda = margs.cuda and torch.cuda.is_available()
    model = Model(margs, state_dict=checkpoint["state_dict"])
    data_dir = args.data_dir or margs.data_dir
    eval_file = args.eval_file or margs.eval_file
    print(f"checkpoint step {model.updates}; eval {data_dir}/{eval_file}; "
          f"cuda={margs.cuda}; concat_parts={getattr(margs, 'concat_parts', False)}")

    data = load_data(data_dir, eval_file, margs.part_length)
    for i, d in enumerate(data):
        d["id1"] = "%d|%s" % (i, d["id1"])
    interface = Interface(margs)
    batches = interface.pre_process(data, training=False)

    n = 0
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row", "id1", "id2", "target", "prob"])
        with torch.no_grad():
            for batch in batches:
                probs = model.predict(batch)
                for i1, i2, t, p in zip(batch["id1"], batch["id2"],
                                        batch["target"], probs):
                    idx, real_id1 = i1.split("|", 1)
                    w.writerow([idx, real_id1, i2, int(t), f"{p[1]:.6f}"])
                    n += 1
    print(f"wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()
