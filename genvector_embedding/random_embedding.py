# coding=utf-8
"""The RANDOM embedding initialisation, norm-matched to the analytic one.

The paper compares two initialisations of the 576-dimensional token embedding:
the analytic matrix embedding (analytic.py: each transposition token is its
24 x 24 permutation matrix, flattened) and a random one of the same dimension,
so that any difference in results is due to structure rather than size. The
random rows are scaled to the same Euclidean norm as the analytic rows,
sqrt(d) (a d x d permutation matrix has d ones), so the two tables also start
at the same scale.

    python -m genvector_embedding.random_embedding --dim 24 --num-vocab 300000 \
        --seed 0 --out-dir .../embeddings/random_normmatched_d576

Writes embedding.npy [num_vocab, d*d] (float32) and meta.json.

The file the paper's runs used (random_normmatched_d576.npy) predates this
script; its seed and exact recipe were not recorded, so a table generated here
has the same distribution but is not that file.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def build_random(num_vocab: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = rng.standard_normal((num_vocab, d * d)).astype(np.float32)
    out *= np.sqrt(d) / np.linalg.norm(out, axis=1, keepdims=True)   # every row: norm sqrt(d)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dim", type=int, default=24, help="d: rows have d*d entries and norm sqrt(d)")
    p.add_argument("--num-vocab", type=int, default=300000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    mat = build_random(args.num_vocab, args.dim, args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "embedding.npy"), mat)
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"config": {"matrix_dim": args.dim, "random": True, "seed": args.seed,
                              "row_norm": float(np.sqrt(args.dim))},
                   "num_vocab": args.num_vocab, "export_dim": args.dim * args.dim}, f, indent=2)
    print(f"wrote {args.out_dir}/embedding.npy [{args.num_vocab} x {args.dim * args.dim}], "
          f"rows of norm {np.sqrt(args.dim):.3f}, seed {args.seed}")


if __name__ == "__main__":
    main()
