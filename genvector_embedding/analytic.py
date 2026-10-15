# coding=utf-8
"""The ANALYTIC embedding: token (i,j) -> the d x d permutation matrix of transposition (i,j).

The training objective (make generating-vector relations close: M(γ1)···M(γr) = I) has an
exact closed-form solution -- the permutation representation itself. Products of permutation
matrices compose exactly as the permutations do, so every relation of every group whose
representation moves points <= d closes EXACTLY, including groups never seen. No training,
no coverage limits, no interference; relation accuracy is 1.0 by construction.

Optionally conjugate every matrix by one fixed random orthogonal Q (M -> Q M Q^T): this
preserves all products/relations exactly while making the exported features dense instead
of sparse binary, which may suit downstream networks better.

    python -m genvector_embedding.analytic --dim 24 --largest-degree 500 \
        --num-vocab 300000 --out-dir genvector_embedding/runs/analytic_d24

Writes embedding.npy [num_vocab, d*d] + meta.json compatible with evaluate.py / export
conventions (tokens with points > d get the identity matrix).
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def build_analytic(num_vocab: int, d: int, largest_degree: int,
                   conjugate: bool = False, seed: int = 0) -> np.ndarray:
    n = largest_degree
    eye = np.eye(d, dtype=np.float32)
    out = np.tile(eye.flatten(), (num_vocab, 1))
    if conjugate:
        rng = np.random.default_rng(seed)
        Q, R = np.linalg.qr(rng.standard_normal((d, d)).astype(np.float32))
        Q = Q * np.sign(np.diag(R))
    count = 0
    for t in range(num_vocab):
        i, j = t % n + 1, t // n + 1
        if i == j or max(i, j) > d:
            continue                      # diagonal (identity) or outside the covered degree
        P = eye.copy()
        P[[i - 1, j - 1]] = P[[j - 1, i - 1]]   # swap rows -> permutation matrix of (i, j)
        if conjugate:
            P = Q @ P @ Q.T
        out[t] = P.flatten()
        count += 1

    # Signature-order tokens (masked data format writes them at sig_offset + c). They are
    # labels, not group elements, so identity-fill would make all signatures look identical
    # to a frozen model; give each a distinct random orthogonal matrix (norm-matched:
    # ||flattened orthogonal|| = sqrt(d), same as the permutation rows).
    sig_offset, num_sigs = largest_degree ** 2, 1024     # order tokens sit just past the transposition grid
    if num_vocab > sig_offset:
        rng_s = np.random.default_rng(seed + 1)
        A = rng_s.standard_normal((min(num_sigs, num_vocab - sig_offset), d, d)).astype(np.float32)
        Qs, Rs = np.linalg.qr(A)
        sgn = np.sign(np.diagonal(Rs, axis1=-2, axis2=-1)); sgn[sgn == 0] = 1.0
        Qs = Qs * sgn[:, None, :]
        out[sig_offset:sig_offset + Qs.shape[0]] = Qs.reshape(Qs.shape[0], d * d)

    print(f"analytic embedding: {count} transposition tokens covered (points <= {d}), "
          f"sig tokens {sig_offset}.. get distinct orthogonal vectors, rest identity  "
          f"[{num_vocab} x {d*d}]")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dim", type=int, default=24, help="matrix dim = covered degree")
    p.add_argument("--largest-degree", type=int, default=500, help="n of S_n numbering the transposition tokens (the tokeniser's largest_degree)")
    p.add_argument("--num-vocab", type=int, default=300000)
    p.add_argument("--conjugate", action="store_true",
                   help="conjugate by a fixed random orthogonal (dense features, same algebra)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    mat = build_analytic(args.num_vocab, args.dim, args.largest_degree,
                         conjugate=args.conjugate, seed=args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "embedding.npy"), mat)
    meta = {"config": {"matrix_dim": args.dim, "analytic": True,
                       "conjugate": args.conjugate, "seed": args.seed},
            "num_vocab": args.num_vocab, "n": args.largest_degree,
            "export_dim": args.dim * args.dim, "best_select_acc": 1.0,
            "selected_on": "analytic (exact by construction)"}
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {args.out_dir}/embedding.npy and meta.json")


if __name__ == "__main__":
    main()
