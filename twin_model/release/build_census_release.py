# coding=utf-8
"""Build results/census_predictions.csv: the data-mining release.

Joins a census confidence dump (predict.py output: row,id1,id2,target,prob)
with LMFDB group properties (lmfdb/cache.json, filled by
lmfdb/prefetch.py) and a pair-category partition, decided from the cached
``cyclic`` / ``abelian`` flags and the group order:

    cross_group             different groups
    same_group_cyclic       cyclic G
    same_group_abelian      abelian non-cyclic G
    same_group_p_group      non-abelian G of prime-power order
    same_group_non_abelian  everything else

Groups absent from the cache (or marked not_in_lmfdb) keep their category but
get blank property columns. Build the cache with lmfdb/prefetch.py (or
mirror_fetch.py) and complete it with lmfdb/gap_records.g +
merge_gap_records.py: the nine groups of order 512 and 1536 are not in the
LMFDB, and a few LMFDB fields are empty. For the released file every cached
value was checked field by field against the GAP computation.

  python twin_model/release/build_census_release.py \
      --conf conf_1000k_ucensus.csv --out results/census_predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROPS = ["order", "abelian", "cyclic", "nilpotent", "supersolvable", "solvable",
         "metacyclic", "metabelian", "exponent", "derived_length", "aut_order",
         "number_conjugacy_classes"]
BOOLS = {"abelian", "cyclic", "nilpotent", "supersolvable", "solvable",
         "metacyclic", "metabelian"}


def is_prime_power(n: int) -> bool:
    for q in range(2, int(n ** 0.5) + 1):
        if n % q == 0:
            while n % q == 0:
                n //= q
            return n == 1
    return True


def prop_cols(cache, group):
    rec = cache.get(group.replace("_", "."))
    if not rec or rec.get("not_in_lmfdb"):
        return [""] * len(PROPS)
    out = []
    for p in PROPS:
        v = rec.get(p)
        if v is None:
            out.append("")
        elif p in BOOLS:
            out.append(int(bool(v)))
        else:
            out.append(v)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--conf", required=True, help="predict.py output CSV")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=os.path.join(HERE, "lmfdb", "cache.json"))
    args = ap.parse_args()

    if not os.path.exists(args.cache):
        raise SystemExit(f"{args.cache} not found: build it with lmfdb/prefetch.py "
                         "(see REPRODUCING.md step 6)")
    cache = json.load(open(args.cache, encoding="utf-8"))

    def category(g1, g2):
        if g1 != g2:
            return "cross_group"
        rec = cache.get(g1.replace("_", "."))
        if rec is None or rec.get("not_in_lmfdb") or rec.get("cyclic") is None:
            raise SystemExit(f"{g1}: no cyclic/abelian flags in the LMFDB cache")
        if rec["cyclic"]:
            return "same_group_cyclic"
        if rec["abelian"]:
            return "same_group_abelian"
        if is_prime_power(int(g1.split("_")[0])):
            return "same_group_p_group"
        return "same_group_non_abelian"

    n = correct = 0
    from collections import Counter
    cats = Counter()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["class_id_1", "class_id_2", "group_1", "group_2", "signature",
                    "pair_category", "same_class", "model_p_same", "model_correct"]
                   + [f"{p}_1" for p in PROPS] + [f"{p}_2" for p in PROPS])
        for r in csv.DictReader(open(args.conf, encoding="utf-8")):
            id1, id2 = r["id1"], r["id2"]
            g1, g2 = id1.rsplit("#", 1)[-1], id2.rsplit("#", 1)[-1]
            sig = id1.split("#", 1)[0].replace("_", ";")
            t, p = int(r["target"]), float(r["prob"])
            ok = int((p >= 0.5) == (t == 1))
            cat = category(g1, g2)
            w.writerow([id1, id2, g1, g2, sig, cat, t, f"{p:.6f}", ok]
                       + prop_cols(cache, g1) + prop_cols(cache, g2))
            n += 1
            correct += ok
            cats[cat] += 1
    print(f"wrote {n} rows to {args.out}; accuracy {correct / n:.4f}")
    for c, k in sorted(cats.items()):
        print(f"  {c}: {k}")


if __name__ == "__main__":
    main()
