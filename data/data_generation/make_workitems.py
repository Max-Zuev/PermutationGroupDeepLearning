# coding=utf-8
"""List the (group, signature) work items still to compute for orbits_fast.py.

A work item is one signature of one group. Items come from the inventory (one
row per (genus, |G|, [g0, c_1, ..., c_r], group id)), restricted to

  - quotient genus 0 (the only case orbits_fast.py computes),
  - |G| <= --max-order and r <= --max-r,

minus groups that already have a file in the orbit store and items already
resolved in the staging directory (a computed .sobj or a .skip.json marker), so
a rerun resumes. Items are written grouped by group id, and within a group by
(r, signature).

Output: one line per item, ``order index c_1 ... c_r``.

  python3 make_workitems.py --inventory INVENTORY \\
      --orbits-dir DIR --staging-dir DIR --max-order 500 --max-r 16 --out FILE
"""

from __future__ import annotations

import argparse
import ast
import os

# orbits_fast.py enumerates vectors (gamma_1, ..., gamma_r) with product 1, i.e.
# no alpha/beta generators: it only handles quotient genus 0. Not a setting.
QUOTIENT_GENUS = 0


def read_inventory(path, max_order, max_r):
    """group id -> set of signatures, for quotient genus 0 within the caps."""
    items = {}
    if not os.path.exists(path):
        raise SystemExit(f"inventory not found: {path}\n(Paulhus's groupsignaturedata.txt, "
                         "expected at storage.root/storage.inventory of data/pipeline.json)")
    with open(path) as f:
        for line in f:
            row = ast.literal_eval(line.replace("[*", "[").replace("*]", "]"))
            fsig = tuple(row[2])                  # (g0, c_1, ..., c_r)
            if fsig[0] != QUOTIENT_GENUS:
                continue
            sig, gid = fsig[1:], tuple(row[3])
            if gid[0] > max_order or len(sig) > max_r:
                continue
            items.setdefault(gid, set()).add(tuple(sig))
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inventory", required=True,
                    help="group/signature inventory (groupsignaturedata.txt)")
    ap.add_argument("--orbits-dir", required=True,
                    help="orbit store: groups with a {order}_{index}.sobj here are done")
    ap.add_argument("--staging-dir", required=True,
                    help="per-item results: items with a .sobj or .skip.json here are done")
    ap.add_argument("--max-order", type=int, required=True, help="largest |G| to include")
    ap.add_argument("--max-r", type=int, required=True,
                    help="largest signature length to include")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    inv = read_inventory(args.inventory, args.max_order, args.max_r)

    out, skipped_done = [], 0
    for gid in sorted(inv):
        if os.path.exists(os.path.join(args.orbits_dir, f"{gid[0]}_{gid[1]}.sobj")):
            skipped_done += 1
            continue
        for sig in sorted(inv[gid], key=lambda s: (len(s), s)):
            base = os.path.join(args.staging_dir, f"{gid[0]}_{gid[1]}__{'_'.join(map(str, sig))}")
            if os.path.exists(base + ".sobj") or os.path.exists(base + ".skip.json"):
                continue
            out.append((gid, sig))

    with open(args.out, "w") as f:
        for gid, sig in out:
            f.write(f"{gid[0]} {gid[1]} {' '.join(map(str, sig))}\n")
    n_groups = len({g for g, _ in out})
    print(f"{len(out)} work items across {n_groups} groups "
          f"(skipped {skipped_done} already-complete groups) -> {args.out}")


if __name__ == "__main__":
    main()
