# coding=utf-8
"""Merge each group's per-signature orbit items into one group file (run under SageMath).

orbits_fast.py leaves one result per (group, signature) item in the staging
directory: ``{order}_{index}__{signature}.sobj`` when computed, or
``{order}_{index}__{signature}.skip.json`` when the signature was skipped (no
generating vectors, or over a limit). A group is combined only when

  - every one of its items (listed from the inventory with the same filters as
    make_workitems.py) has one of the two, and
  - all of its computed items use the same permutation representation of the
    group (see the representation caveat in orbits_fast.py).

The group file ``{order}_{index}.sobj`` in the orbit store holds
``[[signature, classes], ...]`` for the computed items, ordered by (r, signature).
Groups already in the orbit store are left alone. Each other group gets
one status line: COMBINED, INCOMPLETE (items still pending: rerun the orbit
stage), NOTHING_COMPUTED (every signature skipped) or REPRESENTATION_MISMATCH.

  sage -python combine_orbits.py --inventory FILE \\
      --staging-dir DIR --orbits-dir DIR --max-order 500 --max-r 16
"""

import argparse
import os
import sys
from collections import Counter

from sage.all import PermutationGroup, load, save

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_workitems import read_inventory  # noqa: E402


def item_name(gid, sig):
    return f"{gid[0]}_{gid[1]}__{'_'.join(map(str, sig))}"


def same_representation(items):
    """Whether all items were computed in the same permutation representation.

    Every generating vector generates the whole group, so the first vector of
    the first item generates the reference group G. Another item's first vector
    generates a group of the same order; if its elements all lie in G, it is G.
    """
    ref = PermutationGroup(list(items[0][1][0][0]))
    return all(g in ref for _, classes in items[1:] for g in classes[0][0])


def combine_group(gid, sigs, staging_dir, dest):
    """Combine one group if all its items are resolved; return its status."""
    computed, pending = [], 0
    for sig in sorted(sigs, key=lambda s: (len(s), s)):
        base = os.path.join(staging_dir, item_name(gid, sig))
        if os.path.exists(base + ".sobj"):
            computed.append(base + ".sobj")
        elif not os.path.exists(base + ".skip.json"):
            pending += 1
    if pending:
        return f"INCOMPLETE ({pending} of {len(sigs)} items pending)"
    if not computed:
        return "NOTHING_COMPUTED"
    items = [load(f) for f in computed]
    if not same_representation(items):
        return "REPRESENTATION_MISMATCH"
    partial_dir = os.path.join(os.path.dirname(dest), ".partial")
    os.makedirs(partial_dir, exist_ok=True)
    tmp = os.path.join(partial_dir, os.path.basename(dest))
    save(items, tmp)
    os.replace(tmp, dest)
    return f"COMBINED ({len(computed)} signatures)"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inventory", required=True,
                    help="group/signature inventory (groupsignaturedata.txt)")
    ap.add_argument("--staging-dir", required=True, help="per-item results from orbits_fast.py")
    ap.add_argument("--orbits-dir", required=True, help="orbit store: combined group files")
    ap.add_argument("--max-order", type=int, required=True, help="as for make_workitems.py")
    ap.add_argument("--max-r", type=int, required=True, help="as for make_workitems.py")
    args = ap.parse_args()

    inv = read_inventory(args.inventory, args.max_order, args.max_r)
    os.makedirs(args.orbits_dir, exist_ok=True)

    outcomes = Counter()
    for gid in sorted(inv):
        dest = os.path.join(args.orbits_dir, f"{gid[0]}_{gid[1]}.sobj")
        if os.path.exists(dest):
            outcomes["ALREADY_COMBINED"] += 1
            continue
        status = combine_group(gid, inv[gid], args.staging_dir, dest)
        print(f"{gid[0]}_{gid[1]}: {status}", flush=True)
        outcomes[status.split()[0]] += 1
    print("done: " + ", ".join(f"{k} {v}" for k, v in sorted(outcomes.items())))


if __name__ == "__main__":
    main()
