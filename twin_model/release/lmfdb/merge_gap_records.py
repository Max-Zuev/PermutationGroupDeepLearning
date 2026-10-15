# coding=utf-8
"""Merge or check GAP-computed group records (gap_records.g output) against cache.json.

  python merge_gap_records.py gap_records.json            # add records the cache lacks
                                                          # (or holds only as not_in_lmfdb)
  python merge_gap_records.py gap_records.json --check    # compare field by field, change nothing
  python merge_gap_records.py gap_records.json --overwrite

The two sources define every field identically, so --check should report no
differences; the cache behind the released CSV was checked this way over all
its records, and the fields the LMFDB left empty were filled with --overwrite.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fields import PROPS, NUMERIC


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("gap_json")
    p.add_argument("--cache", default=os.path.join(HERE, "cache.json"))
    p.add_argument("--check", action="store_true", help="report differences, write nothing")
    p.add_argument("--overwrite", action="store_true", help="replace cached records too")
    args = p.parse_args()

    gap = json.load(open(args.gap_json, encoding="utf-8"))
    cache = json.load(open(args.cache, encoding="utf-8")) if os.path.exists(args.cache) else {}

    if args.check:
        compared = differ = absent = 0
        for label, rec in gap.items():
            old = cache.get(label)
            if not old or old.get("not_in_lmfdb"):
                absent += 1
                continue
            compared += 1
            for k in PROPS + NUMERIC:
                if old.get(k) != rec[k]:
                    differ += 1
                    print(f"{label}.{k}: cache {old.get(k)!r} gap {rec[k]!r}")
        print(f"compared {compared} records, {differ} differing fields, "
              f"{absent} labels not in the cache")
        return

    added = replaced = 0
    for label, rec in gap.items():
        old = cache.get(label)
        if old and not old.get("not_in_lmfdb") and not args.overwrite:
            continue
        replaced += bool(old)
        added += not old
        cache[label] = {k: rec[k] for k in PROPS + NUMERIC}
    tmp = args.cache + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    os.replace(tmp, args.cache)
    print(f"cache: {added} records added, {replaced} replaced; {len(cache)} total")


if __name__ == "__main__":
    main()
