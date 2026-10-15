# coding=utf-8
"""Fill cache.json with LMFDB gps_groups properties for every dataset group.

cache.json holds one record per group label 'order.index' with the 20 fields
of fields.py; the release (build_census_release.py) reads group properties
from it. This fetches them through the LMFDB web API:

  - orders <= --bulk-max-order: bulk sweep by order (cheap; also caches sibling groups)
  - larger orders: per-label lookups only (order 512 has ~10M groups, 1536 ~4e8 -- a sweep
    is impossible; LMFDB excludes most of those orders anyway), with a page-capped sweep
    first when many labels of one order are needed
  - labels LMFDB doesn't hold get a {"not_in_lmfdb": true} marker so they are never
    re-fetched

Usage (from repo root):
  python twin_model/release/lmfdb/prefetch.py \
      --groups-file twin_model/release/lmfdb/groups.txt

The groups file is one SmallGroup id 'order_index' per line: the basenames of
the tokenised CSVs (``ls <storage.root>/<storage.tokenised.union>``). The
committed groups.txt lists the groups of the released census and of the
training universe; the bulk sweep also caches every group of order <= 200. Groups the LMFDB does not hold (orders 512 and 1536 here)
are computed in GAP instead: gap_records.g + merge_gap_records.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fields import PROPS, NUMERIC

API = "https://www.lmfdb.org/api/gps_groups/"


def get_json(url: str, retries: int = 4):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=45) as r:
                return json.load(r)
        except Exception as e:
            wait = 4.0 * (attempt + 1)
            print(f"  retry {attempt + 1} in {wait:.0f}s: {url} ({e})", file=sys.stderr)
            time.sleep(wait)
    return None


def slim(rec: dict) -> dict:
    return {k: rec.get(k) for k in PROPS + NUMERIC}


def save(cache: dict, path: str):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    os.replace(tmp, path)


def sweep_order(order: int, cache: dict, sleep: float, max_pages: int) -> bool:
    """Paginated fetch of all groups of one order.

    Returns True only when the sweep verifiably reached the last page -- absence from a
    complete sweep proves a label isn't in LMFDB. Page cap or network failure returns
    False so callers fall back to per-label instead of stamping not_in_lmfdb.
    """
    offset = 0
    for _ in range(max_pages):
        payload = get_json(f"{API}?order={order}&_format=json&_offset={offset}")
        if payload is None:
            return False                      # network failure: absence proves nothing
        data = payload.get("data", [])
        for rec in data:
            cache[rec["label"]] = slim(rec)
        if len(data) < 100:
            return True
        offset += len(data)
        time.sleep(sleep)
    print(f"  order {order}: page cap {max_pages} hit, falling back to per-label",
          file=sys.stderr)
    return False


def fetch_label(label: str, cache: dict) -> bool:
    """Exact per-label lookup; marks confirmed-absent labels. Returns success."""
    payload = get_json(f"{API}?label={label}&_format=json")
    if payload is None:
        return False                          # network failure: leave retryable
    data = payload.get("data", [])
    if data:
        cache[label] = slim(data[0])
    else:                                     # confirmed not in LMFDB (e.g. order 512/1536)
        cache[label] = {"order": int(label.split(".")[0]), "not_in_lmfdb": True}
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--groups-file", required=True,
                   help="one 'order_index' SmallGroup id per line")
    p.add_argument("--cache", default="twin_model/release/lmfdb/cache.json")
    p.add_argument("--bulk-max-order", type=int, default=200)
    p.add_argument("--bulk-min-labels", type=int, default=9,
                   help="above bulk-max-order, still try a capped sweep when an order "
                        "needs at least this many labels")
    p.add_argument("--max-pages", type=int, default=30)
    p.add_argument("--sleep", type=float, default=1.0)
    args = p.parse_args()

    with open(args.groups_file, encoding="utf-8") as f:
        labels = sorted({line.strip().replace("_", ".") for line in f if line.strip()})
    cache = {}
    if os.path.exists(args.cache):
        cache = json.load(open(args.cache, encoding="utf-8"))

    by_order = defaultdict(list)
    for l in labels:
        if not cache.get(l):
            by_order[int(l.split(".")[0])].append(l)
    total = sum(len(v) for v in by_order.values())
    print(f"universe: {len(labels)} labels; cached: {len(labels) - total}; "
          f"to fetch: {total} across {len(by_order)} orders")

    done = 0
    for order in sorted(by_order):
        needed = by_order[order]
        swept = False
        if order <= args.bulk_max_order or len(needed) >= args.bulk_min_labels:
            swept = sweep_order(order, cache, args.sleep, args.max_pages)
            time.sleep(args.sleep)
        for l in needed:
            if not cache.get(l):
                if swept and order <= args.bulk_max_order:
                    # full sweep completed but label absent -> confirmed not in LMFDB
                    cache[l] = {"order": order, "not_in_lmfdb": True}
                else:
                    fetch_label(l, cache)
                    time.sleep(args.sleep)
        done += len(needed)
        save(cache, args.cache)               # incremental: safe to interrupt/resume
        print(f"order {order}: {len(needed)} labels done ({done}/{total}); "
              f"cache now {len(cache)} records", flush=True)

    missing = [l for l in labels if not cache.get(l)]
    absent = [l for l in labels if cache.get(l, {}).get("not_in_lmfdb")]
    print(f"\ncomplete: {len(cache)} records in {args.cache}")
    print(f"dataset labels not in LMFDB: {len(absent)}"
          + (f" (e.g. {absent[:8]})" if absent else ""))
    if missing:
        print(f"WARNING: {len(missing)} labels still unfetched (network failures): "
              f"{missing[:8]}", file=sys.stderr)


if __name__ == "__main__":
    main()
