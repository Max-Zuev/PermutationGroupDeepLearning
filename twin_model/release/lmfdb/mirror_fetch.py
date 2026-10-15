# coding=utf-8
"""Bulk-fill the LMFDB cache from the official read-only PostgreSQL mirror.

The web API (used by prefetch.py) sits behind bot
protection that CAPTCHA-challenges an IP after a burst of requests. LMFDB's documented
answer for bulk access is the public mirror devmirror.lmfdb.xyz (anonymous read-only
credentials published in the LMFDB developer docs) -- one SQL round-trip replaces
thousands of HTTP requests.

Fetches all groups of order <= --bulk-max-order (cheap sibling caching) plus the exact
labels listed in --groups-file, writes the same slim record format into the shared cache,
and stamps dataset labels the mirror doesn't hold with {"not_in_lmfdb": true} (fill those
from GAP: gap_records.g + merge_gap_records.py).

Usage (from repo root):
  python twin_model/release/lmfdb/mirror_fetch.py \
      --groups-file twin_model/release/lmfdb/groups.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from decimal import Decimal

import psycopg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fields import PROPS, NUMERIC

DSN = "postgresql://lmfdb:lmfdb@devmirror.lmfdb.xyz:5432/lmfdb"
COLS = ["label"] + PROPS + NUMERIC


def jsonable(v):
    return int(v) if isinstance(v, Decimal) else v


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--groups-file", required=True,
                   help="one 'order_index' SmallGroup id per line")
    p.add_argument("--cache", default="twin_model/release/lmfdb/cache.json")
    p.add_argument("--bulk-max-order", type=int, default=200)
    args = p.parse_args()

    with open(args.groups_file, encoding="utf-8") as f:
        labels = sorted({line.strip().replace("_", ".") for line in f if line.strip()})
    cache = {}
    if os.path.exists(args.cache):
        cache = json.load(open(args.cache, encoding="utf-8"))
    before = len(cache)

    col_sql = ", ".join(f'"{c}"' for c in COLS)
    big = [l for l in labels if int(l.split(".")[0]) > args.bulk_max_order]
    with psycopg.connect(DSN, connect_timeout=30) as conn:
        with conn.cursor() as cur:
            cur.execute(f'SELECT {col_sql} FROM gps_groups WHERE "order" <= %s',
                        (args.bulk_max_order,))
            rows = cur.fetchall()
            cur.execute(f"SELECT {col_sql} FROM gps_groups WHERE label = ANY(%s)", (big,))
            rows += cur.fetchall()
    for row in rows:
        rec = dict(zip(COLS, (jsonable(v) for v in row)))
        cache[rec.pop("label")] = rec

    absent = []
    for l in labels:
        if not cache.get(l):
            cache[l] = {"order": int(l.split(".")[0]), "not_in_lmfdb": True}
            absent.append(l)

    tmp = args.cache + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    os.replace(tmp, args.cache)

    print(f"mirror rows fetched: {len(rows)}; cache {before} -> {len(cache)} records")
    covered = sum(1 for l in labels if cache.get(l) and not cache[l].get("not_in_lmfdb"))
    print(f"dataset coverage: {covered}/{len(labels)} labels with LMFDB data; "
          f"{len(absent)} not in LMFDB" + (f" (e.g. {absent[:8]})" if absent else ""))


if __name__ == "__main__":
    main()
