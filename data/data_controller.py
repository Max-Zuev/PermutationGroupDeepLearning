# coding=utf-8
"""DataController: the single interface between computational scripts and stored data.

All naming lives in ``data/config.py`` (root, directory names, dataset
registry); this module is pure logic: what exists, what remains, and how to
read it. Infrastructure changes are absorbed by config.py; pipeline-shape
changes by this file; computational scripts see neither.

Storage model (one store per stage -- no historical "waves"):
  inventory   INVENTORY                        (g0=0 filtered on load)
  orbits      ORBITS_DIR/{o}_{i}.sobj          final, one file per group
              ORBITS_STAGING/{o}_{i}__{sig}.sobj + .sizes.json
                                               in-flight items + full-size sidecars
  tokenised   TOKENISED[universe]/{o}_{i}.csv  (columns id,sig,r,text)
  datasets    DATASETS registry                (canonical paper names)

CLI (run anywhere the data root is reachable):
  python data/data_controller.py status
  python data/data_controller.py missing --stage orbits|tokenised
  python data/data_controller.py datasets
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402  (data/config.py -- the layout)

csv.field_size_limit(10 ** 9)

_GID_FILE = re.compile(r"^(\d+)_(\d+)\.(sobj|csv)$")


def gid_str(gid: tuple[int, int]) -> str:
    return f"{gid[0]}_{gid[1]}"


def parse_gid(name: str) -> tuple[int, int]:
    o, i = name.split("_")
    return int(o), int(i)


@dataclass
class DataStatus:
    root: str
    inventory_groups: int = 0
    inventory_items: int = 0
    orbit_groups: int = 0
    orbits_missing: int = 0
    tokenised: dict = field(default_factory=dict)      # universe -> group count
    datasets: dict = field(default_factory=dict)       # name -> (exists, bytes)

    def report(self) -> str:
        lines = [f"data root: {self.root}",
                 f"inventory: {self.inventory_groups} groups / "
                 f"{self.inventory_items} (group, signature) items",
                 f"orbits computed: {self.orbit_groups} groups "
                 f"({self.orbit_groups / max(self.inventory_groups, 1):.1%}), "
                 f"missing {self.orbits_missing}"]
        for u, n in self.tokenised.items():
            lines.append(f"tokenised[{u}]: {n} groups")
        for name, (ok, size) in self.datasets.items():
            lines.append(f"dataset {name:<22} "
                         + (f"OK  {size / 1e6:8.1f} MB" if ok else "MISSING"))
        return "\n".join(lines)


class DataController:

    def __init__(self, root: str | os.PathLike | None = None):
        self.root = Path(root) if root is not None else Path(config.DATA_ROOT)
        self._inventory: dict[tuple[int, int], set[tuple[int, ...]]] | None = None

    # ---- inventory ---------------------------------------------------------
    def inventory(self) -> dict[tuple[int, int], set[tuple[int, ...]]]:
        """gid -> set of signatures (quotient genus 0 only). Cached."""
        if self._inventory is None:
            import ast
            items: dict[tuple[int, int], set[tuple[int, ...]]] = {}
            if not (self.root / config.INVENTORY).exists():
                raise SystemExit(f"inventory not found: {self.root / config.INVENTORY}\n"
                                 "(set storage.root in data/pipeline.json; see REPRODUCING.md step 0)")
            with open(self.root / config.INVENTORY, encoding="utf-8") as f:
                for line in f:
                    row = ast.literal_eval(
                        line.replace("[*", "[").replace("*]", "]"))
                    fsig = tuple(row[2])
                    if fsig[0] > 0:
                        continue
                    items.setdefault(tuple(row[3]), set()).add(tuple(fsig[1:]))
            self._inventory = items
        return self._inventory

    # ---- orbits ------------------------------------------------------------
    def orbit_path(self, gid) -> Path | None:
        p = self.root / config.ORBITS_DIR / f"{gid_str(gid)}.sobj"
        return p if p.exists() else None

    def orbits_computed(self) -> set[tuple[int, int]]:
        d = self.root / config.ORBITS_DIR
        if not d.is_dir():
            return set()
        return {(int(m.group(1)), int(m.group(2)))
                for f in d.iterdir()
                if (m := _GID_FILE.match(f.name)) and m.group(3) == "sobj"}

    def orbits_missing(self) -> dict[tuple[int, int], set[tuple[int, ...]]]:
        """Inventory groups with no orbit file (the uncomputed tail)."""
        done = self.orbits_computed()
        return {g: s for g, s in self.inventory().items() if g not in done}

    def load_orbits(self, gid):
        """[[signature, [class -> [generating vectors]]], ...]. Needs Sage."""
        p = self.orbit_path(gid)
        if p is None:
            raise FileNotFoundError(f"no orbit file for {gid_str(gid)}")
        from sage.all import load                   # lazy: Sage-only method
        return load(str(p))

    def orbit_sizes(self, gid) -> dict[tuple[int, ...], list[int]]:
        """FULL orbit sizes (pre-cap) from staging sidecars: sig -> sizes."""
        out = {}
        d = self.root / config.ORBITS_STAGING
        if d.is_dir():
            for f in d.glob(f"{gid_str(gid)}__*.sizes.json"):
                rec = json.loads(f.read_text())
                out[tuple(rec["sig"])] = rec["orbit_sizes"]
        return out

    # ---- tokenised ---------------------------------------------------------
    def tokenised_dir(self, universe: str = "union") -> Path:
        return self.root / config.TOKENISED[universe]

    def tokenised_path(self, gid, universe: str = "union") -> Path | None:
        p = self.tokenised_dir(universe) / f"{gid_str(gid)}.csv"
        return p if p.exists() else None

    def tokenised_computed(self, universe: str = "union") -> set[tuple[int, int]]:
        d = self.tokenised_dir(universe)
        if not d.is_dir():
            return set()
        return {parse_gid(f.stem) for f in d.iterdir()
                if _GID_FILE.match(f.name) and f.suffix == ".csv"}

    def tokenised_missing(self, universe: str = "union") -> set[tuple[int, int]]:
        """Orbit-computed groups not yet in this tokenisation."""
        return self.orbits_computed() - self.tokenised_computed(universe)

    def iter_tokenised(self, gid, universe: str = "union"):
        """Yield rows (class_id, sig, r, text) for one group."""
        p = self.tokenised_path(gid, universe)
        if p is None:
            raise FileNotFoundError(f"{gid_str(gid)} not in {universe}")
        with open(p, encoding="utf-8") as f:
            rdr = csv.reader(f)
            next(rdr)
            for row in rdr:
                yield row[0], row[1], int(row[2]), row[3]

    def class_counts(self, gid, universe: str = "union") -> Counter:
        """class_id -> number of stored vectors (post-cap)."""
        c = Counter()
        for cls_id, *_ in self.iter_tokenised(gid, universe):
            c[cls_id] += 1
        return c

    # ---- datasets ----------------------------------------------------------
    def dataset_path(self, name: str) -> Path:
        return self.root / config.DATASETS[name]

    def dataset_status(self) -> dict[str, tuple[bool, int]]:
        out = {}
        for name in config.DATASETS:
            p = self.dataset_path(name)
            out[name] = (p.exists(), p.stat().st_size if p.exists() else 0)
        return out

    # ---- status ------------------------------------------------------------
    def status(self) -> DataStatus:
        inv = self.inventory()
        s = DataStatus(root=str(self.root),
                       inventory_groups=len(inv),
                       inventory_items=sum(len(v) for v in inv.values()),
                       orbit_groups=len(self.orbits_computed()),
                       orbits_missing=len(self.orbits_missing()),
                       datasets=self.dataset_status())
        for u in config.TOKENISED:
            s.tokenised[u] = len(self.tokenised_computed(u))
        return s


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["status", "missing", "datasets"])
    ap.add_argument("--root", default=None)
    ap.add_argument("--stage", choices=["orbits", "tokenised"], default="orbits")
    ap.add_argument("--universe", default="union")
    args = ap.parse_args()
    dc = DataController(args.root)

    if args.command == "status":
        print(dc.status().report())
    elif args.command == "missing":
        if args.stage == "orbits":
            miss = dc.orbits_missing()
            for g in sorted(miss):
                print(f"{gid_str(g)}\t{len(miss[g])} signatures")
            print(f"-- {len(miss)} groups missing orbits")
        else:
            for g in sorted(dc.tokenised_missing(args.universe)):
                print(gid_str(g))
            print(f"-- {len(dc.tokenised_missing(args.universe))} groups "
                  f"missing from tokenised[{args.universe}]")
    elif args.command == "datasets":
        for name, (ok, size) in dc.dataset_status().items():
            print(f"{name:<22} " + (f"OK  {size / 1e6:8.1f} MB" if ok else "MISSING"))


if __name__ == "__main__":
    main()
