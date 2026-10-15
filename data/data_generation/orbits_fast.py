# coding=utf-8
"""Generating vectors and their topological equivalence classes (run under SageMath).

For a group G = SmallGroup(order, index) and a signature (c_1, ..., c_r) with
quotient genus 0, this finds every generating vector (gamma_1, ..., gamma_r):
elements generating G, gamma_k of order c_k, with product gamma_1 ... gamma_r = 1
(GAP's left-to-right product), and partitions them into classes: orbits under
Aut(G) and the braid moves (gamma_k, gamma_{k+1}) -> (gamma_k gamma_{k+1}
gamma_k^-1, gamma_k).

The file has four parts, in reading order:

  1. GroupTables       build G once and turn it into integer tables, so that
                       everything after this is arithmetic on small integers
  2. enumerate_vectors find all generating vectors of one signature
     partition_orbits  split them into classes
  3. run_item          do both for one (group, signature) and write the files
     validate          recompute an existing group file and compare
  4. main              command line: plan worker, single group, or validate

Elements of G are numbered 0 .. n-1 (n = |G|) in the order GAP lists them, and
a generating vector is a tuple of r such numbers. All group arithmetic goes
through GroupTables.mult, so the product convention is fixed in one place:
mult[a*n + b] is "a then b", GAP's a*b.

A work item is one (group, signature). Its outputs, in the output directory:

  {order}_{index}__{signature}.sobj   [signature, classes]; each class a list of at
                                      most max_per_class vectors (Sage permutation
                                      group elements), the first found by the search
                                      from a fixed seed, not a random sample
  {order}_{index}__{signature}.sizes.json
                                      number of signature-ordered vectors, and each
                                      class's full orbit size (counting vectors in
                                      every ordering, not just signature order)

combine_orbits.py later merges a group's items into {order}_{index}.sobj.

An item is skipped if its search space (search_space_size) exceeds
max_search_space, or if a class grows past max_orbit_size vectors (memory
guard); an item with no generating vectors is skipped too. Skipped items get a
{order}_{index}__{signature}.skip.json marker and are not planned again (delete
the markers to retry after raising a limit).

Modes:
  plan worker:  sage -python orbits_fast.py --plan PLAN --worker K --report R.tsv \\
                    --out-dir DIR --max-per-class N --max-search-space X --max-orbit-size M
  single group: sage -python orbits_fast.py --gid 96 13 --sig 2 4 12 [--sig ...] \\
                    --out-dir DIR --max-per-class N --max-search-space X --max-orbit-size M
  validate:     sage -python orbits_fast.py --gid 24 12 --validate GROUP_FILE_DIR
                    (recompute every signature of an existing {order}_{index}.sobj and
                    check the classes agree)

Caveat: SmallerDegreePermutationRepresentation is not deterministic across GAP
sessions, so the same group built in two processes can get two different
permutation representations. The plan keeps all of a group's items on one worker,
so they share one representation within a run; items of one group computed in
different runs (e.g. after a timeout) can still differ. This happened to 108_39
and 128_924, which are excluded downstream.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

from sage.all import PermutationGroup, Permutation, libgap, load, save


# =============================================================================
# 1. The group as integer tables
# =============================================================================

class GroupTables:
    """SmallGroup(order, index) as integer tables over its elements 0 .. n-1.

    Attributes:
      gid        (order, index)
      G_gap      the group as a GAP permutation group
      n          |G|
      deg        the number of points the permutation representation acts on
      el_imgs    el_imgs[a] = the permutation of element a, as the list of
                 images of the points 1 .. deg (1-based, as GAP writes them)
      mult       flat n*n table: mult[a*n + b] = index of a*b ("a then b")
      identity   index of the identity element
      inv        inv[a] = index of a^-1
      order_of   order_of[a] = order of element a
      by_order   by_order[c] = list of indices of the elements of order c
      aut_gens   one tuple per generator phi of Aut(G): aut_gens[j][a] = index
                 of phi(a), so an automorphism is a permutation of 0 .. n-1

    Everything downstream reads these tables and never touches GAP again, which
    is where the speed comes from: a vector is a tuple of small ints, a product
    is one list lookup, and an automorphism or braid move is a few lookups.
    """

    def __init__(self, order, index):
        self.gid = (order, index)
        self._build_permutation_group(order, index)
        self._build_multiplication_table()
        self._build_inverse_and_order_tables()
        self._build_automorphism_generators()
        # generates() caches its answers here; run_item clears it between items
        self._closure_memo = {}

    # ---- construction steps ------------------------------------------------

    def _build_permutation_group(self, order, index):
        """Get G from GAP's SmallGroups library as a permutation group, and list
        its elements once. The element list fixes the numbering 0 .. n-1."""
        g0 = libgap.SmallGroup(order, index)
        # SmallGroup gives a pc-group or similar; make it a permutation group,
        # then ask GAP for a representation on fewer points (the tokeniser needs
        # the degree to be <= 500, and smaller means shorter words)
        im = libgap.Image(libgap.IsomorphismPermGroup(g0))
        self.G_gap = libgap.Image(libgap.SmallerDegreePermutationRepresentation(im))
        self.n = int(self.G_gap.Size())
        assert self.n == order
        # the trivial group moves no points: give it degree 1 so lists are non-empty
        self.deg = max(1, int(self.G_gap.LargestMovedPoint()))
        self._els = self.G_gap.Elements()          # GAP's element list, fixed order
        # ListPerm(e, deg) = [e(1), e(2), ..., e(deg)], padded with fixed points
        self.el_imgs = [[int(x) for x in libgap.ListPerm(e, self.deg)] for e in self._els]
        # image list -> index, to turn any permutation back into its number
        self._pos = {tuple(img): a for a, img in enumerate(self.el_imgs)}
        self.identity = self._pos[tuple(range(1, self.deg + 1))]

    def _build_multiplication_table(self):
        """mult[a*n + b] = index of the product 'a then b'.

        Applying a then b to a point x gives b(a(x)). With 1-based image lists,
        a(x) = ia[x-1] and so b(a(x)) = ib[ia[x-1] - 1]. Iterating x over the
        points gives the image list of the product, which _pos turns back into
        an element index. This is GAP's product a*b (GAP composes left to right).
        """
        n, deg, pos = self.n, self.deg, self._pos
        mult = [0] * (n * n)
        for a in range(n):
            ia = self.el_imgs[a]
            for b in range(n):
                ib = self.el_imgs[b]
                mult[a * n + b] = pos[tuple(ib[ia[x] - 1] for x in range(deg))]
        self.mult = mult

    def _build_inverse_and_order_tables(self):
        """inv, order_of and by_order, all from the multiplication table.

        The order of a is the smallest o >= 1 with a^o = 1, found by multiplying
        by a until the identity comes back. Then a^-1 = a^(o-1).
        """
        n, mult = self.n, self.mult
        self.inv = [0] * n
        self.order_of = [0] * n
        for a in range(n):
            power, o = a, 1                        # power = a^o
            while power != self.identity:
                power = mult[power * n + a]
                o += 1
            self.order_of[a] = o
            power = self.identity
            for _ in range(o - 1):
                power = mult[power * n + a]
            self.inv[a] = power                    # a^(o-1)
        self.by_order = defaultdict(list)
        for a in range(n):
            self.by_order[self.order_of[a]].append(a)

    def _build_automorphism_generators(self):
        """Generators of Aut(G) as permutations of the element indices.

        GAP computes Aut(G) and gives a (usually small) generating set. For each
        generator phi we record where it sends every element: aut_gens[j][a] is
        the index of phi(element a). Only generators are needed: partition_orbits
        applies them repeatedly, which reaches the whole Aut(G)-orbit.
        """
        aut = self.G_gap.AutomorphismGroup()
        self.aut_gens = []
        for phi in aut.GeneratorsOfGroup():
            self.aut_gens.append(tuple(
                self._pos[tuple(int(x) for x in libgap.ListPerm(libgap.Image(phi, e), self.deg))]
                for e in self._els))

    # ---- queries ---------------------------------------------------------------

    def generates(self, entries):
        """Whether the elements `entries` generate G.

        Computes the subgroup they generate by closure: start from the identity
        and the entries, and keep multiplying everything found so far on the
        right by each entry until nothing new appears. In a finite group inverses
        are positive powers, so products of the entries alone give the whole
        subgroup. They generate G iff the closure has all n elements; the loop
        stops as soon as it does.

        The answer depends only on the SET of entries, so it is cached on that
        set: the many vectors that differ only by reordering or repetition of
        the same elements share one closure computation.
        """
        key = frozenset(entries)
        hit = self._closure_memo.get(key)
        if hit is not None:
            return hit
        n, mult = self.n, self.mult
        seen = {self.identity} | set(key)
        frontier = list(seen)                     # elements whose products are not yet taken
        gens = list(key)
        while frontier and len(seen) < n:
            nxt = []
            for h in frontier:
                base = h * n
                for g in gens:
                    e = mult[base + g]            # h * g
                    if e not in seen:
                        seen.add(e)
                        nxt.append(e)
            frontier = nxt
        res = len(seen) == n
        self._closure_memo[key] = res
        return res


# =============================================================================
# 2. Generating vectors of one signature, and their classes
# =============================================================================

def search_space_size(tables, sig):
    """Number of candidate tuples enumerate_vectors tries for this signature.

    The search fills the first r-1 positions with elements of the required
    orders and derives the last one, so it visits
    |{elements of order c_1}| * ... * |{elements of order c_{r-1}}| tuples.
    Used to skip signatures that would take too long before starting them.
    """
    w = 1
    for c in sig[:-1]:
        w *= len(tables.by_order.get(c, []))
    return w


def enumerate_vectors(tables, sig):
    """All generating vectors of signature sig, as tuples of element indices.

    A vector (g_1, ..., g_r) must have order(g_k) = c_k and g_1 * ... * g_r = 1.
    The product condition determines the last element from the others:
    g_r = (g_1 * ... * g_{r-1})^-1. So the search runs over the first r-1
    positions only (depth-first, position by position, each ranging over the
    elements of order c_k), keeps the running product prods[d] = g_1 * ... * g_d
    as it goes, and at the bottom sets g_r to the inverse of the full product
    and checks two things: that g_r has the required order c_r, and that the
    r elements together generate G.

    The vectors come out in signature order: entry k has order c_k exactly.
    """
    n, mult, inv = tables.n, tables.mult, tables.inv
    slots = [tables.by_order.get(c, []) for c in sig[:-1]]   # candidates per position
    if not all(slots) or not tables.by_order.get(sig[-1]):
        return []                             # some required order has no elements in G
    last_order = sig[-1]
    order_of = tables.order_of
    out = []
    r1 = len(slots)                           # r - 1
    prefix = [0] * r1                         # the current choices for positions 0 .. r-2
    prods = [tables.identity] * (r1 + 1)      # prods[d] = product of prefix[0 .. d-1]

    def dfs(d):
        if d == r1:                           # all r-1 positions chosen
            last = inv[prods[r1]]             # forced last element
            if order_of[last] == last_order:
                vec = tuple(prefix) + (last,)
                if tables.generates(set(vec)):
                    out.append(vec)
            return
        base = prods[d] * n
        for a in slots[d]:
            prefix[d] = a
            prods[d + 1] = mult[base + a]     # prods[d] * a
            dfs(d + 1)

    dfs(0)
    return out


def partition_orbits(tables, sig, vectors, max_per_class, max_orbit_size):
    """Partition the signature-ordered vectors into classes.

    Two vectors are in the same class if one can be reached from the other by
    braid moves and automorphisms. The moves, on a vector v = (g_1, ..., g_r):

      braid at k:      (g_k, g_{k+1}) -> (g_k g_{k+1} g_k^-1, g_k)   (k = 1 .. r-1)
      automorphism:    (g_1, ..., g_r) -> (phi(g_1), ..., phi(g_r))  (phi a generator)

    Both preserve the product g_1 ... g_r = 1 and generation, so they map
    generating vectors to generating vectors. A braid move swaps the ORDERS at
    positions k and k+1 (the conjugate has the order of g_{k+1}), so it can
    leave signature order; such vectors are still part of the orbit and are
    followed, but they are not stored as members.

    Each class is grown by breadth-first search from a seed, the first vector in
    enumeration order that is not yet in a class, applying every move to every
    vector found until nothing new appears. Only the moves themselves are applied, never their
    inverses: on a finite orbit every move has finite order, so its inverse is
    a positive power of it and is reached anyway. The BFS therefore finds the
    full orbit of the seed under Aut(G) and the whole braid group; its members
    in signature order are exactly the orbit of the seed under Aut(G) and the
    signature-preserving braid subgroup (a braid takes one signature-ordered
    vector to another iff it permutes only equal signature entries), which are
    the classes the paper describes.

    Returns a list of (members, size) per class, in the order the classes were
    found: members are the first max_per_class signature-ordered vectors
    encountered (the seed first, then breadth-first order), and size counts
    every vector in the orbit, in signature order or not. Given the tables, the
    whole result is deterministic. Returns None if any orbit grows past
    max_orbit_size vectors, so the caller can skip the item instead of running
    out of memory.
    """
    n, mult, inv = tables.n, tables.mult, tables.inv
    order_of = tables.order_of
    sig_t = tuple(sig)
    unassigned = set(map(tuple, vectors))     # signature-ordered vectors not yet in a class
    classes = []
    r = len(sig)
    # seeds are taken in enumeration order (lexicographic in the element indices),
    # so the classes, their numbering and the stored members are deterministic
    for seed in map(tuple, vectors):
        if seed not in unassigned:
            continue
        unassigned.discard(seed)
        seen = {seed}                         # every vector of this orbit found so far
        frontier = [seed]                     # found in the last round, moves not yet applied
        members = [seed]                      # stored representatives (signature order only)
        size = 1
        while frontier:
            nxt = []
            for v in frontier:
                images = []
                for k in range(r - 1):
                    a, b = v[k], v[k + 1]
                    # a * b * a^-1, read left to right with the 'a then b' table
                    conj = mult[mult[a * n + b] * n + inv[a]]
                    images.append(v[:k] + (conj, a) + v[k + 2:])
                for phi in tables.aut_gens:
                    images.append(tuple(phi[x] for x in v))
                for w in images:
                    if w not in seen:
                        seen.add(w)
                        nxt.append(w)
                        size += 1
                        if size > max_orbit_size:
                            return None
                        if w in unassigned:   # a signature-ordered vector: now classified
                            unassigned.discard(w)
                        if tuple(order_of[x] for x in w) == sig_t and len(members) < max_per_class:
                            members.append(w)
            frontier = nxt
        classes.append((members, size))
    return classes


# =============================================================================
# 3. One work item: compute, write, or validate
# =============================================================================

def to_sage_vectors(tables, classes):
    """Turn each class's members from index tuples into Sage permutation group
    elements (what the orbit files store and the tokeniser reads)."""
    G = PermutationGroup(gap_group=tables.G_gap)
    els = [G(Permutation(img)) for img in tables.el_imgs]
    return [[[els[x] for x in vec] for vec in members] for members, _ in classes]


def _write_atomic(path, text):
    """Write a small file so that it is either complete or absent, never partial."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


# columns of the per-item report row that run_item returns
REPORT_COLUMNS = ("group", "signature", "status", "search_space", "n_vectors",
                  "n_classes", "seconds")


def run_item(tables, sig, out_dir, max_per_class, max_search_space, max_orbit_size):
    """Compute one (group, signature) item; return a dict of REPORT_COLUMNS.

    Writes {order}_{index}__{signature}.sobj ([signature, classes]) and a
    .sizes.json sidecar with the full class sizes. Both are written in
    out_dir/.partial and moved into out_dir when complete (sidecar first), so
    the .sobj exists only for finished items. A skipped item (no vectors, or
    over a limit -- deterministic outcomes) gets a .skip.json marker instead,
    so it is not planned again and combine_orbits.py can treat it as resolved.

    Statuses: OK, EXISTS (already done), SKIPPED_SEARCH_SPACE, NO_VECTORS,
    SKIPPED_ORBIT_SIZE.
    """
    o, i = tables.gid
    tag = "_".join(map(str, sig))
    rec = dict.fromkeys(REPORT_COLUMNS, "")
    rec.update(group=f"{o}_{i}", signature=tag)
    fname = os.path.join(out_dir, f"{o}_{i}__{tag}.sobj")
    skip_marker = fname.replace(".sobj", ".skip.json")
    if os.path.exists(fname) or os.path.exists(skip_marker):
        rec["status"] = "EXISTS"
        return rec

    def skipped(status):
        # record why, and with which limits, so the marker explains itself later
        rec["status"] = status
        _write_atomic(skip_marker, json.dumps({
            "sig": list(sig), "status": status, "search_space": w,
            "max_search_space": max_search_space, "max_orbit_size": max_orbit_size}))
        return rec

    w = search_space_size(tables, sig)
    rec["search_space"] = f"{w:.3g}"
    if w > max_search_space:
        return skipped("SKIPPED_SEARCH_SPACE")
    t0 = time.time()
    vectors = enumerate_vectors(tables, sig)
    tables._closure_memo.clear()              # the cache is per signature; free it
    rec["n_vectors"] = len(vectors)
    if not vectors:
        return skipped("NO_VECTORS")
    classes = partition_orbits(tables, sig, vectors, max_per_class, max_orbit_size)
    if classes is None:
        return skipped("SKIPPED_ORBIT_SIZE")

    # write both files under .partial, then move them into place, sizes first:
    # a .sobj in out_dir is therefore always complete and always has its sidecar
    partial_dir = os.path.join(out_dir, ".partial")
    os.makedirs(partial_dir, exist_ok=True)
    tmp = os.path.join(partial_dir, os.path.basename(fname))
    tmp_sizes = tmp.replace(".sobj", ".sizes.json")
    save([list(sig), to_sage_vectors(tables, classes)], tmp)
    with open(tmp_sizes, "w") as f:
        json.dump({"sig": list(sig), "n_vectors": len(vectors),
                   "orbit_sizes": [s for _, s in classes]}, f)
    os.replace(tmp_sizes, fname.replace(".sobj", ".sizes.json"))
    os.replace(tmp, fname)
    rec.update(status="OK", n_classes=len(classes), seconds=f"{time.time() - t0:.1f}")
    return rec


def validate(tables, group_data_dir):
    """Recompute every signature of group_data_dir/{order}_{index}.sobj and check
    that the stored classes agree with the recomputed ones.

    For each signature: recompute all vectors and classes (no caps), map every
    stored vector to the recomputed class containing it, and check that each
    stored class lands in exactly one recomputed class (not split), that no two
    stored classes land in the same one (not merged), and that the class counts
    agree. Prints a line per signature and PASS/FAIL for the group.

    The stored file must use the same permutation representation as `tables`
    (see the caveat in the module docstring), or the stored elements cannot be
    matched to indices.
    """
    o, i = tables.gid
    old = load(os.path.join(group_data_dir, f"{o}_{i}.sobj"))
    pos = {tuple(img): idx for idx, img in enumerate(tables.el_imgs)}

    def to_idx(vec):
        """A stored vector (Sage elements) -> tuple of element indices."""
        out = []
        for g in vec:
            img = list(g.tuple())               # images of 1 .. (degree of the stored group)
            if len(img) < tables.deg:           # shorter list: the rest are fixed points
                img += list(range(len(img) + 1, tables.deg + 1))
            assert all(img[k] == k + 1 for k in range(tables.deg, len(img))), \
                "old representation moves points beyond the new degree"
            out.append(pos[tuple(img[:tables.deg])])
        return tuple(out)

    ok = True
    for sig, orbits in old:
        sigt = tuple(int(c) for c in sig)
        vectors = enumerate_vectors(tables, sigt)
        classes = partition_orbits(tables, sigt, vectors, max_per_class=float("inf"),
                                   max_orbit_size=float("inf"))
        assign = {}                            # recomputed vector -> its class number
        for ci, (members, _) in enumerate(classes):
            for m in members:
                assign[m] = ci
        old_assign = []                        # per stored class: set of recomputed classes hit
        for oi, orb in enumerate(orbits):
            mapped = {assign[to_idx(v)] for v in orb}
            old_assign.append(mapped)
            if len(mapped) != 1:
                ok = False
                print(f"  sig={sigt}: OLD class {oi} split across new classes {mapped}")
        flat = [next(iter(s)) for s in old_assign if len(s) == 1]
        if len(set(flat)) != len(flat):
            ok = False
            print(f"  sig={sigt}: old classes merged in new partition")
        if len(orbits) != len(classes):
            ok = False
            print(f"  sig={sigt}: class count old {len(orbits)} != new {len(classes)}")
        else:
            print(f"  sig={sigt}: {len(classes)} classes MATCH")
    print(f"{o}_{i}: validation {'PASS' if ok else 'FAIL'}")
    return ok


# =============================================================================
# 4. Command line
# =============================================================================

def assigned_items(plan_path, worker):
    """((order, index), signature) items assigned to `worker`, in plan order.

    The plan (written by data/submit.py) has a header line and then one line
    per item: worker <TAB> order_index <TAB> c1_c2_..._cr.
    """
    items = []
    with open(plan_path) as f:
        next(f)                               # header: worker, group, signature
        for line in f:
            if not line.strip():
                continue
            w, group, sig = line.split()
            if int(w) == worker:
                o, i = group.split("_")
                items.append(((int(o), int(i)), tuple(map(int, sig.split("_")))))
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gid", nargs=2, type=int, help="group as: order index")
    ap.add_argument("--sig", nargs="+", type=int, action="append",
                    help="signature c1 c2 ... cr (repeatable)")
    ap.add_argument("--plan", help="plan file written by data/submit.py")
    ap.add_argument("--worker", type=int, help="this worker's index in the plan")
    ap.add_argument("--report", help="write this worker's per-item report here (TSV)")
    ap.add_argument("--out-dir")
    ap.add_argument("--max-per-class", type=int,
                    help="store at most this many vectors per class")
    ap.add_argument("--max-search-space", type=float,
                    help="skip a signature whose search space (see search_space_size) "
                         "is larger than this")
    ap.add_argument("--max-orbit-size", type=int,
                    help="skip a signature with a class larger than this (memory guard)")
    ap.add_argument("--validate", metavar="GROUP_DATA_DIR")
    args = ap.parse_args()
    limits = (args.max_per_class, args.max_search_space, args.max_orbit_size)
    if not args.validate and None in limits:
        ap.error("--max-per-class, --max-search-space and --max-orbit-size are required")

    def build(gid):
        t0 = time.time()
        tables = GroupTables(*gid)
        print(f"group {gid}: |G|={tables.n} deg={tables.deg} "
              f"aut_gens={len(tables.aut_gens)} setup {time.time() - t0:.1f}s",
              flush=True)
        return tables

    if args.validate:
        sys.exit(0 if validate(build(tuple(args.gid)), args.validate) else 1)
    if args.plan:
        items = assigned_items(args.plan, args.worker)
        print(f"worker {args.worker}: {len(items)} items assigned by {args.plan}")
    else:
        items = [(tuple(args.gid), tuple(sig)) for sig in args.sig or []]
    os.makedirs(args.out_dir, exist_ok=True)

    report = open(args.report, "w") if args.report else None
    if report:
        report.write("\t".join(REPORT_COLUMNS) + "\n")
    outcomes = Counter()
    # the plan lists a group's items consecutively, so the tables are built once
    # per group and reused for all of its signatures
    cache_gid, tables = None, None
    for gid, sig in items:
        try:
            if gid != cache_gid:
                tables = build(gid)
                cache_gid = gid
            rec = run_item(tables, sig, args.out_dir, *limits)
        except Exception as e:                # noqa: BLE001 -- report and continue
            print(f"{gid} sig={sig}: ERROR {str(e)[:120]}", flush=True)
            rec = dict.fromkeys(REPORT_COLUMNS, "")
            rec.update(group=f"{gid[0]}_{gid[1]}", signature="_".join(map(str, sig)),
                       status=f"ERROR:{type(e).__name__}")
        print("  ".join(f"{k}={rec[k]}" for k in REPORT_COLUMNS if rec[k] != ""), flush=True)
        if report:
            report.write("\t".join(str(rec[k]) for k in REPORT_COLUMNS) + "\n")
            report.flush()
        outcomes[rec["status"]] += 1
    if report:
        report.close()
    print("done: " + ", ".join(f"{k} {v}" for k, v in sorted(outcomes.items())))


if __name__ == "__main__":
    main()
