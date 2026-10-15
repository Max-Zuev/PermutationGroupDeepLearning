# coding=utf-8
"""Pipeline launcher: render a stage's job script from data/pipeline.json and
submit it. Holds no parameters of its own.

    python data/submit.py orbits    [--workers N] [--dry-run]
    python data/submit.py tokenise  --universe union|union_r98 [--workers N]
    python data/submit.py pairs     --preset census|census-full|r4|r98|var300|le64-subsets|...

Each command renders ``stages.<name>.template`` by substituting @@TOKENS@@ with
values from the config, writes the rendered script under ``hpc.rendered_dir``
(a permanent record of exactly what ran), and submits it by filename.

``orbits`` first lists the pending (group, signature) items with
make_workitems.py, assigns each group with all its items to one worker, writes
that plan beside the rendered script, and chains the combine stage behind the
array.

``tokenise`` assigns work centrally: it takes the orbit files with no tokenised
CSV yet, deals each group to exactly one worker, and writes that assignment as
a plan file beside the rendered script (workers write their reports there too).
Workers only execute their own rows.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
from data_controller import DataController, gid_str  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _expand(p: str) -> str:
    return os.path.expandvars(p)


def _under_root(*parts: str) -> str:
    return "/".join([config.DATA_ROOT.rstrip("/"), *parts])


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def render(stage_name: str, subs: dict, out_name: str, stamp: str | None = None) -> str:
    """Substitute @@TOKENS@@ into a stage template; return the written path."""
    st = config.stage(stage_name)
    with open(os.path.join(REPO, st["template"]), encoding="utf-8") as f:
        text = f.read()
    base = {
        "MODULES": config.HPC["modules"],
        "CONDA_ENV": config.HPC["conda_env"],
        # expanded here, not left to the shell: SLURM does NOT expand
        # variables inside #SBATCH directives
        "REPO_DIR": _expand(config.HPC["repo_dir"]),
        "LOG_DIR": _expand(config.HPC["log_dir"]),
        "INTERPRETER": st.get("interpreter", "python -u"),
        "TIME": st["resources"]["time"],
        "MEM": st["resources"]["mem"],
        "CPUS": str(st["resources"]["cpus"]),
    }
    base.update(subs)
    for key, val in base.items():
        text = text.replace(f"@@{key}@@", str(val))
    left = [line for line in text.splitlines() if "@@" in line]
    if left:
        raise SystemExit(f"unsubstituted tokens in {stage_name}: {left}")

    dest_dir = _expand(config.HPC["rendered_dir"])
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, f"{out_name}_{stamp or _stamp()}.sh")
    with open(path, "w", newline="\n", encoding="utf-8") as f:
        f.write(text)
    return path


def sbatch(path: str, depend: str | None = None, dry: bool = False) -> str:
    cmd = ["sbatch", "--parsable"]
    if depend:
        cmd.append(f"--dependency=afterok:{depend}")
    cmd.append(path)
    if dry:
        print(f"[dry-run] {' '.join(cmd)}\n--- {path} ---")
        print(open(path, encoding="utf-8").read())
        return "DRYRUN"
    return subprocess.run(cmd, capture_output=True, text=True,
                          check=True).stdout.strip()


# ---------------------------------------------------------------- stages
def write_orbit_plan(items: list[tuple[str, str]], workers: int, path: str) -> list[int]:
    """Assign every group, with all of its items, to exactly one worker.

    items are (group, signature) pairs in make_workitems order. Keeping a group's
    items on one worker means its permutation representation is built once, so
    all of its signatures share one representation. Groups go, most items first,
    to the least-loaded worker. Returns the number of items per worker.
    """
    by_group: dict[str, list[str]] = {}
    for group, sig in items:
        by_group.setdefault(group, []).append(sig)
    load = [0] * workers
    rows = []
    for group in sorted(by_group, key=lambda g: (-len(by_group[g]), g)):
        w = load.index(min(load))
        load[w] += len(by_group[group])
        rows += [(w, group, sig) for sig in by_group[group]]
    with open(path, "w", newline="\n", encoding="utf-8") as f:
        f.write("worker\tgroup\tsignature\n")
        for w, group, sig in sorted(rows, key=lambda r: r[0]):
            f.write(f"{w}\t{group}\t{sig}\n")
    return load


def cmd_orbits(args):
    st = config.stage("orbits")
    p = dict(st["params"])
    stamp = _stamp()
    dest_dir = _expand(config.HPC["rendered_dir"])
    os.makedirs(dest_dir, exist_ok=True)
    run = os.path.join(dest_dir, f"orbits_{stamp}")

    # which items exist; make_workitems.py and combine_orbits.py must agree on this
    selection = ["--inventory", _under_root(config.INVENTORY),
                 "--max-order", str(p["max_order"]), "--max-r", str(p["max_r"])]
    # pending (group, signature) items; read-only on the data, so also run on --dry-run
    listed = subprocess.run([
        sys.executable, os.path.join(REPO, "data/data_generation/make_workitems.py"),
        *selection,
        "--orbits-dir", _under_root(config.ORBITS_DIR),
        "--staging-dir", _under_root(config.ORBITS_STAGING),
        "--out", f"{run}.items.txt"])
    if listed.returncode:                     # make_workitems.py has said what is wrong
        raise SystemExit(listed.returncode)
    with open(f"{run}.items.txt", encoding="utf-8") as f:
        items = [(f"{o}_{i}", "_".join(sig))
                 for o, i, *sig in (line.split() for line in f if line.strip())]
    if not items:
        print("orbits: nothing to compute")
        return
    n_groups = len({g for g, _ in items})
    workers = min(args.workers or st["resources"]["array"], n_groups)
    plan = f"{run}.plan.tsv"
    load = write_orbit_plan(items, workers, plan)

    arr = render("orbits", {
        "JOB_NAME": "orbits", "ARRAY_MAX": workers - 1,
        "PLAN": plan, "REPORT_PREFIX": f"{run}.worker",
        "STAGING_DIR": _under_root(config.ORBITS_STAGING),
        "MAX_SEARCH_SPACE": p["max_search_space"],
        "MAX_ORBIT_SIZE": p["max_orbit_size"],
        "MAX_PER_CLASS": p["max_vectors_per_class"],
    }, "orbits", stamp)
    jid = sbatch(arr, dry=args.dry_run)
    cmb = render("combine", {
        "JOB_NAME": "combine", "STAGING_DIR": _under_root(config.ORBITS_STAGING),
        "ORBITS_DIR": _under_root(config.ORBITS_DIR),
        "INVENTORY": _under_root(config.INVENTORY),
        "MAX_ORDER": p["max_order"], "MAX_R": p["max_r"],
    }, "combine", stamp)
    cid = sbatch(cmb, depend=jid if not args.dry_run else None, dry=args.dry_run)
    print(f"orbits: {len(items)} items in {n_groups} groups -> {workers} workers "
          f"({min(load)}-{max(load)} items each), plan {plan} -> array {jid} "
          f"-> combine {cid}")


def write_plan(groups: list[tuple[str, int]], workers: int, path: str) -> list[int]:
    """Assign every group to exactly one worker and write the plan file.

    Largest orbit files first, dealt round-robin, so each worker gets a similar
    mix of big and small groups. Returns the number of groups per worker.
    """
    ordered = sorted(groups, key=lambda g: (-g[1], g[0]))
    counts = [0] * workers
    with open(path, "w", newline="\n", encoding="utf-8") as f:
        f.write("worker\tgroup\n")
        for i, (group, _) in enumerate(ordered):
            f.write(f"{i % workers}\t{group}\n")
            counts[i % workers] += 1
    return counts


def cmd_tokenise(args):
    st = config.stage("tokenise")
    if args.universe not in st["universes"]:
        raise SystemExit(f"--universe must be one of {list(st['universes'])}")
    dc = DataController()
    groups = [(gid_str(g), dc.orbit_path(g).stat().st_size)
              for g in sorted(dc.tokenised_missing(args.universe))]
    if not groups:
        print(f"tokenise[{args.universe}]: nothing to do")
        return
    workers = min(args.workers or st["resources"]["array"], len(groups))
    stamp = _stamp()
    dest_dir = _expand(config.HPC["rendered_dir"])
    os.makedirs(dest_dir, exist_ok=True)
    run = os.path.join(dest_dir, f"tokenise_{args.universe}_{stamp}")
    plan = f"{run}.plan.tsv"
    counts = write_plan(groups, workers, plan)

    path = render("tokenise", {
        "JOB_NAME": f"tok_{args.universe}", "ARRAY_MAX": workers - 1,
        "PLAN": plan,
        "REPORT_PREFIX": f"{run}.worker",
        "ORBITS_DIR": _under_root(config.ORBITS_DIR),
        "TOK_OUT": _under_root(config.TOKENISED[args.universe]),
        "MAX_R": st["universes"][args.universe]["max_r"],
        "LARGEST_DEGREE": st["params"]["largest_degree"],
        "PART_LENGTH": st["params"]["part_length"],
        "MAX_PER_CLASS": st["params"]["max_vectors_per_class"],
    }, f"tokenise_{args.universe}", stamp)
    jid = sbatch(path, dry=args.dry_run)
    print(f"tokenise[{args.universe}]: {len(groups)} groups -> {workers} workers "
          f"({min(counts)}-{max(counts)} groups each), plan {plan} -> {jid}")


def cmd_pairs(args):
    st = config.stage("pairs")
    presets = st["presets"]
    if args.preset not in presets:
        raise SystemExit(f"--preset must be one of {list(presets)}")
    pr = presets[args.preset]

    if pr["builder"] == "prefix_pairs":
        # prefixes of existing pair files: no tokenised input, no sampler args
        argv = []
        for source, out, rows in pr["copies"]:
            argv += ["--copy", _under_root(source), _under_root(out), str(rows)]
    else:
        argv = ["--tok-dir", _under_root(config.TOKENISED[pr["tokenised"]])]
        argv += ["--out-dir", _under_root(pr["out_dir"])] if "out_dir" in pr \
            else ["--out", _under_root(pr["out"])]
        for k, v in pr["args"].items():
            argv += [f"--{k}", str(v)]

    post = ""
    if "order_split" in pr:
        d = _under_root(pr["out_dir"])
        post = (f'conda run --no-capture-output -n {config.HPC["conda_env"]} '
                f'{st["interpreter"]} data/data_pairs/order_split.py '
                f'--train-in {d}/train.csv --eval-in {d}/eval.csv '
                f'--threshold {pr["order_split"]["threshold"]} --out-dir {d}')

    path = render("pairs", {
        "JOB_NAME": f"pairs_{args.preset}", "PRESET": args.preset,
        "BUILDER": pr["builder"], "BUILD_ARGS": " ".join(argv),
        "POST_STEP": post,
    }, f"pairs_{args.preset}")
    print(f"pairs[{args.preset}] -> {sbatch(path, dry=args.dry_run)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="render and print, submit nothing")
    sub = ap.add_subparsers(dest="stage", required=True)

    o = sub.add_parser("orbits"); o.add_argument("--workers", type=int)
    o.set_defaults(func=cmd_orbits)
    t = sub.add_parser("tokenise"); t.add_argument("--universe", required=True)
    t.add_argument("--workers", type=int); t.set_defaults(func=cmd_tokenise)
    p = sub.add_parser("pairs"); p.add_argument("--preset", required=True)
    p.set_defaults(func=cmd_pairs)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
