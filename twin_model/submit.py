# coding=utf-8
"""Submit training jobs: render job_train.sh.tmpl from training.json and sbatch it.

    python twin_model/submit.py --config cm_u_r98_random [--dry-run]
    python twin_model/submit.py --group table1|table2  [--dry-run]

Each job trains one run config and then writes the best checkpoint's per-pair
probabilities on the run group's test set(s) with predict.py. Settings come
from training.json (resources, directories, wandb) and, for the datasets, from
the data pipeline's configuration it points to. The rendered script is kept
under hpc.rendered_dir as a record of what ran. Needs the json5 package (the
run configs are JSON5).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime

import json5

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

with open(os.path.join(HERE, "training.json"), encoding="utf-8") as _f:
    TRAINING = json.load(_f)
with open(os.path.normpath(os.path.join(HERE, TRAINING["data_pipeline"])), encoding="utf-8") as _f:
    PIPELINE = json.load(_f)
ROOT = PIPELINE["storage"]["root"]


def _expand(p: str) -> str:
    return os.path.expandvars(p)


def _under_root(*parts: str) -> str:
    return "/".join([ROOT.rstrip("/"), *parts])


def run_config(name: str) -> dict:
    with open(os.path.join(HERE, "configs", f"{name}.json5"), encoding="utf-8") as f:
        (cfg,) = json5.load(f)
    return cfg


def render(subs: dict, out_name: str) -> str:
    """Substitute @@TOKENS@@ into the job template; return the written path."""
    hpc = TRAINING["hpc"]
    with open(os.path.join(HERE, hpc["template"]), encoding="utf-8") as f:
        text = f.read()
    base = {
        "MODULES": hpc["modules"], "CONDA_ENV": hpc["conda_env"],
        # expanded here: SLURM does not expand variables inside #SBATCH directives
        "REPO_DIR": _expand(hpc["repo_dir"]), "LOG_DIR": _expand(hpc["log_dir"]),
        "INTERPRETER": hpc["interpreter"],
        "TIME": hpc["resources"]["time"], "MEM": hpc["resources"]["mem"],
        "CPUS": str(hpc["resources"]["cpus"]),
        "PARTITION": hpc["resources"]["partition"], "GRES": hpc["resources"]["gres"],
    }
    base.update(subs)
    for key, val in base.items():
        text = text.replace(f"@@{key}@@", str(val))
    left = [line for line in text.splitlines() if "@@" in line]
    if left:
        raise SystemExit(f"unsubstituted tokens: {left}")
    dest_dir = _expand(hpc["rendered_dir"])
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, f"{out_name}_{datetime.now():%Y%m%d_%H%M%S}.sh")
    with open(path, "w", newline="\n", encoding="utf-8") as f:
        f.write(text)
    return path


def sbatch(path: str, dry: bool) -> str:
    cmd = ["sbatch", "--parsable", path]
    if dry:
        print(f"[dry-run] {' '.join(cmd)}\n--- {path} ---")
        print(open(path, encoding="utf-8").read())
        return "DRYRUN"
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()


def submit(name: str, group: str, dry: bool) -> None:
    cfg = run_config(name)
    store = TRAINING["storage"]
    checkpoint = _under_root(store["outputs"], cfg["output_dir"], cfg["name"], "best.pt")
    predictions_dir = _under_root(store["predictions"])
    steps = []
    for ds in TRAINING["groups"][group]["score_on"]:
        d, f = os.path.split(_under_root(PIPELINE["storage"]["datasets"][ds]))
        steps.append(f'conda run --no-capture-output -n {TRAINING["hpc"]["conda_env"]} '
                     f'{TRAINING["hpc"]["interpreter"]} predict.py --checkpoint $CHECKPOINT '
                     f'--data-dir {d} --eval-file {f} --out {predictions_dir}/{name}__{ds}.csv')
    path = render({
        "JOB_NAME": name, "CONFIG": name, "CHECKPOINT": checkpoint,
        "PREDICTIONS_DIR": predictions_dir, "SCORE_STEPS": "\n".join(steps),
    }, f"train_{name}")
    scored = ", ".join(TRAINING["groups"][group]["score_on"])
    print(f"train[{name}] ({group}, scored on {scored}) -> {sbatch(path, dry)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="render and print, submit nothing")
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--config", help="one run config (configs/<name>.json5)")
    which.add_argument("--group", help="every config of a group in training.json")
    args = ap.parse_args()

    groups = {g: spec for g, spec in TRAINING["groups"].items() if not g.startswith("_")}
    if args.group:
        if args.group not in groups:
            raise SystemExit(f"--group must be one of {list(groups)}")
        todo = [(c, args.group) for c in groups[args.group]["configs"]]
    else:
        owners = [g for g, spec in groups.items() if args.config in spec["configs"]]
        if not owners:
            raise SystemExit(f"{args.config} is in no group of training.json: {list(groups)}")
        todo = [(args.config, owners[0])]
    for name, group in todo:
        submit(name, group, args.dry_run)


if __name__ == "__main__":
    main()
