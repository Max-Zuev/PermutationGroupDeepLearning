# Reproducing the data and results

Everything runs through two configuration files and two submit scripts:
`data/pipeline.json` + `data/submit.py` for the data, `twin_model/training.json`
+ `twin_model/submit.py` for the model. The submit scripts generate SLURM job
scripts and execute them via `sbatch`.

## 0. Set up

1. **Environments.** SageMath (with GAP and the SmallGroups library) for
   `data/`; a Python environment with `twin_model/requirements.txt` for the
   model. The paper used one conda environment for both (SageMath 10.7,
   PyTorch 2.x), named in `hpc.conda_env` of both configuration files.
2. **The inventory.** The list of (group, signature) pairs acting on curves
   of genus ≤ 48 is Jen Paulhus's `groupsignaturedata.txt`
   (www.jenpaulhus.com/research/monodromy.html). Put it at
   `<storage.root>/<storage.inventory>`.
3. **`data/pipeline.json`.** Set `storage.root` (everything the pipeline
   writes goes under it) and the `hpc` section (repository path, log
   directories, module and conda environment names). Leave `stages.*` alone.
4. **`twin_model/training.json`.** Set `hpc` likewise and `wandb.mode`
   (`online`, `offline`, or `disabled`; `online`/`offline` need the wandb
   package and, for `online`, a Weights&Biases account).

## 1. Generating vectors and classes

```bash
python data/submit.py orbits
```

Lists every (group, signature) of the inventory within the caps
(`max_order` 500, `max_r` 16) that has no result yet, assigns each group with
all of its signatures to one array worker, runs the array, and chains the
combine step that merges a group's signatures into one file once all of them
are resolved. Rerunning picks up where the last run stopped; a signature that
exceeds `max_search_space` or `max_orbit_size`, or has no generating vectors,
is marked and not retried (delete its `.skip.json` to retry after raising a
cap). Check progress with `python data/data_controller.py status` and
`... missing --stage orbits`.

Note that the permutation representation GAP picks for a group is not deterministic
across sessions, so a group's signatures are always computed by one worker.

## 2. Tokenisation

```bash
python data/submit.py tokenise --universe union        # r <= 16
python data/submit.py tokenise --universe union_r98    # r <= 98
```

One CSV per group under `storage.tokenised.<universe>`. Each worker's report
beside the rendered script lists what it wrote and dropped per group.

## 3. Pair sets

Note the order.

```bash
python data/submit.py pairs --preset train-union   # train.csv, eval.csv, then the |G| <= 64 split
python data/submit.py pairs --preset big-eval      # eval_gt64_masked.csv for the scaling runs
python data/submit.py pairs --preset r4
python data/submit.py pairs --preset r98
python data/submit.py pairs --preset var300        # needs train-union
python data/submit.py pairs --preset le64-subsets  # needs train-union
python data/submit.py pairs --preset census
python data/submit.py pairs --preset census-full
python data/data_controller.py datasets            # every registered file present?
```

## 4. Embeddings

```bash
python -m genvector_embedding.analytic --dim 24 --num-vocab 300000 \
    --out-dir <root>/embeddings/analytic_d24_v2
python -m genvector_embedding.random_embedding --dim 24 --num-vocab 300000 --seed 0 \
    --out-dir <root>/embeddings/random_normmatched_d576
```

The run configs expect `analytic_d24_v2/embedding.npy` and
`random_normmatched_d576.npy` under `<root>/embeddings` (rename or symlink the
random one). The paper's random embedding file predates
`random_embedding.py`; the script states the recipe but cannot reproduce that
file's seed.

## 5. Training

```bash
python twin_model/submit.py --group table1     # cm_u_{r4,var,r98}_{random,matrix}
python twin_model/submit.py --group table2     # vo_pfix_conv_100k ... vo_pfx_conv_2000k
python twin_model/submit.py --config cm_u_r98_random   # one run
```

Each job trains one config (`twin_model/configs/<name>.json5`, every setting
documented in `configs/default.json5`) and then runs `predict.py` with the
best checkpoint on the group's test set(s), writing
`<root>/predictions/<name>__<dataset>.csv` (row, id1, id2, target,
P(same class)). Accuracy over a predictions file is the paper's number for
that run. To score any checkpoint on any pair file by hand:

```bash
cd twin_model
python evaluate.py <root>/outputs/<output_dir>/<name>/best.pt <pair file>.csv
```

Each training run takes up to the 24-hour wall limit on one GPU; the
schedule and early stopping are described in `src/model.py` and
`src/trainer.py`.

## 6. The release

```bash
python twin_model/release/build_census_release.py \
    --conf <root>/predictions/vo_pfx_conv_1000k__census.csv \
    --out results/census_predictions.csv
```

Joins the predictions with the LMFDB group properties in
`twin_model/release/lmfdb/cache.json` and labels each pair's category. Build
the cache first (it is not committed): `lmfdb/prefetch.py` (web API) or
`lmfdb/mirror_fetch.py` (PostgreSQL mirror) fetch the groups in
`lmfdb/groups.txt`; then `lmfdb/gap_records.g` computes the same fields in GAP
for the groups the LMFDB does not hold (orders 512 and 1536) and
`lmfdb/merge_gap_records.py` adds them. `merge_gap_records.py --check`
compares a GAP computation with the fetched values.

```bash
python twin_model/release/lmfdb/prefetch.py --groups-file twin_model/release/lmfdb/groups.txt
cd twin_model/release/lmfdb && gap -q -b gap_records.g && python merge_gap_records.py gap_records.json
```

## What is not deterministic

- GAP's choice of permutation representation (step 1), which changes the
  stored class representatives but not the classes.
- The random embedding's seed for the paper's own runs (step 4).
- GPU training (step 5): configs run unseeded (`seed: null`); set `seed` and
  `deterministic: true` for a repeatable run.
