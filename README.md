# Deep Learning of Topological Equivalence Classes

Code and data for the paper *Deep Learning of Topological Equivalence
Classes*. A twin network adapted from the RE2 text
matching architecture decides whether two generating vectors of a finite
group action on a Riemann surface lie in the same topological equivalence
class, from the tokenised generating vectors alone.

## Layout

```
data/                 the data pipeline: generating vectors and classes -> tokens -> pair sets
  pipeline.json         its one configuration file (storage, HPC, every dataset parameter and seed)
  submit.py             renders and submits the SLURM job of each stage
  data_controller.py    what exists / what is missing, per stage
  data_generation/      orbits_fast.py: all generating vectors of a (group, signature) and their classes
  data_tokenisation/    tokenise_variable.py: group elements -> transposition tokens
  data_pairs/           pairs_variable.py, pairs_uniform_classes.py, order_split.py, prefix_pairs.py
twin_model/           the model
  training.json         its configuration (directories, logging, GPU job, the paper's run groups)
  submit.py             renders and submits a training job per run config
  configs/              default.json5 (every setting, the paper's values) + one file per run
  src/                  the network, training loop, batching, config parsing
  train.py, evaluate.py, predict.py
  release/              builds results/census_predictions.csv (LMFDB properties + pair categories)
genvector_embedding/  the two embedding initialisations: analytic (permutation matrices) and random
results/              census_predictions.csv, the released model output
```

## The pipeline

```bash
# 1. generating vectors and their equivalence classes (SageMath / GAP)
python data/submit.py orbits                      # plans every pending (group, signature) item,
                                                  # runs the array, then combines per group
# 2. tokenisation
python data/submit.py tokenise --universe union        # signatures of length r <= 16
python data/submit.py tokenise --universe union_r98    # r <= 98
# 3. pair sets (see stages.pairs.presets for the exact parameters and seeds)
python data/submit.py pairs --preset train-union  # 2M train / 60k eval, then the |G| <= 64 split
python data/submit.py pairs --preset big-eval     # the |G| > 64 evaluation set
python data/submit.py pairs --preset r4           # r = 4 arm
python data/submit.py pairs --preset r98          # r <= 98 arm
python data/submit.py pairs --preset var300       # r <= 16 arm (300k/10k prefix of train-union)
python data/submit.py pairs --preset le64-subsets # the training-set-size series
python data/submit.py pairs --preset census       # test set, |G| > 64, class-uniform
python data/submit.py pairs --preset census-full  # test set, all orders, size-matched
# what exists so far
python data/data_controller.py status
# 4. embeddings
python -m genvector_embedding.analytic --out-dir <root>/embeddings/analytic_d24_v2
python -m genvector_embedding.random_embedding --out-dir <root>/embeddings/random_normmatched_d576
# 5. training; each job scores its best checkpoint on the group's test set(s)
python twin_model/submit.py --group table1        # embedding x signature length (six runs)
python twin_model/submit.py --group table2        # training-set size (eight runs)
# 6. the release
python twin_model/release/build_census_release.py \
    --conf <root>/predictions/vo_pfx_conv_1000k__census.csv --out results/census_predictions.csv
```

`REPRODUCING.md` goes through these steps in detail. Each stage's script
documents what it computes and how; `data/data_generation/orbits_fast.py` and
`twin_model/src/network.py` are the places to start reading.

## What the stages compute

1. **Generating vectors and classes.** For every group and signature of
   quotient genus 0 acting on a surface of genus ≤ 48 (the inventory of
   Paulhus), all generating vectors are enumerated and partitioned into orbits
   under Aut(G) and the braid moves, following Behn–Rodríguez–Rojas. The group
   is built once in GAP and turned into integer tables; everything else is
   arithmetic on small integers. Up to 200 vectors per class are stored.
2. **Tokenisation.** Each element of a generating vector is written as a word
   in transpositions from its cycle decomposition, each transposition (i, j)
   becoming the token (i−1) + 500(j−1); each element is one block of 128
   tokens (its order token, the word, padding). A generating vector of r
   elements is 128·r tokens.
3. **Pair sets.** Training pairs are sampled at random (positives uniform over
   classes, negatives uniform over signatures); the test sets are
   class-uniform censuses (4 positive and 4 negative pairs per class). The
   order split keeps training pairs with both groups of order ≤ 64 and
   evaluation pairs with both above.
4. **The model.** Each generator of each vector goes through the RE2 block
   stack (CNN encoder, alignment against the other vector's generator,
   fusion), is max-pooled over tokens, then the generators are pooled with a
   1-d convolution and a max; a symmetric prediction head compares the two
   sides.

## Requirements

| part | needs |
|---|---|
| `data/` | SageMath (with GAP and the SmallGroups library); the Python it bundles |
| `twin_model/`, `genvector_embedding/` | Python ≥ 3.10, PyTorch ≥ 2.0, NumPy, json5, tqdm, scikit-learn (`twin_model/requirements.txt`); optionally wandb and tensorboardX |
| `twin_model/release/` | building the LMFDB group-property cache needs network access (`lmfdb/prefetch.py`) or `psycopg` (`lmfdb/mirror_fetch.py`), and GAP for the groups the LMFDB does not hold (`lmfdb/gap_records.g`) |

## Released census predictions (`results/census_predictions.csv`)

The predictions of the model trained on 10⁶ pairs from groups of order ≤ 64,
over the class-uniform census of 86,760 pairs from groups of order > 64 (none
of which it saw in training): one row per pair with the two class ids and
groups, the signature, the pair category (cross-group, or same-group cyclic /
other abelian / non-abelian p-group / other non-abelian), the ground truth,
the model's P(same class), whether it was right, and LMFDB properties of both
groups (abelian, cyclic, nilpotent, supersolvable, solvable, metacyclic,
metabelian, exponent, derived length, |Aut(G)|, number of conjugacy classes).
Properties come from the LMFDB small-groups data (`twin_model/release/lmfdb/`);
the nine groups of order 512 and 1536, which the LMFDB does not hold, carry
the same invariants computed in GAP, and every cached value was checked against
GAP before the release was built.

## Paper

Link to the preprint will be added on availability.

## License

GPL-3.0, see `LICENSE`. `twin_model/src` is adapted from the RE2 PyTorch
implementation (Alibaba, Apache-2.0).
