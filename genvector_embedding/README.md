# genvector_embedding — the two embedding initialisations

The "matrix embedding" arm of the paper: each transposition token is
initialised to its **24 × 24 permutation matrix**, flattened to a
576-dimensional embedding vector.

## Why analytic (no training)

The natural training objective — make every generating-vector relation close,
`M(γ1)···M(γr) = I` — has an exact closed-form solution: the permutation
representation itself. Products of permutation matrices compose exactly as the
permutations do, so every relation of every group whose representation moves
at most `d` points holds **exactly, by construction**, including for groups
never seen. Relation accuracy is 1.0 with no training. (A gradient-trained
variant existed earlier in the project and was superseded by the closed form;
see the git history.)

Coverage caveats, by design:

- transpositions moving a point beyond `d = 24` are initialised to the
  identity matrix (exactness for the full universe would need `d ≈ 500`,
  i.e. 250,000-dimensional embeddings);
- signature-order tokens are not group elements and get distinct random
  orthogonal matrices, norm-matched to the permutation rows.

## Usage

```bash
python -m genvector_embedding.analytic --dim 24 --largest-degree 500 \
    --num-vocab 300000 --out-dir <root>/embeddings/analytic_d24_v2
```

Writes `embedding.npy` (`[num_vocab, d*d]` float32) plus `meta.json`. Point a
twin-model config at it:

```json5
embedding_dim: 576,
embedding_new: false,
embedding_path: '<out-dir>/embedding.npy',
embedding_requires_grad: true,   // fine-tuned during training
```

## The random embedding (`random_embedding.py`)

The comparison arm: Gaussian rows rescaled to norm sqrt(d), the norm of the
permutation-matrix rows above, so the two tables start at the same scale and
dimension and differ only in structure.

```bash
python -m genvector_embedding.random_embedding --dim 24 --num-vocab 300000 --seed 0 \
    --out-dir <root>/embeddings/random_normmatched_d576
```

The file the paper's runs used predates this script and its seed was not
recorded; the script states the recipe rather than reproducing that file.

## Token-id layout (why `--num-vocab 300000`)

Token ids are structural coordinates, not a learned vocabulary: transposition
`(i, j)` has id `(i-1) + 500*(j-1)` (500 = the tokeniser's maximum permutation
degree), giving a 250,000-slot grid; signature tokens sit above it at
`250000 + c`. 300,000 is that span rounded up — most rows are never selected
and never receive gradient, which costs memory only.
