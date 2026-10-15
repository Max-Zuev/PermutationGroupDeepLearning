# twin_model

The model: a twin network adapted from the RE2 text matching architecture
(Yang et al. 2019, https://github.com/alibaba-edu/simple-effective-text-matching-pytorch),
applied generator by generator to a pair of tokenised generating vectors, and
pooled over generators. `src/network.py` describes the forward pass and how it
relates to the paper's Figure 1; `src/model.py` the optimiser, learning-rate
schedule and the sample/update unit conversion; `src/trainer.py` the loop and
the stopping rules; `src/interface.py` and `src/utils/loader.py` the batching.

```
training.json       directories (under the data pipeline's storage root), wandb, GPU job, run groups
submit.py           python twin_model/submit.py --group table1 | --config NAME [--dry-run]
job_train.sh.tmpl   the job: train, then predict.py on the group's test set(s)
configs/default.json5   every run setting with the paper's values, documented
configs/<run>.json5     one per run of the paper: data, embedding, output directory
train.py            python train.py configs/<run>.json5
evaluate.py         python evaluate.py <checkpoint>.pt <pairs>.csv   (metrics)
predict.py          python predict.py --checkpoint <ckpt> --data-dir D --eval-file F --out P.csv
release/            build_census_release.py; lmfdb/ holds the group-property cache it joins
                    (the fetchers that build cache.json, and the GAP script that recomputes/checks it)
```

Paths in a run config are relative: `data_dir` to the data pipeline's
storage root, `embedding_path` and `output_dir` to the `embeddings` and
`outputs` directories named in `training.json`. A run writes
`<outputs>/<output_dir>/<name>/` with `best.pt`, `best.embeddings.npy`,
`args.json5` and `log.txt`.

## License

`src/` is adapted from RE2's PyTorch implementation, Apache License 2.0; the
files keep its header. The rest of the repository is GPL-3.0.
