# test_results_mdas — a NEGATIVE result, retained as a contrast case

**Do not use these numbers.** This directory holds a directed-graph rerun that used the
**wrong alignment**. It is kept only so the das_best-vs-das_mdas distinction is concrete
and checkable rather than a claim in a commit message.

The RAVEL directed IIA numbers the paper uses come from
[`../test_results_dasbest/`](../test_results_dasbest/).

## What this run was

| | |
|---|---|
| checkpoint | `das_mdas_best.pt` (layer 14, k 2048, d_model 4096) |
| featurizer | `das_mdas_best_featurizer/` |
| `featurizer_featurizer` sha256 | `19f134c138405de5fc9c0fba1cdc9616a163f7888f712d56fca2cd91cda5d07e` |
| `featurizer_inverse_featurizer` sha256 | `c5c58510f76ff9625101a6ccac8a5ed6f549557a70b278fec20ff3a08c4fc96f` |
| environment | causalab `cd7e8b3`, torch 2.14.0+cu130, transformers 5.17.0, numpy 2.5.3 |
| nodes / interventions | 200 / 39,800, `--batch_size 32` |

Two things were wrong with it at once — the alignment **and** the environment — so it is
not a controlled comparison against `test_results_dasbest/`. It cannot be used to
attribute the discrepancy to either cause on its own.

## Why it was rejected

The paper's graph is `../test_results/graph.pkl`, and because that graph was built as
`D & D.T`, it forces directed IIA over any node set into
`[density, (density + 1) / 2]`. Two of the three buckets fall outside:

| bucket | n | directed IIA | required interval | |
|---|---|---|---|---|
| English | 38 | 0.9168 | [0.9815, 0.9908] | **below the floor** |
| Spanish | 23 | 0.9960 | [0.9881, 0.9941] | **above the ceiling** |
| residual | 139 | 0.1399 | [0.0684, 0.5342] | inside |
| overall | 200 | 0.1707 | [0.1043, 0.5522] | inside |

Being below the floor means ~91 ordered interventions failed that the published graph says
must have succeeded. `(D & D.T)` also differs from `../test_results/graph.pkl` in 1,244 of
40,000 entries (622 of 19,900 unordered pairs), where the das_best run differs in **zero**.

## The methodological point, which is the reason this is kept

This run's *overall* directed IIA is 0.1707, against the 0.17 recorded in
`../das_mdas_results.json` for layer 14 / k 2048 — agreement to 0.03 standard errors of
that 300-pair estimate. That looked like strong evidence the alignment was right. It was
not. The disagreement was concentrated where it mattered most:

| bucket | within-bucket pairs | disagreeing | rate |
|---|---|---|---|
| English | 703 | 108 | **15.4%** |
| Spanish | 253 | 5 | 2.0% |
| residual | 9,591 | 218 | 2.3% |
| overall | 19,900 | 622 | 3.1% |

Five times the global rate, inside the 38-node bucket whose published headline value is
0.98. **A matching global average is not evidence about a subset.** `das_mdas` is the same
architecture at the same layer as `das_best`, just a different training run, which is
exactly the kind of thing that reproduces an aggregate while relocating individual
borderline decisions.

## Reproducing the comparison

```bash
python experiments/factual_recall/scripts/ravel_directed_iia.py --dir test_results_mdas
```

`pair_scores.npz` holds all 39,800 raw per-ordered-pair outcomes, so the whole contrast is
recomputable offline with no model and no GPU.
