# Handoff: redraw Figures 3 and 4 with directed IIA

**Audience:** the agent/person running this on a machine with GPU + model access.
**Written:** 2026-09-15.

## 1. Why

The paper uses the name "IIA" for three different quantities, and Figures 3 and 4 each
mix two of them inside a single figure. The decision is: **"IIA" means directed IIA
everywhere** in Figure 3, Figure 4 and Table 1 (the appendix γ-sensitivity table).

Definitions, which match this codebase's own naming (`step2_partition.py`,
`report_iia_buckets_make_counterfactual_all.py`, `graph.ipynb`):

| Name | Definition | Code field |
|---|---|---|
| **Directed IIA** | correct ordered pairs / `n(n-1)`; edge `i->j` = "when `i` is source and `j` is base, prediction correct" | `directed_iia`, `iia_whole_graph` |
| **Undirected density** | pair counts only if the intervention works in *both* directions; this is what the quasi-clique search optimizes | `undirected_density`, `compute_subgraph_iia` |
| **DAS/MDAS held-out IIA** | alignment score over sampled counterfactual pairs; *not* a graph statistic | `das_*_test_iia_by_attribute*.json` |

## 2. What is already correct — do not touch

- **Figure 3**, both classifier rows in both panels (`o1,o2,o3 Features` and `SAE Classifier`).
  They already match `directed_iia` in `experiments/logic_task/artifacts/partition_results_das/iia_buckets_*_classifier_directed.json`.
- **Figure 4**, the `Query-Group Index Results` and `SAE Classifier Results` panels,
  including their `Overall IIA = 0.46` titles (= directed IIA of the 199-node graph, 0.4635).
- **Table 1 panels (a) and (b)** — already directed.
- **Figure 5 / Table 1 panel (d)** (RAVEL) — intentionally left undirected, see §6.

## 3. What must change

Only the **Raw Partition** row of each figure. Bar *lengths* are counts and do not change;
only the IIA labels and one panel title move.

**Figure 3** (`figs and tables/logic2_new.png`, repo copy `ArXiv/figs/logic2_new.png`):

| Panel / bar | Now | Change to |
|---|---|---|
| o5 panel, Raw Partition, Target Bucket n=142 | IIA: 0.98 | **0.99** |
| o5 panel, Raw Partition, Other Bucket n=58 | IIA: 0.89 | **0.95** |
| o4 panel, Raw Partition, Target Bucket n=115 | IIA: 0.98 | **0.99** |
| o4 panel, Raw Partition, Other Bucket n=84 | IIA: 0.68 | **0.78** |

**Figure 4** (`figs and tables/entity_binding_new.png`, repo `ArXiv/figs/entity_binding_new.png` + `.svg`):

| Panel / bar | Now | Change to |
|---|---|---|
| Raw Partition panel title | Overall IIA = 0.44 | **Overall IIA = 0.48** |
| Raw Partition, Target Bucket n=129 | IIA=0.98 | **0.99** |
| Raw Partition, Other Bucket n=378 | IIA=0.11 | **0.31** |

Note: the existing `0.44` could not be reproduced from any saved artifact. The directed
overall IIA of the 507-node graph is `0.4798`; the group-cell mean is `0.4601`; the
undirected density is `0.2381`. Use **0.48**.

## 4. Verify before drawing

Run from the repo root — no GPU or model needed:

```bash
python experiments/recompute_directed_iia.py
```

It recomputes every number from the tracked artifacts and asserts them against the values
verified on 2026-09-15. If an assert fails, stop and investigate rather than redrawing.

## 5. How to redraw

**The original plotting code for Figures 3 and 4 is not in this repo.** An exhaustive
search found no file containing the panel labels ("Raw Partition", "Target Bucket").
The nearest style references that *are* here:

- `experiments/factual_recall/notebooks/plot.ipynb` cell 4 — Figure 5's horizontal
  stacked bars, green palette (`#CFE3CF`, `#D6E5D6`, `#465C47`), `$\mathbf{...}$` y-labels
  of the form `Target Bucket\nn=..., IIA: ...`. This is the closest template for Figure 3.
- `experiments/entity_binding/notebooks/graph.ipynb` cell 6 — query-group distribution
  per cluster, and the `iia_*` vs `undirected_density_*` field convention.

**Figure 4 is the easy one.** `ArXiv/figs/entity_binding_new.svg` is a wrapper: 3 `<text>`
elements ("Task", "Diagnosis", "Alignment") and 4 `<image>` elements holding embedded
raster panels. Regenerate only the Diagnosis panel (3 stacked heatmap rows) and swap that
one `<image>`, then re-export the PNG.

**Figure 3 has no SVG** — only the PNG. Either rebuild the whole right-hand bar panel and
recomposite, or edit the four labels in an image editor.

### The SAE blocker (read before regenerating a whole panel)

The SAE-classifier rows' **bucket memberships were never saved** — the jsons store only `n`
and the metrics. To redraw those rows from scratch you must re-run the SAE classifier to
recover per-input assignments:

- Figure 3: GPT-2-small + its SAE, logic task, `sae_l1_lr_full_features_L5_P78_interventionop4.pkl`
  (o4) and `..._L8_P81_interventionop5.pkl` (o5). Targets to reproduce: o4 → 87/112, o5 → 152/48.
- Figure 4: Gemma-2-2B + gemma-scope L15, `SAE_Gemma2_results/sae_classifier_15_10.pkl`.
  ⚠️ The saved run gives **63/136**, but the figure shows **51/148** — the figure came from a
  different run. Reproduce 51/148 or regenerate that row honestly with whatever the rerun gives.

Since those rows are already directed and correct, the cheapest correct outcome is to leave
them as-is and change only the Raw Partition row.

## 6. RAVEL is deliberately excluded

Table 1 panel (d) and Figure 5 stay **undirected**, and the table caption now says so.
Reason, verified three ways:

1. The paper's RAVEL graph is `experiments/factual_recall/artifacts/test_results/graph.pkl`
   — exactly symmetric, density 0.1043. Its `test_results_meta.json` has no `graph_type`
   field; it came from the older code path whose docstring reads "Undirected edge (i,j)
   only if both (i->j) and (j->i) are consistent" (`step2_das.py`, ~line 466).
2. The paper's alignment is **MDAS**: Figure 5's per-attribute row matches
   `das_mdas_test_iia_by_attribute_diff_gold.json` exactly (Language 0.1433 = the paper's 14.3%).
3. The only directed RAVEL graph is `test_results_das/graph.pkl` (`training_method: das`,
   `graph_type: directed`) — a **different alignment**: its `A & A.T` has density 0.2456 and
   disagrees with the paper's graph in 7,544 of 40,000 entries.

Current `step2_das.py` writes `test_results_{method}`; no `test_results_mdas/` exists.
Producing a directed MDAS graph therefore requires **re-running the interchange
interventions** on Llama-3.1-8B over 200×199 ordered pairs. If you do that, panel (d) can be
converted and the caption exception removed.

## 7. Data you need (and one file that was missing)

Tracked and sufficient for everything except the SAE rows:

- `experiments/logic_task/artifacts/partition_results_das/graph_das_L5_P78_200_op4_with_directed.pkl` (dict: `undirected`, `directed`)
- `experiments/logic_task/artifacts/partition_results_das/graph_das_L7_P77_200_op5_with_directed.pkl`
- `experiments/logic_task/artifacts/partition_results_das/graph_dataset_200_op4.pkl`, `graph_dataset_200_op5_ds.pkl`
- `experiments/logic_task/artifacts/linear_classifiers_das/natural_lr_*.pkl` (reproduce the natural-feature rows with no model)
- `experiments/entity_binding/artifacts/partition_results/graph_filling_liquids_15_10_512_google_gemma-2-2b-it.pkl` (undirected, 507)
- `experiments/entity_binding/artifacts/{classifier_filter_results,query_group_filter_results}/graph_whole_*_200_*.pkl` (directed, 199)

**⚠️ Newly added, must be committed:**
`experiments/entity_binding/artifacts/partition_results/graph_directed_filling_liquids_15_10_512_google_gemma-2-2b-it.pkl`

This is the **directed 507-node entity-binding graph**, without which Figure 4's Raw
Partition row and Table 1 panel (c) cannot be computed. It previously existed only at
`archived/entity_binding/classifier_filter_stale/graph_training_whole_filling_liquids_15_10_150_google_gemma-2-2b-it.pkl`,
which is inside the ignored `archived/` tree. It satisfies `A & A.T ==` the tracked
undirected graph exactly, so it is the directed version of the same run.

## 8. Where the paper lives

The ICLR submission is **not** in this repo. It is at
`D:\Stat\Causality\Causal Abstraction\Multi_Hypothesis_Testing_Project_All_In_One\ICLR2026`
on the author's machine, and its figures are in `figs and tables/`. The repo's
`ArXiv/figs/*.png` are higher-resolution copies of the same figures (same numbers), not
byte-identical. Regenerated images must be copied back into the paper's `figs and tables/`.

Table 1 (`ICLR2026/sections/appendix.tex`, `tab:gamma-sensitivity`) has already been
updated on the author's machine: panel (c) converted to directed IIA, the γ=0.98 row added
to all four panels, and the caption amended for the panel (d) exception.
