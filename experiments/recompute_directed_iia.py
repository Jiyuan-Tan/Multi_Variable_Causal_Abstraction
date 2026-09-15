"""Recompute every directed-IIA number needed to redraw Figures 3 and 4.

Run from the repo root:
    python experiments/recompute_directed_iia.py

No GPU or model required. Values are asserted against the numbers verified on 2026-09-15;
if an assertion fails, investigate before redrawing anything.

See FIGURE_REDRAW_HANDOFF.md for what changes in each figure and why.
"""

import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE / "entity_binding" / "scripts"))
from partition_graph_quasi_clique import quasi_clique_partition  # noqa: E402

LOGIC = HERE / "logic_task" / "artifacts" / "partition_results_das"
EB = HERE / "entity_binding" / "artifacts"
GAMMA = 0.98
TOL = 5e-4
ROUND_TOL = 5e-3  # for values quoted from the paper table at 2 decimals


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def directed_iia(adj, nodes=None):
    """Fraction of ordered pairs (i, j), i != j, whose interchange intervention is correct."""
    a = np.asarray(adj, dtype=bool)
    if nodes is not None:
        a = a[np.ix_(nodes, nodes)]
    n = a.shape[0]
    if n <= 1:
        return 1.0
    x = a.copy()
    np.fill_diagonal(x, False)
    return float(x.sum()) / (n * (n - 1))


def undirected_density(adj, nodes=None):
    """Edge density where an edge requires the intervention to succeed in both directions."""
    a = np.asarray(adj, dtype=bool)
    return directed_iia(a & a.T, nodes)


def check(label, got, want, tol=TOL):
    """Compare a recomputed value against the expected one.

    IIA/density values are checked at 4-decimal precision. Pr[primitive] values are
    quoted from the paper table at 2 decimals, so they use tol=ROUND_TOL instead.
    """
    ok = abs(got - want) <= tol
    print("  %-46s %.4f   (expected %.4f) %s" % (label, got, want, "OK" if ok else "MISMATCH"))
    assert ok, "%s: got %.6f, expected %.6f (tol %.4f)" % (label, got, want, tol)


def logic_pass(name, graph_file, dataset_file, primitive, expected):
    print("\n[%s]" % name)
    obj = load(LOGIC / graph_file)
    U = np.array(obj["undirected"], dtype=bool)
    D = np.array(obj["directed"], dtype=bool)
    U = U & U.T
    ds = load(LOGIC / dataset_file)
    n = U.shape[0]
    assert not (D == D.T).all(), "directed matrix should not be symmetric"
    assert ((D & D.T) == U).all(), "undirected graph must equal directed AND its transpose"

    check("overall directed IIA", directed_iia(D), expected["overall_directed"])
    check("overall undirected density", undirected_density(U), expected["overall_undirected"])

    labels = quasi_clique_partition(U, 2, GAMMA, min_clique_size=2, method="greedy")
    for cluster, exp in zip((0, 1), (expected["target"], expected["other"])):
        nodes = np.where(labels == cluster)[0].tolist()
        vals = [ds[i]["base_labels"].get(primitive) for i in nodes]
        frac = sum(1 for v in vals if v is True) / len(vals)
        tag = "target" if cluster == 0 else "other "
        assert len(nodes) == exp["n"], "%s size %d != %d" % (tag, len(nodes), exp["n"])
        print("  %s bucket: n=%d" % (tag, len(nodes)))
        check("  %s directed IIA  <- FIGURE USES THIS" % tag, directed_iia(D, nodes), exp["directed"])
        check("  %s undirected density (old value)" % tag, undirected_density(U, nodes), exp["undirected"])
        check("  %s Pr[%s]" % (tag, primitive), frac, exp["prim"], tol=ROUND_TOL)
    return n


def entity_binding():
    print("\n[entity binding, layer 15]")
    U = np.array(load(EB / "partition_results" / "graph_filling_liquids_15_10_512_google_gemma-2-2b-it.pkl"), dtype=bool)
    directed_path = EB / "partition_results" / "graph_directed_filling_liquids_15_10_512_google_gemma-2-2b-it.pkl"
    if not directed_path.exists():
        print("  MISSING: %s" % directed_path)
        print("  See FIGURE_REDRAW_HANDOFF.md section 7 - this file must be committed.")
        return
    D = np.array(load(directed_path), dtype=bool)
    assert ((D & D.T) == U).all(), "directed graph is not the directed version of the tracked undirected graph"
    ds = load(EB / "partition_results" / "filtered_input_samples_filling_liquids_15_10_512_google_gemma-2-2b-it.pkl")
    groups = np.array([d["query_group"] for d in ds[: U.shape[0]]])

    check("overall directed IIA  <- PANEL TITLE (0.44 -> 0.48)", directed_iia(D), 0.4798)
    check("overall undirected density (old value)", undirected_density(U), 0.2381)

    labels = quasi_clique_partition(U, 2, GAMMA, min_clique_size=2, method="greedy")
    expected = {0: dict(n=129, directed=0.9907, undirected=0.9815, start_tail=1.000),
                1: dict(n=378, directed=0.3088, undirected=0.1099, start_tail=0.222)}
    for cluster, exp in expected.items():
        nodes = np.where(labels == cluster)[0].tolist()
        tag = "target" if cluster == 0 else "other "
        frac = sum(1 for i in nodes if groups[i] in (0, 1, 2, 9)) / len(nodes)
        assert len(nodes) == exp["n"], "%s size %d != %d" % (tag, len(nodes), exp["n"])
        print("  %s bucket: n=%d" % (tag, len(nodes)))
        check("  %s directed IIA  <- FIGURE USES THIS" % tag, directed_iia(D, nodes), exp["directed"])
        check("  %s undirected density (old value)" % tag, undirected_density(U, nodes), exp["undirected"])
        check("  %s Pr[start/tail group]" % tag, frac, exp["start_tail"], tol=ROUND_TOL)

    # 199-node held-out graphs behind the two classifier panels (already directed in the figure)
    g199 = np.array(load(EB / "classifier_filter_results" / "graph_whole_filling_liquids_15_10_200_google_gemma-2-2b-it.pkl"), dtype=bool)
    ds199 = load(EB / "classifier_filter_results" / "filtered_dataset_filling_liquids_15_10_200_google_gemma-2-2b-it.pkl")
    grp199 = np.array([d["query_group"] for d in ds199[: g199.shape[0]]])
    check("199-node overall directed IIA (panel titles 0.46)", directed_iia(g199), 0.4635)
    keep = [i for i in range(g199.shape[0]) if grp199[i] in (0, 1, 2, 9)]
    rest = [i for i in range(g199.shape[0]) if grp199[i] not in (0, 1, 2, 9)]
    assert (len(keep), len(rest)) == (79, 120), "query-group buckets should be 79/120"
    check("  query-group target directed IIA (figure 0.90)", directed_iia(g199, keep), 0.9004)
    check("  query-group other  directed IIA (figure 0.19)", directed_iia(g199, rest), 0.1900)


def main():
    logic_pass(
        "logic o4 @ layer 5, position 78",
        "graph_das_L5_P78_200_op4_with_directed.pkl",
        "graph_dataset_200_op4.pkl",
        "op1",
        dict(overall_directed=0.8560, overall_undirected=0.7557,
             target=dict(n=115, directed=0.9900, undirected=0.9800, prim=0.10),
             other=dict(n=84, directed=0.7771, undirected=0.6836, prim=1.00)),
    )
    logic_pass(
        "logic o5 @ layer 7, position 77",
        "graph_das_L7_P77_200_op5_with_directed.pkl",
        "graph_dataset_200_op5_ds.pkl",
        "op4",
        dict(overall_directed=0.6803, overall_undirected=0.6158,
             target=dict(n=142, directed=0.9903, undirected=0.9810, prim=0.05),
             other=dict(n=58, directed=0.9453, undirected=0.8905, prim=0.86)),
    )
    entity_binding()
    print("\nAll checks passed. Figure targets are in FIGURE_REDRAW_HANDOFF.md section 3.")


if __name__ == "__main__":
    main()
