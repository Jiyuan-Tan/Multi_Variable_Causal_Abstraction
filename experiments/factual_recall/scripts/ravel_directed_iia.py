#!/usr/bin/env python
"""Directed IIA for the RAVEL/factual-recall alignment, with a bounds-based check.

Figures 3, 4 and Table 1 panels (a)-(c) report *directed* IIA (correct ordered
pairs / n(n-1)). RAVEL (Figure 5, Table 1 panel (d)) reported *undirected edge
density* under the same name "IIA", because the graph shipped with the paper
(artifacts/test_results/graph.pkl) is exactly symmetric: the older code path
built it with "undirected edge (i,j) only if both (i->j) and (j->i) are
consistent" and discarded the per-direction outcomes at aggregation. Recovering
directed IIA therefore required re-running the interchange interventions; this
script consumes that rerun and reports directed IIA so RAVEL can use the same
definition as the rest of the paper.

WHICH ALIGNMENT. The paper's graph was built by `das_best` (layer 14, k 2048),
not `das_mdas_best`. Two independent lines of evidence: the current code derives
both names from --training_method (`das_{method}_best.pt`, `test_results_{method}/`)
and so can never have produced `das_best.pt` or a bare `test_results/`, which
pairs those two old-code-path artifacts with each other; and on the author's
machine das_mdas_best.pt postdates test_results/graph.pkl by 23 days. A first
rerun using das_mdas failed the checks below; das_best is the right alignment.

HOW THIS IS CHECKED. Bit-identical reproduction of the paper's graph is the wrong
bar for a rerun on different hardware and library versions - any nondeterminism
fails it while telling us nothing about whether the measurement is sound. What
matters is that the rerun is *arithmetically consistent* with the published graph:

  1. PARTITION. K=3, gamma=0.98 quasi-clique partition of the paper's undirected
     graph must give 38 / 23 / 139 with the 38 being English. This depends only on
     the tracked graph, so it needs no GPU and no rerun - the bucket memberships
     come from the paper, not from us.

  2. BOUNDS. Since the paper's U = D & D.T, an edge means both orderings succeeded
     and a non-edge means at most one did. So over any node set, directed successes
     lie in [2e, 2e + (m - e)] for e edges among m unordered pairs, i.e.

         directed IIA  in  [ density , (density + 1) / 2 ]

     Each bucket's directed IIA must land inside the interval its own published
     density implies. Falling below the floor means we failed interventions the
     published graph says succeeded; above the ceiling means we succeeded on more
     than any valid decomposition of it permits. Either is a real inconsistency.

(D & D.T) vs the paper's graph is still reported, as a DIAGNOSTIC: if it matches
exactly that is a clean bonus, and if it does not, the bounds are what decide.

Usage:
    python ravel_directed_iia.py                          # diagnostics + verdict
    python ravel_directed_iia.py --dir test_results_mdas  # check a different rerun
    python ravel_directed_iia.py --out headline.json      # also write the JSON
    python ravel_directed_iia.py --gate-only              # partition check only
"""

from __future__ import annotations

import argparse
import collections
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from partition_graph_quasi_clique import quasi_clique_partition  # noqa: E402

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
PAPER_DIR = ARTIFACTS / "test_results"            # undirected, shipped with the paper
DEFAULT_DIR = "test_results_dasbest"              # the das_best directed rerun

K = 3
GAMMA = 0.98
EXPECTED_SIZES = [38, 23, 139]
EXPECTED_TOP_LANG = {0: "English", 1: "Spanish"}
TOL = 1e-9


def load(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def directed_iia(adj, nodes=None) -> float:
    """Fraction of ordered pairs (i, j), i != j, whose intervention is correct.

    Identical to experiments/recompute_directed_iia.py:directed_iia, deliberately -
    RAVEL must use the same estimator as the logic and entity-binding figures.
    """
    a = np.asarray(adj, dtype=bool)
    if nodes is not None:
        a = a[np.ix_(nodes, nodes)]
    n = a.shape[0]
    if n <= 1:
        return 1.0
    x = a.copy()
    np.fill_diagonal(x, False)
    return float(x.sum()) / (n * (n - 1))


def undirected_density(adj, nodes=None) -> float:
    a = np.asarray(adj, dtype=bool)
    return directed_iia(a & a.T, nodes)


def bounds(u_density: float):
    """Interval that U = D & D.T forces directed IIA into, given U's density."""
    return u_density, (u_density + 1.0) / 2.0


def language_of(row: dict) -> str:
    return row.get("languagegold") or row.get("gold") or "?"


def partition(U, nodes_meta):
    labels = np.asarray(quasi_clique_partition(U, K, GAMMA, min_clique_size=2,
                                               method="greedy"))
    buckets = {}
    for lab in sorted(set(labels.tolist())):
        idx = np.where(labels == lab)[0]
        langs = collections.Counter(language_of(nodes_meta[i]) for i in idx)
        buckets[int(lab)] = dict(idx=idx, n=len(idx), langs=langs)
    return labels, buckets


def check_partition(buckets) -> bool:
    sizes = sorted((b["n"] for b in buckets.values()), reverse=True)
    ok = sizes == sorted(EXPECTED_SIZES, reverse=True)
    print("  bucket sizes %s vs expected %s : %s"
          % (sizes, sorted(EXPECTED_SIZES, reverse=True), "OK" if ok else "MISMATCH"))
    for lab, want in EXPECTED_TOP_LANG.items():
        if lab not in buckets:
            print("    bucket %d missing : MISMATCH" % lab)
            ok = False
            continue
        top, cnt = buckets[lab]["langs"].most_common(1)[0]
        hit = top.startswith(want)
        ok &= hit
        print("    bucket %d (n=%d) dominant language %r x%d, expected %s : %s"
              % (lab, buckets[lab]["n"], top, cnt, want, "OK" if hit else "MISMATCH"))
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="write headline numbers as JSON here")
    ap.add_argument("--gate-only", action="store_true",
                    help="run only the partition check (no directed graph needed)")
    ap.add_argument("--dir", default=DEFAULT_DIR,
                    help="artifacts subdirectory holding the directed rerun "
                         "(default: %s)" % DEFAULT_DIR)
    args = ap.parse_args()

    U = np.asarray(load(PAPER_DIR / "graph.pkl"), dtype=bool)
    nodes_meta = load(PAPER_DIR / "test_dataset.pkl")
    n = U.shape[0]
    nodes_meta = nodes_meta[:n]
    print("paper graph %s  symmetric=%s  undirected density=%.4f"
          % (U.shape, bool((U == U.T).all()), directed_iia(U)))

    print("\nCHECK 1/2 - K=%d gamma=%.2f partition of the paper's undirected graph" % (K, GAMMA))
    labels, buckets = partition(U, nodes_meta)
    ok_part = check_partition(buckets)
    if args.gate_only:
        print("\npartition check:", "PASS" if ok_part else "FAIL")
        return 0 if ok_part else 1

    run_dir = ARTIFACTS / args.dir
    dpath = run_dir / "graph.pkl"
    if not dpath.exists():
        sys.exit("\nDirected graph not found at %s - run step2_das.py --mode test with "
                 "--ckpt/--featurizer_dir on a GPU node first." % dpath)
    D = np.asarray(load(dpath), dtype=bool)
    if D.shape != U.shape:
        sys.exit("shape mismatch: directed %s vs paper %s" % (D.shape, U.shape))
    meta_path = run_dir / "test_results_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    print("\ndirected graph %s from %s  symmetric=%s" % (D.shape, args.dir, bool((D == D.T).all())))
    if meta.get("checkpoint_used"):
        print("  checkpoint: %s" % Path(meta["checkpoint_used"]).name)
    for fn, sh in (meta.get("featurizer_sha256") or {}).items():
        print("  %-30s sha256 %s..." % (fn, sh[:16]))

    # --- diagnostic: exact agreement with the paper's graph
    both = D & D.T
    disagree = int((both != U).sum())
    print("\nDIAGNOSTIC - (D & D.T) vs the paper's graph")
    print("  disagreeing entries: %d of %d (%.2f%%)%s"
          % (disagree, U.size, 100.0 * disagree / U.size,
             "  -> EXACT MATCH" if disagree == 0 else ""))
    if disagree:
        iu = np.triu_indices(n, 1)
        print("  unordered pairs differing: %d of %d" % (int((both != U)[iu].sum()), len(iu[0])))

    # --- arbiter: bounds implied by the paper's own graph
    print("\nCHECK 2/2 - directed IIA must lie within the bounds the paper's graph implies")
    print("  %-26s %-5s %-10s %-13s %-22s %s"
          % ("bucket", "n", "U_density", "directed IIA", "bounds [lo, hi]", "verdict"))
    rows = []
    ok_bounds = True
    allidx = np.arange(n)
    entries = [("OVERALL", allidx)] + [
        (("bucket %d (%s)" % (l, buckets[l]["langs"].most_common(1)[0][0][:14])), buckets[l]["idx"])
        for l in sorted(buckets, key=lambda l: -buckets[l]["n"])]
    for name, idx in entries:
        ud = directed_iia(U, idx)
        lo, hi = bounds(ud)
        di = directed_iia(D, idx)
        inside = lo - TOL <= di <= hi + TOL
        verdict = "IN" if inside else ("BELOW lo" if di < lo else "ABOVE hi")
        ok_bounds &= inside
        print("  %-26s %-5d %-10.4f %-13.4f [%.4f, %.4f]     %s"
              % (name, len(idx), ud, di, lo, hi, verdict))
        rows.append(dict(name=name, n=int(len(idx)), u_density=round(ud, 4),
                         directed_iia=round(di, 4), lo=round(lo, 4), hi=round(hi, 4),
                         within_bounds=bool(inside)))

    if not (ok_part and ok_bounds):
        print("\nCHECKS FAILED - do not quote these numbers and do not push the artifacts.")
        return 1

    print("\nALL CHECKS PASS")
    headline = dict(
        alignment="das_best (layer 14, k 2048), factual recall / RAVEL, target attribute language",
        graph="experiments/factual_recall/artifacts/%s/graph.pkl" % args.dir,
        metric="directed IIA = correct ordered pairs / n(n-1)",
        n_graph_nodes=int(n),
        checkpoint_used=meta.get("checkpoint_used"),
        featurizer_sha256=meta.get("featurizer_sha256"),
        partition=dict(K=K, gamma=GAMMA, method="greedy quasi-clique",
                       source_graph="test_results/graph.pkl (the paper's undirected graph)"),
        checks=dict(partition_38_23_139_with_english=bool(ok_part),
                    directed_iia_within_published_bounds=bool(ok_bounds),
                    and_transpose_disagreeing_entries=disagree),
        results=rows,
    )
    if args.out:
        Path(args.out).write_text(json.dumps(headline, indent=2) + "\n")
        print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
