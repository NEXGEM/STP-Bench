"""Spot x gene expression -> spot x pathway activity scoring, ported from
analysis/src/figure4_geneset.py::_score_pathways in the main STpredBench
repo."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def score_pathways(
    expr: np.ndarray,
    genes: Sequence[str],
    gene_sets: Dict[str, List[str]],
    method: str = "rank_mean",
) -> pd.DataFrame:
    """Convert spot x gene expression to spot x pathway activity.

    - ssgsea: gseapy.ssgsea per-spot single-sample GSEA NES.
    - rank_mean (default): fast ssGSEA-like score — within each spot, genes
      are ranked and centered (`rank - (n_genes+1)/2`); each pathway's score
      is the mean centered rank of its genes (allows negative correlations).
    - singscore: PySingscore algorithm (`rank(method='min') / N - 0.5`).
    - zscore_mean: mean of per-gene z-scored expression for pathway genes.
    """
    genes = list(genes)
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    terms = [term for term, gs in gene_sets.items() if any(g in gene_to_idx for g in gs)]
    scores = np.full((expr.shape[0], len(terms)), np.nan, dtype=float)

    if method == "ssgsea":
        import gseapy as gp

        data = pd.DataFrame(expr.T, index=genes, columns=[f"spot_{i}" for i in range(expr.shape[0])])
        ssgsea_res = gp.ssgsea(
            data=data,
            gene_sets={term: gene_sets[term] for term in terms},
            outdir=None,
            sample_norm_method="rank",
            min_size=1,
            max_size=max(len(genes), 1),
            permutation_num=0,
            threads=4,
            no_plot=True,
            seed=42,
            verbose=False,
        )
        if ssgsea_res.res2d is None or ssgsea_res.res2d.empty:
            return pd.DataFrame(scores, columns=terms)
        mat = ssgsea_res.res2d.pivot(index="Name", columns="Term", values="NES")
        mat = mat.reindex(index=data.columns, columns=terms)
        scores = mat.to_numpy(dtype=float)
    elif method == "zscore_mean":
        means = np.nanmean(expr, axis=0, keepdims=True)
        stds = np.nanstd(expr, axis=0, keepdims=True)
        zexpr = (expr - means) / np.where(stds == 0, np.nan, stds)
        for j, term in enumerate(terms):
            idx = [gene_to_idx[g] for g in gene_sets[term] if g in gene_to_idx]
            scores[:, j] = np.nanmean(zexpr[:, idx], axis=1)
    elif method in {"rank_mean", "ssgsea_like"}:
        centered_ranks = np.empty_like(expr, dtype=float)
        for i in range(expr.shape[0]):
            ranks = rankdata(expr[i], method="average")
            centered_ranks[i] = ranks - (expr.shape[1] + 1) / 2
        for j, term in enumerate(terms):
            idx = [gene_to_idx[g] for g in gene_sets[term] if g in gene_to_idx]
            scores[:, j] = np.nanmean(centered_ranks[:, idx], axis=1)
    elif method == "singscore":
        N = expr.shape[1]
        ranked = pd.DataFrame(expr, columns=genes).rank(axis=1, method="min", ascending=True).values
        for j, term in enumerate(terms):
            idx = [gene_to_idx[g] for g in gene_sets[term] if g in gene_to_idx]
            scores[:, j] = np.nanmean(ranked[:, idx], axis=1) / N - 0.5
    else:
        raise ValueError(f"Unsupported score_method: {method}")

    return pd.DataFrame(scores, columns=terms)


# Re-exported for backwards compatibility — moved to downstream.common since
# deconvolution/evaluate.py needs it too and it has no gene-enrichment-
# specific logic.
from downstream.common import safe_corr  # noqa: E402,F401
