"""SpaGCN spatial-domain clustering, ported from
src/spatial_domain/run_spagcn_xenium.py in the main STpredBench repo.

Count-scale handling (expm1_round_clip for predictions, round/clip only for
GT) is delegated to `downstream.common.to_count_scale` by the caller
(run.py/evaluate.py) — this module always receives an adata already in
raw-count-like scale and never re-derives that itself, so there is a single
place (common.py) that encodes the pred-vs-GT scale asymmetry.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score


def import_spagcn():
    try:
        import SpaGCN as spg
    except Exception as exc:  # pragma: no cover - environment check
        raise ImportError(
            "Could not import SpaGCN. Install it first (see requirements/downstream.txt):\n"
            "  pip install SpaGCN==1.2.7\n"
            f"Original error: {exc}"
        ) from exc
    return spg


def _xy_coords(adata) -> Tuple[np.ndarray, np.ndarray]:
    if "spatial" not in adata.obsm:
        raise KeyError("adata.obsm['spatial'] is required for SpaGCN clustering.")
    coords = np.asarray(adata.obsm["spatial"])
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("adata.obsm['spatial'] must have at least two columns.")
    return coords[:, 0].astype(float), coords[:, 1].astype(float)


def preprocess_for_spagcn(spg, adata):
    adata = adata.copy()
    spg.prefilter_genes(adata, min_cells=3)
    spg.prefilter_specialgenes(adata)
    import scanpy as sc

    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    if sparse.issparse(adata.X):
        adata.X = adata.X.toarray()
    return adata


def match_labels_to_reference(ref_labels: pd.Series, pred_labels: pd.Series) -> Tuple[Dict[str, str], float]:
    """Hungarian-match predicted cluster IDs onto reference cluster IDs by
    maximizing overlap. Returns (mapping, match_accuracy)."""
    ref_cats = sorted(ref_labels.astype(str).unique())
    pred_cats = sorted(pred_labels.astype(str).unique())
    if not ref_cats or not pred_cats:
        return {}, float("nan")

    contingency = pd.crosstab(pred_labels.astype(str), ref_labels.astype(str))
    contingency = contingency.reindex(index=pred_cats, columns=ref_cats, fill_value=0)
    row_ind, col_ind = linear_sum_assignment(-contingency.to_numpy())
    mapping = {pred_cats[i]: ref_cats[j] for i, j in zip(row_ind, col_ind)}
    matched = sum(contingency.iloc[i, j] for i, j in zip(row_ind, col_ind))
    accuracy = matched / len(ref_labels) if len(ref_labels) else float("nan")
    return mapping, float(accuracy)


def run_spagcn_sample(spg, adata, params: Dict[str, Any]):
    """Cluster one sample's expression into spatial domains.

    `params` fields (see config/downstream/defaults.yaml[spatial_domain]):
    n_clusters, p, l_start/l_end/l_tol/l_max_run, res_start/res_step,
    tol, lr, search_epochs, max_epochs, refine, refine_shape, seed.
    """
    seed = params.get("seed", 100)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass

    adata_train = preprocess_for_spagcn(spg, adata)
    if adata_train.n_obs < adata.n_obs:
        adata = adata[adata_train.obs_names].copy()

    x, y = _xy_coords(adata_train)
    adj = spg.calculate_adj_matrix(x=x, y=y, histology=False)

    l_value = params.get("l_value")
    if l_value is None:
        l_value = spg.search_l(
            params.get("p", 0.5), adj,
            start=params.get("l_start", 0.01), end=params.get("l_end", 1000),
            tol=params.get("l_tol", 0.01), max_run=params.get("l_max_run", 100),
        )

    n_clusters = params.get("n_clusters", 7)
    resolution = params.get("resolution")
    if resolution is None:
        resolution = spg.search_res(
            adata_train, adj, l_value, n_clusters,
            start=params.get("res_start", 0.2), step=params.get("res_step", 0.1),
            tol=params.get("tol", 5e-3), lr=params.get("lr", 0.05),
            max_epochs=params.get("search_epochs", 20),
            r_seed=seed, t_seed=seed, n_seed=seed,
        )

    clf = spg.SpaGCN()
    clf.set_l(l_value)
    clf.train(
        adata_train, adj, init_spa=True, init="louvain", res=resolution,
        tol=params.get("tol", 5e-3), lr=params.get("lr", 0.05),
        max_epochs=params.get("max_epochs", 200),
    )
    y_pred, prob = clf.predict()

    adata.obs["spagcn_pred"] = pd.Categorical(y_pred.astype(str))
    adata.obsm["spagcn_prob"] = prob
    adata.uns["spagcn"] = {
        "l": float(l_value),
        "p": float(params.get("p", 0.5)),
        "resolution": float(resolution),
        "n_clusters": int(n_clusters),
        "genes": int(adata.n_vars),
        "cells": int(adata.n_obs),
    }

    if params.get("refine", True):
        refined_pred = spg.refine(
            sample_id=adata.obs_names.tolist(),
            pred=adata.obs["spagcn_pred"].tolist(),
            dis=adj,
            shape=params.get("refine_shape", "square"),
        )
        adata.obs["spagcn_refined_pred"] = pd.Categorical([str(label) for label in refined_pred])

    return adata


def compare_domains(pred_adata, gt_adata, sample_id: str) -> list[Dict[str, Any]]:
    """One row per label key (spagcn_pred, spagcn_refined_pred if present)
    comparing predicted vs GT-derived domain labels: ARI/NMI/AMI (label-
    permutation invariant) plus a Hungarian-matched label accuracy."""
    common = pred_adata.obs_names.intersection(gt_adata.obs_names)
    rows = []
    if len(common) == 0:
        return rows

    for key in ("spagcn_pred", "spagcn_refined_pred"):
        if key not in pred_adata.obs or key not in gt_adata.obs:
            continue
        y_pred = pred_adata.obs.loc[common, key].astype(str)
        y_gt = gt_adata.obs.loc[common, key].astype(str)
        _, match_accuracy = match_labels_to_reference(y_gt, y_pred)
        rows.append({
            "sample_id": sample_id,
            "label_key": key,
            "n_common": int(len(common)),
            "ari": adjusted_rand_score(y_gt, y_pred),
            "nmi": normalized_mutual_info_score(y_gt, y_pred),
            "ami": adjusted_mutual_info_score(y_gt, y_pred),
            "hungarian_match_accuracy": match_accuracy,
        })
    return rows
