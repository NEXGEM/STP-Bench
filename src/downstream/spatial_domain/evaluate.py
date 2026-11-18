"""Compare SpaGCN spatial domains clustered from PREDICTED vs GROUND-TRUTH
ST expression (ARI/NMI/AMI + Hungarian-matched label accuracy), ported from
src/spatial_domain/run_spagcn_xenium.py::compare_domains in the main
STpredBench repo.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import pandas as pd
import scanpy as sc

from downstream.common import (
    downstream_output_dir,
    list_fold_samples,
    load_gene_panel,
    load_gt_adata,
    load_pred_adata,
    subset_to_genes,
    to_count_scale,
)

from .clustering import compare_domains, import_spagcn, run_spagcn_sample


_CLUSTER_KEYS = ("n_cluster_candidates", "silhouette_max_cells", "p", "l_start", "l_end", "l_tol", "l_max_run",
                 "res_start", "res_step", "tol", "lr", "search_epochs", "max_epochs", "refine", "refine_shape",
                 "seed")


def silk_gt_cache_dir(data_dir, genes, params) -> str:
    import hashlib
    import json as _json

    key = _json.dumps([os.path.abspath(str(data_dir)), list(genes), [[k, params.get(k)] for k in _CLUSTER_KEYS]],
                      default=str)
    return os.path.join("output", "downstream_cache", "spatial_domain_silk_gt", hashlib.md5(key.encode()).hexdigest()[:12])


def spatial_domain_samples(cfg, params):
    """Samples to analyse for this fold. `use_all_pred_samples: true` takes every sample that has a
    prediction file (for holdout sets where all samples are test samples for every fold's model),
    instead of only the ids.csv `fold_<k> == test` subset."""
    if params.get("use_all_pred_samples"):
        from glob import glob

        return sorted(os.path.splitext(os.path.basename(p))[0] for p in glob(os.path.join(cfg.DATA.pred_path_fold, "*.h5ad")))
    return list_fold_samples(cfg)


def evaluate_spatial_domain(cfg) -> Dict[str, Any]:
    params = cfg.DATA.downstream
    cpm = cfg.DATA.get("cpm", False)
    genes = load_gene_panel(params.get("gene_path") or cfg.MODEL.gene_path)
    spg = import_spagcn()

    out_dir = downstream_output_dir(cfg, "spatial_domain_silk" if params.get("n_cluster_candidates") else "spatial_domain")
    gt_cache_dir = os.path.join(out_dir, "eval", "gt")
    if params.get("n_cluster_candidates"):
        # GT domains (and the k chosen on them) do not depend on the model, so
        # share them across models/folds instead of re-running 4x SpaGCN each time.
        gt_cache_dir = silk_gt_cache_dir(cfg.DATA.data_dir, genes, params)
    os.makedirs(gt_cache_dir, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for sample_id in spatial_domain_samples(cfg, params):
        pred_path = os.path.join(out_dir, f"{sample_id}.h5ad")
        pred_clustered = sc.read_h5ad(pred_path) if os.path.isfile(pred_path) else None

        gt_path = os.path.join(gt_cache_dir, f"{sample_id}.h5ad")
        if os.path.isfile(gt_path):
            gt_clustered = sc.read_h5ad(gt_path)
        else:
            try:
                gt_adata = load_gt_adata(cfg.DATA.data_dir, sample_id)
            except FileNotFoundError:
                continue
            gt_adata = to_count_scale(subset_to_genes(gt_adata, genes), source="gt", cpm=cpm)
            gt_clustered = run_spagcn_sample(spg, gt_adata, params)
            gt_clustered.write_h5ad(gt_path)

        k_ref = int(gt_clustered.uns["spagcn"]["n_clusters"]) if params.get("n_cluster_candidates") else None
        if pred_clustered is not None and k_ref is not None and int(pred_clustered.uns["spagcn"]["n_clusters"]) != k_ref:
            pred_clustered = None  # stale cache clustered with a different k
        if pred_clustered is None:
            # same k as the reference (GT) domains of this sample
            pred_adata = to_count_scale(
                subset_to_genes(load_pred_adata(cfg.DATA.pred_path_fold, sample_id), genes),
                source="pred", cpm=cpm,
            )
            pred_clustered = run_spagcn_sample(spg, pred_adata, params, n_clusters=k_ref)
            pred_clustered.write_h5ad(pred_path)

        sample_rows = compare_domains(pred_clustered, gt_clustered, sample_id)
        for r in sample_rows:
            r["k_ref"] = int(gt_clustered.uns["spagcn"]["n_clusters"])
            r["k_pred"] = int(pred_clustered.uns["spagcn"]["n_clusters"])
        rows.extend(sample_rows)

    metrics_path = os.path.join(out_dir, "eval", "metrics.csv")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(metrics_path, index=False)

    return {"mode": "spatial_domain", "metrics_path": metrics_path, "per_sample": rows}
