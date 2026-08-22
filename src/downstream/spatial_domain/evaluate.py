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


def evaluate_spatial_domain(cfg) -> Dict[str, Any]:
    params = cfg.DATA.downstream
    cpm = cfg.DATA.get("cpm", False)
    genes = load_gene_panel(cfg.MODEL.gene_path)
    spg = import_spagcn()

    out_dir = downstream_output_dir(cfg, "spatial_domain")
    gt_cache_dir = os.path.join(out_dir, "eval", "gt")
    os.makedirs(gt_cache_dir, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for sample_id in list_fold_samples(cfg):
        pred_path = os.path.join(out_dir, f"{sample_id}.h5ad")
        if os.path.isfile(pred_path):
            pred_clustered = sc.read_h5ad(pred_path)
        else:
            pred_adata = to_count_scale(
                subset_to_genes(load_pred_adata(cfg.DATA.pred_path_fold, sample_id), genes),
                source="pred", cpm=cpm,
            )
            pred_clustered = run_spagcn_sample(spg, pred_adata, params)

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

        rows.extend(compare_domains(pred_clustered, gt_clustered, sample_id))

    metrics_path = os.path.join(out_dir, "eval", "metrics.csv")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(metrics_path, index=False)

    return {"mode": "spatial_domain", "metrics_path": metrics_path, "per_sample": rows}
