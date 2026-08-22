"""Compare gene-set enrichment computed on PREDICTED vs GROUND-TRUTH ST
expression (per-pathway Pearson/Spearman/MAE across spots), ported from
analysis/src/figure4_geneset.py::_process_single_sample in the main
STpredBench repo.

Expected `cfg` shape — same as `run.run_gene_enrichment`, plus
`cfg.DATA.data_dir` (for ground truth h5ad resolution).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from downstream.common import (
    downstream_output_dir,
    list_fold_samples,
    load_gene_panel,
    load_gt_adata,
    load_pred_adata,
    subset_to_genes,
    to_log_scale,
)

from .genesets import load_gene_sets
from .scoring import safe_corr, score_pathways


def _sample_rows(pred_adata, gt_adata, gene_sets, overlap_counts, min_overlap, method, sample_id) -> List[Dict[str, Any]]:
    common_genes = [g for g in pred_adata.var_names if g in set(gt_adata.var_names)]
    if not common_genes:
        return []

    pred_x = pred_adata[:, common_genes].X
    pred_x = pred_x.toarray() if hasattr(pred_x, "toarray") else pred_x
    gt_x = gt_adata[:, common_genes].X
    gt_x = gt_x.toarray() if hasattr(gt_x, "toarray") else gt_x

    def _threshold(raw_size: int) -> int:
        if isinstance(min_overlap, float) and 0 < min_overlap < 1:
            return max(5, int(raw_size * min_overlap))
        return min_overlap

    sample_gene_sets = {
        term: [g for g in gs if g in set(common_genes)] for term, gs in gene_sets.items()
    }
    sample_gene_sets = {
        term: gs for term, gs in sample_gene_sets.items() if len(gs) >= _threshold(len(gene_sets[term]))
    }
    if not sample_gene_sets:
        return []

    gt_scores = score_pathways(gt_x, common_genes, sample_gene_sets, method=method)
    pred_scores = score_pathways(pred_x, common_genes, sample_gene_sets, method=method)

    gt_vals, pred_vals = gt_scores.values, pred_scores.values
    mae_vals = np.nanmean(np.abs(gt_vals - pred_vals), axis=0)
    gt_stds = np.nanstd(gt_vals, axis=0)
    pred_stds = np.nanstd(pred_vals, axis=0)

    rows = []
    for j, term in enumerate(gt_scores.columns):
        rows.append({
            "pearson": safe_corr(gt_vals[:, j], pred_vals[:, j], method="pearson"),
            "spearman": safe_corr(gt_vals[:, j], pred_vals[:, j], method="spearman"),
            "mae": float(mae_vals[j]),
            "gt_std": float(gt_stds[j]),
            "pred_std": float(pred_stds[j]),
            "overlap_n": int(overlap_counts.get(term, len(sample_gene_sets[term]))),
            "pathway": term,
            "sample_id": sample_id,
        })
    return rows


def evaluate_gene_enrichment(cfg) -> Dict[str, Any]:
    params = cfg.DATA.downstream
    cpm = cfg.DATA.get("cpm", False)
    genes = load_gene_panel(cfg.MODEL.gene_path)
    min_overlap = params.get("min_overlap", 5)
    gene_sets, overlap_counts = load_gene_sets(
        params.get("library", "MSigDB_Hallmark_2020"),
        genes,
        min_overlap=min_overlap,
        cache_dir=params.get("cache_dir", "output/downstream_cache/genesets"),
    )
    method = params.get("score_method", "rank_mean")

    all_rows: List[Dict[str, Any]] = []
    for sample_id in list_fold_samples(cfg.DATA.pred_path_fold):
        try:
            pred_adata = load_pred_adata(cfg.DATA.pred_path_fold, sample_id)
            gt_adata = load_gt_adata(cfg.DATA.data_dir, sample_id)
        except FileNotFoundError:
            continue
        pred_adata = to_log_scale(subset_to_genes(pred_adata, genes), source="pred", cpm=cpm)
        gt_adata = to_log_scale(subset_to_genes(gt_adata, genes), source="gt", cpm=cpm)
        all_rows.extend(
            _sample_rows(pred_adata, gt_adata, gene_sets, overlap_counts, min_overlap, method, sample_id)
        )

    out_dir = os.path.join(downstream_output_dir(cfg, "gene_enrichment"), "eval")
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "metrics.csv")

    per_sample_df = pd.DataFrame(all_rows)
    per_sample_df.to_csv(metrics_path, index=False)

    per_pathway = (
        per_sample_df.groupby("pathway")[["pearson", "spearman", "mae"]].mean().reset_index()
        if not per_sample_df.empty
        else per_sample_df
    )

    return {
        "mode": "gene_enrichment",
        "metrics_path": metrics_path,
        "per_sample": all_rows,
        "per_pathway": per_pathway.to_dict("records") if not per_pathway.empty else [],
    }
