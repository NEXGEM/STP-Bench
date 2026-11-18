"""Compare cell2location deconvolution computed on PREDICTED vs GROUND-TRUTH
ST expression (per-cell-type Pearson correlation across spots), ported from
analysis/src/figure_deconvolution.py in the main STpredBench repo.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import pandas as pd

from downstream.common import (
    downstream_output_dir,
    list_fold_samples,
    load_gt_adata,
    load_pred_adata,
    safe_corr,
    to_count_scale,
)

from .run import _load_or_train_signatures
from .spatial import run_cell2location_sample


_C2L_KEYS = ("n_cells_per_location", "detection_alpha", "spatial_epochs", "spatial_batch_size", "spatial_posterior_samples",
             "reference_path", "labels_key", "batch_key", "max_cells_per_label", "reference_epochs")


def deconv_gt_cache_dir(data_dir, genes, params) -> str:
    """GT deconvolution does not depend on the model/fold: share one cache per (data, gene panel, cell2location params)."""
    import hashlib
    import json as _json

    key = _json.dumps([os.path.abspath(str(data_dir)), list(genes), [[k, params.get(k)] for k in _C2L_KEYS]], default=str)
    return os.path.join("output", "downstream_cache", "deconvolution_gt", hashlib.md5(key.encode()).hexdigest()[:12])


def deconv_samples(cfg, params):
    """`use_all_pred_samples: true` = every sample with a prediction file (holdout sets: all samples are test for every fold's model)."""
    if params.get("use_all_pred_samples"):
        from glob import glob

        return sorted(os.path.splitext(os.path.basename(p))[0] for p in glob(os.path.join(cfg.DATA.pred_path_fold, "*.h5ad")))
    return list_fold_samples(cfg)


def evaluate_deconvolution(cfg) -> Dict[str, Any]:
    cpm = cfg.DATA.get("cpm", False)
    signatures, accelerator = _load_or_train_signatures(cfg)
    params = cfg.DATA.downstream

    out_dir = downstream_output_dir(cfg, "deconvolution")
    gt_cache_dir = os.path.join(out_dir, "eval", "gt")
    if params.get("share_gt_cache"):
        from downstream.common import load_gene_panel

        gt_cache_dir = deconv_gt_cache_dir(cfg.DATA.data_dir, load_gene_panel(cfg.MODEL.gene_path), params)
    os.makedirs(gt_cache_dir, exist_ok=True)

    if params.get("gt_seed_dir") and params.get("share_gt_cache"):
        import shutil

        os.makedirs(gt_cache_dir, exist_ok=True)
        for fn in os.listdir(params["gt_seed_dir"]):
            if fn.endswith(".csv") and not os.path.isfile(os.path.join(gt_cache_dir, fn)):
                shutil.copy(os.path.join(params["gt_seed_dir"], fn), os.path.join(gt_cache_dir, fn))

    def _run(adata, sample_id):
        _, abundance = run_cell2location_sample(
            adata, signatures, sample_name=sample_id,
            n_cells_per_location=params.get("n_cells_per_location", 30),
            detection_alpha=params.get("detection_alpha", 200),
            spatial_epochs=params.get("spatial_epochs", 30000),
            spatial_batch_size=params.get("spatial_batch_size", 2048),
            spatial_posterior_samples=params.get("spatial_posterior_samples", 1000),
            accelerator=accelerator,
        )
        return abundance

    rows: List[Dict[str, Any]] = []
    for sample_id in deconv_samples(cfg, params):
        pred_csv = os.path.join(out_dir, f"{sample_id}.q05_cell_abundance_w_sf.csv")
        if os.path.isfile(pred_csv):
            pred_abundance = pd.read_csv(pred_csv, index_col=0)
        else:
            pred_adata = to_count_scale(load_pred_adata(cfg.DATA.pred_path_fold, sample_id), source="pred", cpm=cpm)
            pred_abundance = _run(pred_adata, sample_id)

        gt_csv = os.path.join(gt_cache_dir, f"{sample_id}.q05_cell_abundance_w_sf.csv")
        if os.path.isfile(gt_csv):
            gt_abundance = pd.read_csv(gt_csv, index_col=0)
        else:
            try:
                gt_adata = load_gt_adata(cfg.DATA.data_dir, sample_id)
            except FileNotFoundError:
                continue
            gt_adata = to_count_scale(gt_adata, source="gt", cpm=cpm)
            gt_abundance = _run(gt_adata, sample_id)
            if gt_abundance is not None:
                gt_abundance.to_csv(gt_csv)

        if pred_abundance is None or gt_abundance is None:
            continue

        common_spots = pred_abundance.index.intersection(gt_abundance.index)
        common_celltypes = [c for c in pred_abundance.columns if c in gt_abundance.columns]
        for cell_type in common_celltypes:
            rows.append({
                "sample_id": sample_id,
                "cell_type": cell_type,
                "n_spots": int(len(common_spots)),
                "pearson": safe_corr(
                    gt_abundance.loc[common_spots, cell_type].to_numpy(),
                    pred_abundance.loc[common_spots, cell_type].to_numpy(),
                ),
            })

    metrics_path = os.path.join(out_dir, "eval", "metrics.csv")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(metrics_path, index=False)

    return {"mode": "deconvolution", "metrics_path": metrics_path, "per_sample": rows}
