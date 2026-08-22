"""Run cell2location deconvolution on PREDICTED ST expression.

Expected `cfg` shape — see downstream/gene_enrichment/run.py's docstring,
plus `cfg.DATA.output_dir`, `cfg.DATA.name`, `cfg.DATA.train_data_name`
(needed to locate the shared, model-independent reference-signature cache).
`cfg.DATA.downstream` must include `reference_path` (required, no default —
see config/downstream/defaults.yaml / DATA.downstream.deconvolution in the
data config).
"""

from __future__ import annotations

import os
from typing import Any, Dict

from downstream.common import (
    downstream_output_dir,
    list_fold_samples,
    load_gene_panel,
    load_pred_adata,
    to_count_scale,
)

from .reference import get_accelerator, train_or_load_signatures
from .spatial import run_cell2location_sample


def _signatures_cache_path(cfg) -> str:
    return os.path.join(
        cfg.DATA.output_dir, cfg.DATA.name, "_downstream_cache", "deconvolution",
        cfg.DATA.train_data_name, "reference_signatures.csv",
    )


def _load_or_train_signatures(cfg):
    params = cfg.DATA.downstream
    genes = load_gene_panel(cfg.MODEL.gene_path)
    accelerator = get_accelerator(params.get("accelerator", "auto"))
    return train_or_load_signatures(
        reference_path=params["reference_path"],
        genes=genes,
        labels_key=params.get("labels_key", "cell_type_predicted"),
        batch_key=params.get("batch_key", "sample"),
        layer=params.get("reference_layer", "count"),
        cache_path=_signatures_cache_path(cfg),
        reference_epochs=params.get("reference_epochs", 250),
        reference_batch_size=params.get("reference_batch_size", 2500),
        reference_posterior_samples=params.get("reference_posterior_samples", 1000),
        accelerator=accelerator,
        max_cells_per_label=params.get("max_cells_per_label"),
    ), accelerator


def run_deconvolution(cfg) -> Dict[str, Any]:
    params = cfg.DATA.downstream
    cpm = cfg.DATA.get("cpm", False)
    signatures, accelerator = _load_or_train_signatures(cfg)

    out_dir = downstream_output_dir(cfg, "deconvolution")
    os.makedirs(out_dir, exist_ok=True)

    manifest: Dict[str, str] = {}
    for sample_id in list_fold_samples(cfg):
        adata = load_pred_adata(cfg.DATA.pred_path_fold, sample_id)
        adata = to_count_scale(adata, source="pred", cpm=cpm)
        result_adata, abundance = run_cell2location_sample(
            adata, signatures, sample_name=sample_id,
            n_cells_per_location=params.get("n_cells_per_location", 30),
            detection_alpha=params.get("detection_alpha", 200),
            spatial_epochs=params.get("spatial_epochs", 30000),
            spatial_batch_size=params.get("spatial_batch_size", 2048),
            spatial_posterior_samples=params.get("spatial_posterior_samples", 1000),
            accelerator=accelerator,
        )

        h5ad_path = os.path.join(out_dir, f"{sample_id}.h5ad")
        result_adata.write_h5ad(h5ad_path)
        if abundance is not None:
            abundance.to_csv(os.path.join(out_dir, f"{sample_id}.q05_cell_abundance_w_sf.csv"))
        manifest[sample_id] = h5ad_path

    return {
        "mode": "deconvolution", "output_dir": out_dir, "samples": manifest,
        "signatures_path": _signatures_cache_path(cfg),
    }
