"""Run SpaGCN spatial-domain clustering on PREDICTED ST expression.

Expected `cfg` shape — see downstream/gene_enrichment/run.py's docstring;
same contract, `cfg.DATA.downstream` here holds spatial_domain params
(see config/downstream/defaults.yaml).
"""

from __future__ import annotations

import os
from typing import Any, Dict

from downstream.common import (
    downstream_output_dir,
    list_fold_samples,
    load_gene_panel,
    load_pred_adata,
    subset_to_genes,
    to_count_scale,
)

from .clustering import import_spagcn, run_spagcn_sample


def run_spatial_domain(cfg) -> Dict[str, Any]:
    params = cfg.DATA.downstream
    cpm = cfg.DATA.get("cpm", False)
    genes = load_gene_panel(cfg.MODEL.gene_path)
    spg = import_spagcn()

    out_dir = downstream_output_dir(cfg, "spatial_domain")
    os.makedirs(out_dir, exist_ok=True)

    manifest: Dict[str, str] = {}
    for sample_id in list_fold_samples(cfg):
        adata = load_pred_adata(cfg.DATA.pred_path_fold, sample_id)
        adata = subset_to_genes(adata, genes)
        adata = to_count_scale(adata, source="pred", cpm=cpm)
        clustered = run_spagcn_sample(spg, adata, params)

        out_path = os.path.join(out_dir, f"{sample_id}.h5ad")
        clustered.write_h5ad(out_path)
        manifest[sample_id] = out_path

    return {"mode": "spatial_domain", "output_dir": out_dir, "samples": manifest}
