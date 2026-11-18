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
from .evaluate import silk_gt_cache_dir, spatial_domain_samples


def run_spatial_domain(cfg) -> Dict[str, Any]:
    params = cfg.DATA.downstream
    cpm = cfg.DATA.get("cpm", False)
    genes = load_gene_panel(params.get("gene_path") or cfg.MODEL.gene_path)
    spg = import_spagcn()

    out_dir = downstream_output_dir(cfg, "spatial_domain_silk" if params.get("n_cluster_candidates") else "spatial_domain")
    os.makedirs(out_dir, exist_ok=True)

    manifest: Dict[str, str] = {}
    for sample_id in spatial_domain_samples(cfg, params):
        adata = load_pred_adata(cfg.DATA.pred_path_fold, sample_id)
        adata = subset_to_genes(adata, genes)
        adata = to_count_scale(adata, source="pred", cpm=cpm)
        k_ref = None
        if params.get("n_cluster_candidates"):
            # k is chosen on GT only; reuse the cached GT k for this sample. Without a
            # cached GT result there is no valid k yet, so leave clustering to evaluate().
            gt_path = os.path.join(silk_gt_cache_dir(cfg.DATA.data_dir, genes, params), f"{sample_id}.h5ad")
            if not os.path.isfile(gt_path):
                continue
            import scanpy as sc

            k_ref = int(sc.read_h5ad(gt_path).uns["spagcn"]["n_clusters"])
        clustered = run_spagcn_sample(spg, adata, params, n_clusters=k_ref)

        out_path = os.path.join(out_dir, f"{sample_id}.h5ad")
        clustered.write_h5ad(out_path)
        manifest[sample_id] = out_path

    return {"mode": "spatial_domain", "output_dir": out_dir, "samples": manifest}
