"""Run gene-set enrichment scoring on PREDICTED ST expression.

Expected `cfg` shape (an addict.Dict, built by
`STPred._run_single_downstream_action`):

    cfg.DATA.pred_path_fold   # "{output_dir}/{data}/{model}/[{train_data}/]fold{k}"
    cfg.DATA.cpm              # bool — whether predictions are log1p(CPM) or log1p(raw)
    cfg.MODEL.gene_path       # trained gene panel json ({"genes": [...]})
    cfg.DATA.downstream       # merged gene_enrichment params: library, score_method,
                              # min_overlap, cache_dir (see config/downstream/defaults.yaml)
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from downstream.common import (
    downstream_output_dir,
    list_fold_samples,
    load_gene_panel,
    load_pred_adata,
    subset_to_genes,
    to_log_scale,
)

from .genesets import load_gene_sets
from .scoring import score_pathways


def run_gene_enrichment(cfg) -> Dict[str, Any]:
    params = cfg.DATA.downstream
    genes = load_gene_panel(cfg.MODEL.gene_path)
    gene_sets, overlap_counts = load_gene_sets(
        params.get("library", "MSigDB_Hallmark_2020"),
        genes,
        min_overlap=params.get("min_overlap", 5),
        cache_dir=params.get("cache_dir", "output/downstream_cache/genesets"),
    )

    out_dir = downstream_output_dir(cfg, "gene_enrichment")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "gene_sets_used.json"), "w") as f:
        json.dump(
            {"library": params.get("library"), "pathways": {t: overlap_counts[t] for t in gene_sets}},
            f,
            indent=2,
        )

    method = params.get("score_method", "rank_mean")
    manifest: Dict[str, str] = {}
    for sample_id in list_fold_samples(cfg):
        pred_adata = load_pred_adata(cfg.DATA.pred_path_fold, sample_id)
        pred_adata = subset_to_genes(pred_adata, genes)
        pred_adata = to_log_scale(pred_adata, source="pred", cpm=cfg.DATA.get("cpm", False))

        pred_x = pred_adata.X.toarray() if hasattr(pred_adata.X, "toarray") else pred_adata.X
        sample_gene_sets = {
            term: [g for g in gs if g in set(pred_adata.var_names)] for term, gs in gene_sets.items()
        }
        pred_scores = score_pathways(pred_x, list(pred_adata.var_names), sample_gene_sets, method=method)
        pred_scores.index = pred_adata.obs_names

        out_path = os.path.join(out_dir, f"{sample_id}.csv")
        pred_scores.to_csv(out_path)
        manifest[sample_id] = out_path

    return {"mode": "gene_enrichment", "output_dir": out_dir, "samples": manifest}
