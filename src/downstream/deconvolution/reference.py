"""cell2location reference cell-type signature training/caching, ported
from src/deconvolution/run_cell2location_xenium.py::load_reference_subset /
train_or_load_signatures in the main STpredBench repo.

Signatures depend only on the reference scRNA atlas + trained gene panel,
not on any particular model/fold, so `train_or_load_signatures` caches its
result at a (data, train_data)-scoped path shared across every model
evaluated on that data (see downstream/deconvolution/run.py).
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import sparse


def import_cell2location():
    try:
        from cell2location.models import Cell2location, RegressionModel
    except Exception as exc:  # pragma: no cover - environment check
        raise ImportError(
            "Could not import cell2location. Install it first (see requirements/downstreams/deconvolution.txt).\n"
            f"Original error: {exc}"
        ) from exc
    return Cell2location, RegressionModel


def get_accelerator(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "gpu" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def coerce_count_matrix(matrix):
    """cell2location's models require strict integer count dtype — unlike
    `downstream.common.to_count_scale` (which keeps float32 so SpaGCN's
    in-place normalization doesn't break), cast to int64 here right before
    handing data to cell2location."""
    if sparse.issparse(matrix):
        matrix = matrix.copy()
        matrix.data = np.nan_to_num(matrix.data, nan=0, posinf=None, neginf=0)
        matrix.data = np.rint(matrix.data)
        matrix.data = np.clip(matrix.data, 0, None).astype(np.int64, copy=False)
        matrix.eliminate_zeros()
        return matrix

    matrix = np.nan_to_num(np.asarray(matrix), nan=0, posinf=None, neginf=0)
    matrix = np.rint(matrix)
    matrix = np.clip(matrix, 0, None)
    return matrix.astype(np.int64, copy=False)


def load_reference_subset(
    ref_path: str,
    genes: List[str],
    labels_key: str,
    layer: str,
    batch_key: str,
    max_cells_per_label: Optional[int],
    seed: int,
):
    import scanpy as sc

    use_x_layer = layer.lower() in {"x", "none"}
    backed = sc.read_h5ad(ref_path, backed="r")
    try:
        if labels_key not in backed.obs:
            raise KeyError(f"{labels_key!r} not found in reference .obs")
        if batch_key not in backed.obs:
            raise KeyError(f"{batch_key!r} not found in reference .obs")
        if not use_x_layer and layer not in backed.layers:
            raise KeyError(f"{layer!r} not found in reference .layers")
        if "feature_name" not in backed.var:
            raise KeyError("'feature_name' not found in reference .var")

        feature_names = backed.var["feature_name"].astype(str)
        gene_set = set(genes)
        keep_var = feature_names.isin(gene_set).to_numpy()
        adata = backed[:, keep_var].to_memory()
        adata.var_names = feature_names.loc[keep_var].to_numpy()
        adata.var_names_make_unique()
    finally:
        backed.file.close()

    adata = adata[~adata.obs[labels_key].isna()].copy()
    adata.X = coerce_count_matrix(adata.X)
    if not use_x_layer and layer in adata.layers:
        adata.layers[layer] = coerce_count_matrix(adata.layers[layer])

    if max_cells_per_label:
        rng = np.random.default_rng(seed)
        selected = []
        for _, obs_idx in adata.obs.groupby(labels_key, observed=True).indices.items():
            obs_idx = np.asarray(obs_idx)
            if len(obs_idx) > max_cells_per_label:
                obs_idx = rng.choice(obs_idx, size=max_cells_per_label, replace=False)
            selected.append(obs_idx)
        selected_idx = np.concatenate(selected)
        selected_idx.sort()
        adata = adata[selected_idx].copy()

    return adata


def train_or_load_signatures(
    reference_path: str,
    genes: List[str],
    labels_key: str,
    batch_key: str,
    layer: str,
    cache_path: str,
    reference_epochs: int = 250,
    reference_batch_size: int = 2500,
    reference_posterior_samples: int = 1000,
    accelerator: str = "cpu",
    max_cells_per_label: Optional[int] = None,
    seed: int = 0,
    force: bool = False,
) -> pd.DataFrame:
    """Returns genes x cell_types signature matrix, cached at `cache_path`."""
    if os.path.isfile(cache_path) and not force:
        return pd.read_csv(cache_path, index_col=0)

    _, RegressionModel = import_cell2location()
    adata_ref = load_reference_subset(
        ref_path=reference_path, genes=genes, labels_key=labels_key,
        layer=layer, batch_key=batch_key, max_cells_per_label=max_cells_per_label, seed=seed,
    )

    setup_kwargs = {"adata": adata_ref, "labels_key": labels_key, "batch_key": batch_key}
    if layer.lower() not in {"x", "none"}:
        setup_kwargs["layer"] = layer
    RegressionModel.setup_anndata(**setup_kwargs)
    model = RegressionModel(adata_ref)
    model.train(max_epochs=reference_epochs, batch_size=reference_batch_size, accelerator=accelerator)
    adata_ref = model.export_posterior(
        adata_ref, sample_kwargs={"num_samples": reference_posterior_samples, "batch_size": reference_batch_size},
    )

    factor_names = adata_ref.uns["mod"]["factor_names"]
    if "means_per_cluster_mu_fg" in adata_ref.varm:
        signatures = adata_ref.varm["means_per_cluster_mu_fg"][
            [f"means_per_cluster_mu_fg_{name}" for name in factor_names]
        ].copy()
    else:
        signatures = adata_ref.var[[f"means_per_cluster_mu_fg_{name}" for name in factor_names]].copy()
    signatures.columns = factor_names
    signatures.index = adata_ref.var_names

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    signatures.to_csv(cache_path)
    model.save(cache_path.rsplit(".", 1)[0] + ".model", overwrite=True)
    return signatures
