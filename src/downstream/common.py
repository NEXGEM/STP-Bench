"""Shared I/O and scale-conversion helpers for downstream analyses.

Every downstream mode (gene_enrichment, deconvolution, spatial_domain) reads
predicted ST from `{pred_path_fold}/{sample_id}.h5ad` (the same convention
`run_evaluation`/`run_inference` already establish, see
`core/utils/train_utils.py`) and ground truth from `{st_dir(data_dir)}/
{sample_id}.h5ad` (the same source `STDataset.load_st` reads from, see
`dataset/base_dataset.py`).

Scale note: predicted `.h5ad` files are always in log1p (or log1p(CPM) if
`DATA.cpm=True`) scale, because that's the scale `normalize_adata()` puts
training targets in and `BaseModule.save_predictions()` writes model output
back out with no inverse transform. Ground-truth `.h5ad` files on disk are
raw counts. `to_log_scale`/`to_count_scale` below convert each source to
whichever scale a given downstream algorithm expects.
"""

from __future__ import annotations

import json
import os
import warnings
from glob import glob
from typing import Any, Dict, List, Sequence


def _st_dir(data_dir: str) -> str:
    """Same logic as `dataset.path_utils.st_dir`, duplicated (not imported)
    because `dataset/__init__.py` unconditionally imports every dataset
    class (cv2, torch_geometric, ...) as a side effect of importing ANY
    submodule of `dataset` — even a direct `from dataset.path_utils import
    st_dir` triggers that whole chain. Downstream analyses (gene_enrichment
    in particular) don't need any of it just to resolve one path."""
    st_path = os.path.join(data_dir, "st")
    adata_path = os.path.join(data_dir, "adata")
    return st_path if os.path.isdir(st_path) or not os.path.isdir(adata_path) else adata_path


def load_gt_adata(data_dir: str, sample_id: str):
    """Load ground-truth ST for one sample, raw counts, no transform."""
    import scanpy as sc

    path = os.path.join(_st_dir(data_dir), f"{sample_id}.h5ad")
    return sc.read_h5ad(path)


def load_pred_adata(pred_path_fold: str, sample_id: str):
    """Load predicted ST for one sample, log1p (or log1p(CPM)) scale, as written."""
    import scanpy as sc

    path = os.path.join(pred_path_fold, f"{sample_id}.h5ad")
    return sc.read_h5ad(path)


def list_fold_samples(cfg) -> List[str]:
    """Sample IDs to run this fold's downstream analysis on.

    Restricts to samples actually assigned "test" for this fold in `data`'s
    own `ids.csv` (`fold_<fold>` column), intersected with what has a
    prediction file under `{pred_path_fold}/{sample_id}.h5ad`. A plain glob
    of the prediction directory is NOT enough on its own: that directory can
    accumulate stale `.h5ad` files from unrelated runs (a different fold, an
    old experiment, a WSI-only predict() call) that were never part of this
    fold's real test set, and would otherwise get silently ingested here too.

    Falls back to the plain glob (with a warning) when `cfg.DATA.meta_dir`/
    `cfg.DATA.fold` aren't set, or when `ids.csv` has no `fold_<fold>` column
    at all -- e.g. an external dataset with no CV split, where every sample
    is legitimately available under every fold's checkpoint.
    """
    pred_path_fold = cfg.DATA.pred_path_fold
    available = {
        os.path.splitext(os.path.basename(path))[0]
        for path in glob(os.path.join(pred_path_fold, "*.h5ad"))
    }

    meta_dir = cfg.DATA.get("meta_dir")
    fold = cfg.DATA.get("fold")
    if meta_dir is None or fold is None:
        warnings.warn(
            "list_fold_samples() called without cfg.DATA.meta_dir/fold -- falling "
            "back to globbing every *.h5ad in the prediction directory, which "
            "cannot distinguish this fold's real test set from stale files left "
            "by unrelated runs.",
            stacklevel=2,
        )
        return sorted(available)

    import pandas as pd

    ids_path = os.path.join(meta_dir, "ids.csv")
    ids = pd.read_csv(ids_path)
    fold_col = f"fold_{fold}"
    if fold_col not in ids.columns:
        return sorted(set(ids["sample_id"].astype(str)) & available)

    test_ids = set(ids.loc[ids[fold_col] == "test", "sample_id"].astype(str))
    return sorted(test_ids & available)


def load_gene_panel(gene_path: str) -> List[str]:
    """Read the trained gene panel list, same `{'genes': [...]}` shape as
    `_external_gene_overlap`/`_user_gene_overlap` in api/stpbench.py."""
    with open(gene_path) as f:
        return json.load(f)["genes"]


def subset_to_genes(adata, genes: Sequence[str]):
    """Replicate STDataset.load_st's gene-subsetting behavior (warn + subset
    to the intersection on partial overlap) without importing STDataset
    itself, so downstream code isn't coupled to Lightning dataset machinery."""
    genes = list(genes)
    if adata.var_names.isin(genes).sum() < len(genes):
        common_genes = [g for g in genes if g in set(adata.var_names)]
        warnings.warn(
            f"Some requested genes are not found in this sample's h5ad. "
            f"Using {len(common_genes)} / {len(genes)} genes.",
            stacklevel=2,
        )
        return adata[:, common_genes].copy()
    return adata[:, genes].copy()


def downstream_output_dir(cfg, mode: str) -> str:
    """Extends the existing `pred_path_fold` convention with a `downstream/
    <mode>/` suffix — no new top-level output namespace is introduced."""
    return os.path.join(cfg.DATA.pred_path_fold, "downstream", mode)


def to_log_scale(adata, source: str, cpm: bool):
    """Bring GT/pred to the same log-scale gene_enrichment scores directly on.

    source='pred': predictions are already log1p (or log1p(CPM)) — returned
    unchanged. source='gt': GT h5ad on disk is raw counts — apply the exact
    same `normalize_adata()` used to build training targets, so GT and pred
    end up in the identical scale before pathway scoring.
    """
    if source == "pred":
        return adata
    if source == "gt":
        return _normalize_adata(adata, cpm=cpm)
    raise ValueError(f"Unknown source: {source!r}; expected 'pred' or 'gt'.")


def _normalize_adata(adata, cpm: bool = False, smooth: bool = False):
    """Byte-for-byte copy of `core.utils.train_utils.normalize_adata` — the
    exact function used to build training targets — duplicated rather than
    imported because `core/__init__.py` unconditionally imports
    BaseModule/BaseDataModule (pulling in torch_geometric etc.) as a side
    effect of importing ANY submodule of `core`, even a direct `from
    core.utils.train_utils import normalize_adata`. Downstream analyses
    (gene_enrichment in particular) don't need the Lightning training stack
    just to reuse this one function. Keep in sync with the original if it
    ever changes."""
    import numpy as np
    import scanpy as sc

    normed_adata = adata.copy()

    if cpm:
        sc.pp.normalize_total(normed_adata, target_sum=1e4)

    sc.pp.log1p(normed_adata)

    if smooth:
        new_X = []
        for _, df_row in normed_adata.obs.iterrows():
            row = int(df_row["array_row"])
            col = int(df_row["array_col"])

            neighbors_index = normed_adata.obs[
                ((normed_adata.obs["array_row"] >= row - 1) & (normed_adata.obs["array_row"] <= row + 1))
                & ((normed_adata.obs["array_col"] >= col - 1) & (normed_adata.obs["array_col"] <= col + 1))
            ].index
            neighbors = normed_adata[neighbors_index]
            nb_neighbors = len(neighbors)

            avg = neighbors.X.sum(0) / nb_neighbors
            new_X.append(avg)

        new_X = np.stack(new_X)
        normed_adata.X = new_X

    return normed_adata


def to_count_scale(adata, source: str, cpm: bool, min_value: int = 0):
    """Bring GT/pred to a raw-count-like scale for deconvolution/spatial_domain.

    source='pred': if cpm=False, np.expm1 is the exact inverse of the log1p
    applied to training targets, recovering raw counts (then rounded/clipped).
    If cpm=True, expm1 recovers CPM-normalized values, not true raw UMI
    counts (library-size information was lost) — a warning is emitted and
    the caller should prefer a cpm=False model/fold for these two modes.
    source='gt': GT h5ad on disk is already raw counts — only rounding/
    clipping is applied, no expm1.

    Values are rounded to integers but kept as float32 (not cast to int64):
    SpaGCN's own preprocessing (`sc.pp.normalize_per_cell`) divides in place
    and fails on an integer dtype array. cell2location, which does require
    strict integer dtype, casts this float32-but-integer-valued array to
    int64 itself right before use (see
    `downstream/deconvolution/reference.py`'s `coerce_count_matrix`) — the
    same split the main repo's own scripts make (SpaGCN's
    `transform_count_matrix` casts to float32; cell2location's casts to
    int64).
    """
    import numpy as np
    from scipy import sparse

    if source not in ("pred", "gt"):
        raise ValueError(f"Unknown source: {source!r}; expected 'pred' or 'gt'.")

    adata = adata.copy()
    X = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)

    if source == "pred":
        if cpm:
            warnings.warn(
                "to_count_scale(source='pred', cpm=True): predictions are in "
                "log1p(CPM) scale, so expm1 recovers CPM-normalized values, "
                "NOT true raw counts (library-size information was lost during "
                "training-time normalization). Deconvolution/spatial-domain "
                "results computed on this data are an approximation only — "
                "prefer a cpm=False trained model/fold for these two modes.",
                stacklevel=2,
            )
        X = np.expm1(X)

    X = np.rint(X)
    X = np.clip(X, min_value, None)
    adata.X = X.astype(np.float32)
    return adata


def safe_corr(x, y, method: str = "pearson") -> float:
    """Pearson/Spearman correlation, NaN-safe: drops non-finite pairs and
    returns NaN when fewer than 3 finite pairs remain or either side is
    constant (zero variance). Shared by gene_enrichment (per-pathway) and
    deconvolution (per-cell-type) evaluation."""
    import warnings

    import numpy as np
    from scipy.stats import pearsonr, spearmanr

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    x, y = x[mask], y[mask]
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*correlation coefficient is not defined.*")
        if method == "spearman":
            return spearmanr(x, y).correlation
        return pearsonr(x, y).statistic
