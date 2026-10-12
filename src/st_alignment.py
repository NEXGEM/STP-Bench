"""Pair ST spots with their image patches at load time.

Preprocessing never rewrites the ST files under `<asset_dir>/st` (or `adata`): they keep every
spot of the original data, together with `obsm`/`uns`. Which spots are used, and in which order,
is defined by the `barcode` dataset of `<asset_dir>/patches/<sample>.h5` -- patch extraction
drops spots that fall outside the slide or off tissue. Every reader that needs ST rows to line up
with patches goes through `read_st()` / `align_to_patches()`, so row i of the returned AnnData
and patch i are the same spot.

A top-level module with no heavy imports (no torch / core / dataset packages), so lightweight
scripts (get_geneset.py, downstream) can use it, and model scripts that are themselves named
preprocess.py do not shadow it.
"""
import os

import h5py
import numpy as np


def patch_path_for_st(st_path):
    """Patch h5 that belongs to `<asset_dir>/{st,adata}/<sample>.h5ad`, or None if there is none."""
    asset_dir = os.path.dirname(os.path.dirname(os.path.abspath(st_path)))
    name = os.path.splitext(os.path.basename(st_path))[0]
    for fname in (f"{name}.h5", f"{name}_patches.h5"):
        path = os.path.join(asset_dir, "patches", fname)
        if os.path.isfile(path):
            return path
    return None


def read_patch_barcodes(patch_path):
    """Barcodes of a patch h5 as a str array, or None if the file has no barcode dataset."""
    with h5py.File(patch_path, "r") as f:
        key = "barcodes" if "barcodes" in f else "barcode"
        if key not in f:
            return None
        return f[key][:].reshape(-1).astype(str)


def align_to_patches(adata, patch_path):
    """Subset and reorder `adata` rows to the barcode order of `patch_path`.

    Returns `adata` itself when it already matches (or there is no patch file / barcode list).
    Raises if a patch barcode has no spot in `adata`: dropping it silently would shift every
    later row against its patch.
    """
    if patch_path is None:
        return adata
    barcodes = read_patch_barcodes(patch_path)
    if barcodes is None:
        return adata
    if len(barcodes) == adata.n_obs and np.array_equal(np.asarray(adata.obs_names, dtype=str), barcodes):
        return adata
    if not adata.obs_names.is_unique:
        raise ValueError(f"Cannot pair spots with {patch_path}: the ST data has duplicated obs names.")
    idx = adata.obs_names.get_indexer(barcodes)
    missing = barcodes[idx < 0]
    if len(missing):
        raise ValueError(
            f"{len(missing)}/{len(barcodes)} patch barcodes of {patch_path} have no spot in the ST data "
            f"(e.g. {', '.join(missing[:3])}); the ST file and the patches come from different spot tables.")
    return adata[idx].copy()


def read_st(st_path):
    """Read an ST h5ad and pair its rows with the sample's patches (see module docstring)."""
    import scanpy as sc

    return align_to_patches(sc.read_h5ad(st_path), patch_path_for_st(st_path))
