"""Path helpers for dataset manifests."""

from __future__ import annotations

import os


def patch_dir(data_dir: str) -> str:
    return os.path.join(data_dir, "patches")


def st_dir(data_dir: str) -> str:
    st_path = os.path.join(data_dir, "st")
    adata_path = os.path.join(data_dir, "adata")
    return st_path if os.path.isdir(st_path) or not os.path.isdir(adata_path) else adata_path


def emb_dir(data_dir: str) -> str:
    return os.path.join(data_dir, "emb")


# Aliases used in combine_dataset.py
resolve_patch_dir = patch_dir
resolve_st_dir = st_dir
resolve_emb_dir = emb_dir
