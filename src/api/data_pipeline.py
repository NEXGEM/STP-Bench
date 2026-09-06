"""Data preprocessing pipeline exposed via api namespace."""

import os
import sys
import argparse
import subprocess
import warnings
from glob import glob
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple

import h5py

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from preprocess.pipeline.utils import setup_paths, get_available_gpus
from preprocess.pipeline.preprocess import (
    preprocess_data,
    extract_features_single,
    extract_features_parallel,
)


def _wants_feature(feature_type, name: str) -> bool:
    """Check whether `name` (e.g. 'neighbor') is requested by a feature_type
    value that may be a single string ('all', 'global', ...) or a list of
    feature names (e.g. ['global', 'neighbor'])."""
    if isinstance(feature_type, (list, tuple, set)):
        return name in feature_type
    return feature_type in ('all', name)


class DataPipeline:
    """
    Unified data processing pipeline for ST.

    The pipeline integrates:
    - Data preprocessing for ST data and WSI images
    - Feature extraction for image data
    - Model training and inference hooks
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.repo_root = os.path.abspath(self.config.get('repo_root', Path(__file__).resolve().parents[2]))
        self.python = self.config.get('python', sys.executable)
        self._setup_defaults()
        self._setup_dirs()
        self.total_gpus = len(self.config['gpus'])
        self.mode = self.config['mode']

    def _script_path(self, relative_path: str) -> str:
        return os.path.join(self.repo_root, relative_path)

    def _setup_defaults(self):
        """Set default configuration values if not provided."""
        defaults = {
            'mode': 'raw',
            'platform': 'visium',
            'slide_ext': '.svs',
            'patch_size': 224,
            'slide_level': 0,
            'n_splits': 4,
            'n_top_hvg': 50,
            'n_top_heg': 1000,
            'n_top_hmhvg': 200,
            'patch_encoder': 'cigar',
            'num_t': 9,
            'num_n': 25,
            'dst_pixel_size': 0.5,
            'batch_size': 512,
            'num_workers': 4,
            'gpus': [0],
            'feature_type': 'global',
            'geneset': 'HMHVG',
            'overwrite': False,
            'save_neighbor_imgs': False,
            'extract_from_wsi': False,
        }
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value

    def _abs(self, path: str) -> str:
        """Resolve path relative to repo_root if not already absolute."""
        if not path:
            return path
        return path if os.path.isabs(path) else os.path.join(self.repo_root, path)

    def _setup_dirs(self):
        """Setup directory structure."""
        self.input_dir = self._abs(self.config.get('input_dir'))
        self.output_dir = self._abs(self.config.get('output_dir'))
        if not self.output_dir:
            raise ValueError("output_dir must be specified")

        mode = self.config['mode']
        self.mode = mode
        self.asset_dir = self.output_dir
        # Plain 'inference' mode (extract_from_wsi=False) never extracts
        # patches — they already exist under input_dir (which may be a
        # separate, read-only asset directory from output_dir; see
        # STPred._materialize_wsi_data_config's asset_dir kind). Every
        # other case (raw/stpbench, or 'inference' with
        # extract_from_wsi=True — a raw-WSI predict() target) extracts
        # patches INTO asset_dir itself, so that's where they're read back
        # from. Feature extraction reads patches from patch_source_dir but
        # still writes embeddings under asset_dir (writable) regardless.
        self.patch_source_dir = (
            self.input_dir
            if mode == 'inference' and not self.config.get('extract_from_wsi')
            else self.asset_dir
        )
        self.metadata_dir = self._abs(self.config.get('meta_dir')) or self.output_dir
        self.wsi_dataroot = f"{self.input_dir}/wsis" if mode == 'stpbench' else self.input_dir
        feature_type = self.config.get('feature_type')
        needed_features = tuple(
            name for name in ('global', 'neighbor', 'target') if _wants_feature(feature_type, name)
        )
        self.dirs = setup_paths(self.output_dir, features=needed_features)

    def preprocess(self):
        """Run preprocessing steps based on mode."""
        mode = self.config['mode']
        print(f"Running preprocessing for mode: {mode}")
        if mode == 'stpbench':
            ids_path = os.path.join(self.metadata_dir, 'ids.csv')
            if not os.path.isfile(ids_path):
                raise FileNotFoundError(
                    "[stpbench mode] ids.csv must be prepared before running preprocessing.\n"
                    f"  Expected: {ids_path}\n"
                    "  Create a CSV with a 'sample_id' column listing the samples for this dataset."
                )
        # 'inference' is excluded here (whether or not extract_from_wsi is
        # set): it's a per-target-identity mode where a dataset-wide ids.csv
        # snapshot doesn't mean "this call's specific slide is done" — it may
        # be stale, left over from a DIFFERENT slide predicted earlier into
        # the same output_dir. extract_patches_from_wsi/
        # extract_patches_from_coords_file already do their own correct,
        # per-slide skip-if-exists check (see 'exists, skip!').
        if mode in ['raw', 'stpbench'] and self._has_processed_data() and not self.config['overwrite']:
            print(f"Processed data already found at {self.output_dir}. Skipping raw preprocessing.")
        else:
            save_neighbors = _wants_feature(self.config['feature_type'], 'neighbor')
            # preprocess_data() runs prepare_data.py as a subprocess and
            # raises RuntimeError (with the subprocess's full output
            # embedded) on a nonzero exit code -- a per-sample crash partway
            # through (e.g. one bad WSI file) must not be silently treated
            # as "raw preprocessing done", or every step downstream
            # (genesets/splits/feature extraction) proceeds against an
            # incomplete patch set and fails confusingly much later, far
            # from the actual cause.
            try:
                preprocess_data(
                    input_dir=self.input_dir,
                    output_dir=self.asset_dir,
                    meta_dir=self.metadata_dir,
                    mode=mode,
                    platform=self.config['platform'],
                    slide_ext=self.config['slide_ext'],
                    patch_size=self.config['patch_size'],
                    slide_level=self.config['slide_level'],
                    save_neighbors=save_neighbors,
                    save_neighbor_imgs=self.config['save_neighbor_imgs'] if save_neighbors else False,
                    num_n=self.config['num_n'],
                    dst_pixel_size=self.config['dst_pixel_size'],
                    overwrite=self.config['overwrite'],
                    coords_path=self.config.get('coords_path'),
                    extract_from_wsi=self.config.get('extract_from_wsi', False),
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Raw preprocessing failed for data_dir={self.asset_dir!r} (mode={mode!r}): {exc}"
                ) from exc


    def align_st(self, sample_ids: list = None, overwrite: bool = False):
        """Align h5ad files to their patch barcodes and re-save.

        sample_ids: if given, only align those samples; otherwise scan all st/*.h5ad.
        """
        from preprocess.prepare_data import align_st_to_patches
        align_st_to_patches(self.asset_dir, sample_ids=sample_ids, overwrite=overwrite)

    def prepare_genesets(self):
        """Prepare gene sets for training."""
        if self._has_genesets() and not self.config['overwrite']:
            print(f"Gene sets already found at {self.output_dir}. Skipping gene set preparation.")
            return
        print("Preparing gene sets...")
        cmd = [
            self.python,
            self._script_path("src/preprocess/get_geneset.py"),
            "--st_dir",
            f"{self.asset_dir}/st",
            "--output_dir",
            self.metadata_dir,
            "--id_path",
            f"{self.metadata_dir}/ids.csv",
            "--n_top_hvg",
            str(self.config['n_top_hvg']),
            "--n_top_heg",
            str(self.config['n_top_heg']),
            "--n_top_hmhvg",
            str(self.config['n_top_hmhvg']),
            "--method",
            str(self.config['geneset']),
        ]
        subprocess.run(cmd, check=True, cwd=self.repo_root)

    def split_data(self):
        """Split data for cross-validation."""
        if self._has_splits() and not self.config['overwrite']:
            print(f"Cross-validation splits already found at {self.output_dir}. Skipping split generation.")
            return
        print("Splitting data for cross-validation...")
        cmd = [
            self.python,
            self._script_path("src/preprocess/split_data.py"),
            "--input_dir",
            self.metadata_dir,
            "--n_splits",
            str(self.config['n_splits']),
        ]
        subprocess.run(cmd, check=True, cwd=self.repo_root)

    def run_extraction(self, transform_type: str = 'eval'):
        """Extract image features."""
        feature_type = self.config['feature_type']
        assert feature_type in ['global', 'neighbor', 'target', 'all'], \
            "feature_type must be 'global', 'neighbor', 'target' or 'all'"

        self._reuse_existing_embeddings(
            [f for f in ('global', 'neighbor', 'target') if _wants_feature(feature_type, f)]
        )

        # For an asset-dir predict() target (mode='inference',
        # extract_from_wsi=False), neighbor patches live in the read-only
        # SOURCE dir (patch_source_dir), not the writable asset_dir --
        # every other case extracts (or already has) them under asset_dir
        # itself. Computed once here, not duplicated per GPU-count branch
        # below: `_drop_neighbor_imgs()` deletes 'img' from whatever this
        # points at post-extraction, and having it be right in only one of
        # two copies is exactly the kind of drift that caused this path to
        # corrupt patches/*.h5 in the read-only asset dir in the first
        # place (see _drop_neighbor_imgs's own read-only guard).
        neighbor_patch_dataroot = (
            f"{self.patch_source_dir}/patches/neighbor"
            if self.config['mode'] == 'inference' and not self.config.get('extract_from_wsi')
            else f"{self.asset_dir}/patches/neighbor"
        )

        gpus = self.config['gpus']
        if self.total_gpus > 1:
            print(f"Running feature extraction in parallel on {self.total_gpus} GPUs...")
            sample_ids = self._get_sample_ids()
            if feature_type in ['global', 'all']:
                print("Extracting global features...")
                ok = extract_features_parallel(
                    wsi_dataroot=self.wsi_dataroot,
                    patch_dataroot=f"{self.patch_source_dir}/patches",
                    embed_dataroot=f"{self.asset_dir}/emb/global",
                    slide_ext=self.config['slide_ext'],
                    patch_encoder=self.config['patch_encoder'],
                    num_tiles=1,
                    feature_type='global',
                    batch_size=self.config['batch_size'],
                    num_workers=self.config['num_workers'],
                    overwrite=self.config['overwrite'],
                    gpus=gpus,
                    transform_type=transform_type,
                    sample_ids=sample_ids,
                    mode=self.mode,
                )
                self._raise_if_extraction_failed(ok, "global")
            # neighbor_patch_dataroot: computed once, above -- see there.
            if feature_type in ['neighbor', 'all']:
                print("Extracting neighbor features...")
                ok = extract_features_parallel(
                    wsi_dataroot=self.wsi_dataroot,
                    patch_dataroot=neighbor_patch_dataroot,
                    embed_dataroot=f"{self.asset_dir}/emb/neighbor",
                    slide_ext=self.config['slide_ext'],
                    patch_encoder=self.config['patch_encoder'],
                    num_tiles=self.config['num_n'],
                    feature_type='neighbor',
                    batch_size=self.config['batch_size'],
                    num_workers=self.config['num_workers'],
                    overwrite=self.config['overwrite'],
                    gpus=gpus,
                    transform_type=transform_type,
                    sample_ids=sample_ids,
                    mode=self.mode,
                )
                self._raise_if_extraction_failed(ok, "neighbor")
                if self.config.get('save_neighbor_imgs'):
                    self._drop_neighbor_imgs(neighbor_patch_dataroot)
            if feature_type in ['target', 'all']:
                print("Extracting target features...")
                ok = extract_features_parallel(
                    wsi_dataroot=self.wsi_dataroot,
                    patch_dataroot=f"{self.patch_source_dir}/patches",
                    embed_dataroot=f"{self.asset_dir}/emb/target",
                    slide_ext=self.config['slide_ext'],
                    patch_encoder=self.config['patch_encoder'],
                    num_tiles=self.config['num_t'],
                    feature_type='target',
                    batch_size=self.config['batch_size'],
                    num_workers=self.config['num_workers'],
                    overwrite=self.config['overwrite'],
                    gpus=gpus,
                    transform_type=transform_type,
                    sample_ids=sample_ids,
                    mode=self.mode,
                )
                self._raise_if_extraction_failed(ok, "target")
        else:
            print("Running feature extraction in single GPU mode...")
            if feature_type in ['global', 'all']:
                print("Extracting global features...")
                ok = extract_features_single(
                    wsi_dataroot=self.wsi_dataroot,
                    patch_dataroot=f"{self.patch_source_dir}/patches",
                    embed_dataroot=f"{self.asset_dir}/emb/global",
                    slide_ext=self.config['slide_ext'],
                    patch_encoder=self.config['patch_encoder'],
                    num_tiles=1,
                    feature_type='global',
                    batch_size=self.config['batch_size'],
                    num_workers=self.config['num_workers'],
                    overwrite=self.config['overwrite'],
                    gpus=self.config['gpus'],
                    transform_type=transform_type,
                    id_path=f"{self.metadata_dir}/ids.csv",
                    mode=self.mode,
                )
                self._raise_if_extraction_failed(ok, "global")
            # neighbor_patch_dataroot: computed once, above -- see there.
            if feature_type in ['neighbor', 'all']:
                print("Extracting neighbor features...")
                ok = extract_features_single(
                    wsi_dataroot=self.wsi_dataroot,
                    patch_dataroot=neighbor_patch_dataroot,
                    embed_dataroot=f"{self.asset_dir}/emb/neighbor",
                    slide_ext=self.config['slide_ext'],
                    patch_encoder=self.config['patch_encoder'],
                    num_tiles=self.config['num_n'],
                    feature_type='neighbor',
                    batch_size=self.config['batch_size'],
                    num_workers=self.config['num_workers'],
                    overwrite=self.config['overwrite'],
                    gpus=self.config['gpus'],
                    transform_type=transform_type,
                    id_path=f"{self.metadata_dir}/ids.csv",
                    mode=self.mode,
                )
                self._raise_if_extraction_failed(ok, "neighbor")
                if self.config.get('save_neighbor_imgs'):
                    self._drop_neighbor_imgs(neighbor_patch_dataroot)
            if feature_type in ['target', 'all']:
                print("Extracting target features...")
                ok = extract_features_single(
                    wsi_dataroot=self.wsi_dataroot,
                    patch_dataroot=f"{self.patch_source_dir}/patches",
                    embed_dataroot=f"{self.asset_dir}/emb/target",
                    slide_ext=self.config['slide_ext'],
                    patch_encoder=self.config['patch_encoder'],
                    num_tiles=self.config['num_t'],
                    feature_type='target',
                    batch_size=self.config['batch_size'],
                    num_workers=self.config['num_workers'],
                    overwrite=self.config['overwrite'],
                    gpus=self.config['gpus'],
                    transform_type=transform_type,
                    id_path=f"{self.metadata_dir}/ids.csv",
                    mode=self.mode,
                )
                self._raise_if_extraction_failed(ok, "target")

    @staticmethod
    def _raise_if_extraction_failed(ok: bool, feature_type: str) -> None:
        if not ok:
            raise RuntimeError(f"Feature extraction failed for feature_type={feature_type}.")

    def _reuse_existing_embeddings(self, feature_types: list) -> None:
        """Symlink already-extracted embeddings from an asset-dir predict()
        target's own source directory into asset_dir before extraction
        runs, so the "already exists, skip" check every extract_features_*
        call already has finds and reuses them.

        Only patches/ got this treatment before (via a directory-level
        symlink in STPred._materialize_wsi_data_config) -- emb/ was never
        checked at all, so a predict() call against an asset dir that
        already had every embedding it needed still re-extracted all of
        them from scratch, every single call. For a neighbor-feature model
        on a real cohort that's hours of avoidable GPU time.

        This links individual <sample>.h5 files rather than the whole emb/
        tree, unlike patches/ -- asset_dir must stay writable so a model
        needing an embedding NOT already present in the source can still
        have it extracted fresh right alongside the reused ones.

        extract_features_single/_parallel write via `h5py.File(path,
        mode='w')`, which for a path that is a symlink follows it and
        truncates the file it points to -- confirmed empirically, not
        assumed. Two consequences of that, both handled below:

        - When `overwrite=True`, never leave (or create) a symlink at any
          destination this function manages -- extraction will open every
          one of them in 'w' mode regardless of whether it's "supposed" to
          be reused, so a symlink left over from an earlier, non-overwrite
          call against the same asset_dir would have its real target (in
          the read-only source dir) silently truncated the moment
          extraction reached that sample.
        - When reusing (not overwrite), refresh a destination whose
          symlink target no longer matches the current source instead of
          leaving it alone -- otherwise a second predict() call reusing
          the same `asset_dir`/output_dir against a different source (or
          after the original source's embeddings were regenerated) keeps
          serving a stale embedding from the first call's source, and
          extraction's own "already exists" check has no way to notice."""
        if self.mode != 'inference' or self.config.get('extract_from_wsi'):
            return  # only the asset-dir predict() case has a separate, read-only source dir
        if os.path.abspath(self.patch_source_dir) == os.path.abspath(self.asset_dir):
            return  # nothing separate to reuse from
        model_name = self.config['patch_encoder']
        sample_ids = self._get_sample_ids()
        overwrite = bool(self.config.get('overwrite'))
        for feature in feature_types:
            src_dir = os.path.join(self.patch_source_dir, 'emb', feature, f'features_{model_name}')
            dst_dir = os.path.join(self.asset_dir, 'emb', feature, f'features_{model_name}')
            if not os.path.isdir(src_dir) and not os.path.isdir(dst_dir):
                continue
            os.makedirs(dst_dir, exist_ok=True)
            for sample_id in sample_ids:
                src = os.path.join(src_dir, f'{sample_id}.h5')
                dst = os.path.join(dst_dir, f'{sample_id}.h5')
                if overwrite:
                    if os.path.islink(dst):
                        os.remove(dst)
                    continue
                if not os.path.isfile(src):
                    continue
                if os.path.islink(dst):
                    if os.readlink(dst) == src:
                        continue  # already correctly linked
                    os.remove(dst)  # stale -- pointed at a different/older source
                elif os.path.lexists(dst):
                    continue  # a real (non-symlink) file already sits here -- never touch it
                os.symlink(src, dst)

    def _drop_neighbor_imgs(self, patch_dir: str):
        """Delete img dataset from neighbor h5 files after feature extraction.

        Refuses to touch anything outside asset_dir (the location this
        pipeline instance actually owns/writes into). For an asset-dir
        predict() target, `patch_dir` can legitimately be the *read-only*
        source directory's own patches/neighbor/ (see the neighbor_patch_
        dataroot fix in run_extraction() -- that's the CORRECT place to
        read neighbor patches FROM for that case) -- but this cleanup step
        must never delete data there, or it silently corrupts a shared/
        read-only asset directory for every other consumer of it, the
        same class of bug the dataroot fix above addresses."""
        asset_root = os.path.abspath(self.asset_dir)
        if os.path.commonpath([asset_root, os.path.abspath(patch_dir)]) != asset_root:
            return
        print("Removing neighbor patch images to free disk space...")
        for h5_path in glob(f"{patch_dir}/*.h5"):
            try:
                with h5py.File(h5_path, 'a') as f:
                    if 'img' in f:
                        del f['img']
            except PermissionError:
                # Best-effort disk-space cleanup: some shared patch files
                # are root-owned with no group write bit, so this can't
                # always succeed. Failing to free space on one file must
                # not crash the whole preprocessing run.
                warnings.warn(f"Skipping neighbor-image cleanup for {h5_path}: no write permission.")

    def _get_sample_ids(self):
        """Get list of sample IDs from patches directory."""
        id_file = f"{self.metadata_dir}/ids.csv"
        if os.path.exists(id_file):
            return pd.read_csv(id_file)['sample_id'].tolist()
        patch_files = glob(f"{self.asset_dir}/patches/*.h5")
        return [os.path.splitext(os.path.basename(f))[0] for f in patch_files]

    def _has_processed_data(self) -> bool:
        id_path = f"{self.metadata_dir}/ids.csv"
        if not os.path.isfile(id_path):
            return False
        ids = pd.read_csv(id_path)['sample_id'].dropna().astype(str).tolist()
        if not ids:
            return False
        needs_st = self.mode != "inference"
        has_base = all(
            (
                os.path.isfile(f"{self.asset_dir}/patches/{sample_id}.h5")
                or os.path.isfile(f"{self.asset_dir}/patches/{sample_id}_patches.h5")
            )
            and (
                not needs_st
                or os.path.isfile(f"{self.asset_dir}/st/{sample_id}.h5ad")
                or os.path.isfile(f"{self.asset_dir}/adata/{sample_id}.h5ad")
            )
            for sample_id in ids
        )
        if not has_base:
            return False
        if self.mode != "inference" and _wants_feature(self.config.get("feature_type"), "neighbor"):
            return all(
                os.path.isfile(f"{self.asset_dir}/patches/neighbor/{sample_id}.h5")
                or os.path.isfile(f"{self.asset_dir}/patches/neighbor/{sample_id}_patches.h5")
                for sample_id in ids
            )
        return True

    def _has_genesets(self) -> bool:
        return bool(glob(f"{self.metadata_dir}/*genes.json"))

    def _has_splits(self) -> bool:
        id_path = f"{self.metadata_dir}/ids.csv"
        if not os.path.isfile(id_path):
            return False
        cols = pd.read_csv(id_path, nrows=1).columns
        return any(col.startswith("fold_") for col in cols)

    def run_pipeline(self):
        """Run the complete pipeline."""
        print("Starting ST pipeline...")
        self.preprocess()
        if self.mode in ['raw', 'stpbench']:
            self.prepare_genesets()
            self.split_data()
        self.run_extraction()
        print("Pipeline complete!")


def main():
    parser = argparse.ArgumentParser(description="Data Pipeline")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--mode", type=str, default="raw", choices=["raw", "stpbench", "inference"], help="Pipeline mode")
    parser.add_argument("--extract_from_wsi", action="store_true", help="For mode=inference: extract patches from a raw WSI instead of reusing existing ones")
    parser.add_argument("--platform", type=str, default="visium", help="ST platform (visium, xenium, etc.)")
    parser.add_argument("--slide_ext", type=str, default=".svs", help="Slide file extension")
    parser.add_argument("--patch_size", type=int, default=224, help="Patch size for extraction")
    parser.add_argument("--slide_level", type=int, default=0, help="Slide pyramid level")
    # parser.add_argument("--save_neighbors", action="store_true", help="Save neighbor patches")
    parser.add_argument("--patch_encoder", type=str, default="cigar", help="Model name for feature extraction")
    parser.add_argument("--num_n", type=int, default=5, help="Number of neighbors for feature extraction")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for feature extraction")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--n_splits", type=int, default=4, help="Number of data splits")
    parser.add_argument("--n_top_hvg", type=int, default=50, help="Number of top highly variable genes")
    parser.add_argument("--n_top_heg", type=int, default=1000, help="Number of top highly expressed genes")
    parser.add_argument("--total_gpus", type=int, default=1, help="Total GPUs to use")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results")

    args = parser.parse_args()
    config = vars(args)
    pipeline = DataPipeline(config)
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()
