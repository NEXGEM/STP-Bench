"""Data preprocessing pipeline exposed via api namespace."""

import os
import sys
import argparse
import subprocess
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
        self.metadata_dir = self._abs(self.config.get('meta_dir')) or self.output_dir
        self.wsi_dataroot = f"{self.input_dir}/wsis" if mode == 'stpbench' else self.input_dir
        self.dirs = setup_paths(self.output_dir)

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
        if mode in ['raw', 'stpbench'] and self._has_processed_data() and not self.config['overwrite']:
            print(f"Processed data already found at {self.output_dir}. Skipping raw preprocessing.")
        else:
            save_neighbors = _wants_feature(self.config['feature_type'], 'neighbor')
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
            )


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

        gpus = self.config['gpus']
        if self.total_gpus > 1:
            print(f"Running feature extraction in parallel on {self.total_gpus} GPUs...")
            sample_ids = self._get_sample_ids()
            if feature_type in ['global', 'all']:
                print("Extracting global features...")
                ok = extract_features_parallel(
                    wsi_dataroot=self.wsi_dataroot,
                    patch_dataroot=f"{self.asset_dir}/patches",
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
            neighbor_patch_dataroot = f"{self.asset_dir}/patches" if self.config['mode'] == 'inference' else f"{self.asset_dir}/patches/neighbor"
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
                    patch_dataroot=f"{self.asset_dir}/patches",
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
                    patch_dataroot=f"{self.asset_dir}/patches",
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
            neighbor_patch_dataroot = f"{self.asset_dir}/patches" if self.config['mode'] == 'inference' else f"{self.asset_dir}/patches/neighbor"
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
                    patch_dataroot=f"{self.asset_dir}/patches",
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

    def _drop_neighbor_imgs(self, patch_dir: str):
        """Delete img dataset from neighbor h5 files after feature extraction."""
        print("Removing neighbor patch images to free disk space...")
        for h5_path in glob(f"{patch_dir}/*.h5"):
            with h5py.File(h5_path, 'a') as f:
                if 'img' in f:
                    del f['img']

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
        has_base = all(
            (
                os.path.isfile(f"{self.asset_dir}/patches/{sample_id}.h5")
                or os.path.isfile(f"{self.asset_dir}/patches/{sample_id}_patches.h5")
            )
            and (
                os.path.isfile(f"{self.asset_dir}/st/{sample_id}.h5ad")
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
