import os
import sys
import subprocess
from glob import glob
from pathlib import Path
from typing import List, Dict, Union, Optional

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

# Ensure module can import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.utils import run_command


REPO_ROOT = Path(__file__).resolve().parents[3]


def _repo_script(relative_path: str) -> str:
    return str(REPO_ROOT / relative_path)


def preprocess_data(
    input_dir: str,
    output_dir: str,
    meta_dir: str = None,
    mode: str = 'raw',
    platform: str = 'visium',
    slide_ext: str = '.svs',
    patch_size: int = 224,
    slide_level: int = 0,
    save_neighbors: bool = False,
    save_neighbor_imgs: bool = False,
    num_n: int = 5,
    dst_pixel_size: float = 0.5
) -> None:
    """
    Run data preprocessing

    Args:
        input_dir: Input directory containing data
        output_dir: Output directory for processed data
        mode: Processing mode ('raw', 'stpbench', or 'inference')
        platform: ST platform type
        slide_ext: Slide file extension
        patch_size: Size of extracted patches
        slide_level: Slide pyramid level for extraction
        save_neighbors: Whether to save neighbor patches
        num_n: Number of neighbors to extract
        dst_pixel_size: Desired pixel size for patches (in microns)
    """
    cmd = [
        sys.executable, _repo_script("src/preprocess/prepare_data.py"),
        "--input_dir", input_dir,
        "--output_dir", output_dir,
        "--mode", mode,
        "--platform", platform,
        "--slide_ext", slide_ext,
        "--patch_size", str(patch_size),
        "--slide_level", str(slide_level),
        "--num_n", str(num_n),
        "--dst_pixel_size", str(dst_pixel_size)
    ]

    if meta_dir is not None:
        cmd.extend(["--meta_dir", meta_dir])
    
    if save_neighbors:
        cmd.append("--save_neighbors")
    if save_neighbor_imgs:
        cmd.append("--save_neighbor_imgs")

    return_code, output = run_command(cmd, cwd=str(REPO_ROOT))
    
    if return_code != 0:
        print(f"Error occurred during preprocessing: {output}")
        return False
    
    return True


def extract_features_single(
    wsi_dataroot: str,
    patch_dataroot: str,
    embed_dataroot: str,
    slide_ext: str = '.svs',
    patch_encoder: str = 'cigar',
    feature_type: str = 'global',
    num_tiles: int = 1,
    batch_size: int = 1024,
    num_workers: int = 4,
    overwrite: bool = False,
    gpus: List[int] = [0],
    transform_type: str = 'eval',
    id_path: Optional[str] = None,
    mode: str = 'raw'
) -> None:
    """
    Extract image features

    Args:
        wsi_dataroot: Directory containing WSI images
        patch_dataroot: Directory containing patches
        embed_dataroot: Output directory for embeddings
        slide_ext: Slide file extension
        patch_encoder: Model name for feature extraction
        num_tiles: Number of tiles for features to be extracted
        feature_type: Type of features to extract (global or neighbor or target)
        batch_size: Batch size for processing
        num_workers: Number of workers for data loading
        overwrite: Whether to overwrite existing embeddings
        id_path: Optional path to CSV file with sample IDs to process
    """

    cmd = [
        "CUDA_VISIBLE_DEVICES=" + str(gpus[0]),
        sys.executable, _repo_script("src/preprocess/extract_img_features.py"),
        "--wsi_dataroot", wsi_dataroot,
        "--patch_dataroot", patch_dataroot,
        "--embed_dataroot", embed_dataroot,
        "--slide_ext", slide_ext,
        "--patch_encoder", patch_encoder,
        "--num_tiles", str(num_tiles),
        "--feature_type", feature_type,
        "--batch_size", str(batch_size),
        "--num_workers", str(num_workers),
        "--transform_type", transform_type,
        "--mode", mode
    ]
    
    if overwrite:
        cmd.append("--overwrite")
    
    if id_path:
        cmd.extend(["--id_path", id_path])
    
    return_code, output = run_command(cmd, cwd=str(REPO_ROOT))
    
    if return_code != 0:
        print(f"Error occurred during feature extraction: {output}")
        return False
    
    return True


def extract_features_parallel(
    wsi_dataroot: str,
    patch_dataroot: str,
    embed_dataroot: str,
    slide_ext: str = '.svs',
    patch_encoder: str = 'cigar',
    num_tiles: int = 1,
    feature_type: str = 'global',
    batch_size: int = 1024,
    num_workers: int = 4,
    overwrite: bool = False,
    gpus: List[int] = [0,1],
    transform_type: str = 'eval',
    sample_ids: List[str] = [],
    mode: str = 'raw'
) -> None:
    """
    Extract image features

    Args:
        wsi_dataroot: Directory containing WSI images
        patch_dataroot: Directory containing patches
        embed_dataroot: Output directory for embeddings
        slide_ext: Slide file extension
        patch_encoder: Model name for feature extraction
        num_tiles: Number of tiles for features to be extracted
        feature_type: Type of features to extract (global or neighbor or target)
        batch_size: Batch size for processing
        num_workers: Number of workers for data loading
        total_gpus: Total number of GPUs to use
        overwrite: Whether to overwrite existing embeddings
        id_path: Optional path to CSV file with sample IDs to process
    """
    num_gpus = len(gpus)
    
    # Split samples across GPUs
    gpu_samples = {i: [] for i in range(num_gpus)}
    for i, sample_id in enumerate(sample_ids):
        gpu_idx = i % num_gpus
        gpu_samples[gpu_idx].append(sample_id)
        
    processes = []
    for gpu_idx, samples in gpu_samples.items():
        if not samples:
            continue
            
        # Create ID file for this GPU
        id_file = f"{embed_dataroot}/gpu_{gpu_idx}_ids.csv"
        pd.DataFrame({'sample_id': samples}).to_csv(id_file, index=False)
        
        # Prepare command
        cmd = [
            "CUDA_VISIBLE_DEVICES=" + str(gpus[gpu_idx]),
            sys.executable, _repo_script("src/preprocess/extract_img_features.py"),
            "--id_path", id_file,
            "--wsi_dataroot", wsi_dataroot,
            "--patch_dataroot", patch_dataroot,
            "--embed_dataroot", embed_dataroot,
            "--slide_ext", slide_ext,
            "--patch_encoder", patch_encoder,
            "--num_tiles", str(num_tiles),
            "--feature_type", feature_type,
            "--batch_size", str(batch_size),
            "--num_workers", str(num_workers),
            "--transform_type", transform_type,
            "--mode", mode
        ]
        
        if overwrite:
            cmd.append("--overwrite")
        
        # Start process
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpus[gpu_idx])
        process = subprocess.Popen(cmd[1:], env=env, cwd=str(REPO_ROOT))
        processes.append(process)
    
    # Wait for all processes to complete
    for process in processes:
        process.wait()
        
    # Clean up ID files
    for gpu_idx in gpu_samples:
        id_file = f"{embed_dataroot}/gpu_{gpu_idx}_ids.csv"
        if os.path.exists(id_file):
            os.remove(id_file)
    
    return True
