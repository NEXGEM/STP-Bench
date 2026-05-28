"""Extract multi-resolution (level 1, 2) patches for M2ORT / M2OST.

Generates patches/level{n}/{sample_id}.h5 files aligned to the existing
level-0 patches, which are required by M2OSTDataset.load_img(level=1/2).

Usage (called automatically via extra_preprocess in model config):
    python extract_levels.py
        --data_dir  /path/to/processed_data
        --meta_dir  input/ncche/visium
        --input_dir /path/to/raw_data        # WSI / raw ST source
        --platform  visium
        --mode      raw
        --levels    1,2
        [--overwrite]
"""

import argparse
import os
import sys

import pandas as pd
from tqdm import tqdm

# resolve src/ on the path regardless of working directory
_here = os.path.dirname(os.path.abspath(__file__))
_src = os.path.abspath(os.path.join(_here, "../../../../"))
sys.path.insert(0, _src)
sys.path.insert(0, os.path.join(_src, "src"))

from preprocess.prepare_data import save_patches

# maps pyramid level index → dst_pixel_size (µm/px)
_LEVEL_TO_MPP = {1: 1.0, 2: 2.0, 3: 4.0}


def _read_ids(data_dir, meta_dir):
    for candidate in [
        meta_dir and os.path.join(meta_dir, "ids.csv"),
        os.path.join(data_dir, "ids.csv"),
        os.path.join(data_dir, "patches"),   # derive from existing patches
    ]:
        if not candidate:
            continue
        if os.path.isfile(candidate):
            return pd.read_csv(candidate)["sample_id"].dropna().astype(str).tolist()
    # fallback: scan patches dir
    import glob
    patches = glob.glob(os.path.join(data_dir, "patches", "*.h5"))
    return [os.path.splitext(os.path.basename(p))[0].replace("_patches", "") for p in patches]


def main():
    parser = argparse.ArgumentParser(description="Extract multi-resolution patches for M2ORT/M2OST")
    parser.add_argument("--data_dir",  required=True,  help="Processed data root (contains patches/)")
    parser.add_argument("--meta_dir",  default=None,   help="Directory containing ids.csv")
    parser.add_argument("--asset_dir", default=None,   help="Alias for data_dir (passed by pipeline)")
    parser.add_argument("--input_dir", default=None,   help="Raw data root (WSI / SpaceRanger output per sample)")
    parser.add_argument("--platform",  default="visium", help="ST platform (visium, xenium, hest, …)")
    parser.add_argument("--mode",      default="raw",  choices=["raw", "stpbench", "inference"])
    parser.add_argument("--levels",    default="1,2",  help="Comma-separated pyramid levels to extract (e.g. 1,2)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_dir = args.asset_dir or args.data_dir
    meta_dir = args.meta_dir and os.path.abspath(args.meta_dir)

    levels = [int(l.strip()) for l in args.levels.split(",") if l.strip()]
    unknown = [l for l in levels if l not in _LEVEL_TO_MPP]
    if unknown:
        parser.error(f"Unsupported levels: {unknown}. Supported: {list(_LEVEL_TO_MPP)}")

    if not args.input_dir:
        print(
            "[extract_levels] WARNING: --input_dir not provided. "
            "Multi-resolution patches cannot be extracted without raw WSI data. "
            "Skipping."
        )
        return

    sample_ids = _read_ids(data_dir, meta_dir)
    if not sample_ids:
        print(f"[extract_levels] No sample IDs found under {data_dir}. Skipping.")
        return

    platform = "hest" if args.mode == "stpbench" else args.platform

    for level in levels:
        dst_pixel_size = _LEVEL_TO_MPP[level]
        level_dir = os.path.join(data_dir, "patches", f"level{level}")
        print(f"\n[extract_levels] Level {level} (mpp={dst_pixel_size}) → {level_dir}")
        os.makedirs(level_dir, exist_ok=True)

        for name in tqdm(sample_ids, desc=f"level{level}"):
            out_path = os.path.join(level_dir, f"{name}.h5")
            if os.path.isfile(out_path) and not args.overwrite:
                continue
            save_patches(
                name=name,
                input_dir=args.input_dir,
                output_dir=data_dir,
                platform=platform,
                save_targets=True,
                save_neighbors=False,
                dst_pixel_size=dst_pixel_size,
            )

    print("\n[extract_levels] Done.")


if __name__ == "__main__":
    main()
