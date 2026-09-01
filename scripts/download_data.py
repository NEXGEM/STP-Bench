#!/usr/bin/env python
"""Download STP-Bench data from Hugging Face — full dataset, by dataset, or by sample.

Examples:
    # Full dataset (~444 GB)
    python scripts/download_data.py --local_dir /path/to/stp_bench

    # One or more datasets (namespace/name, matches config/data/<namespace>/<name>.yaml)
    python scripts/download_data.py --local_dir /path/to/stp_bench --dataset ncche/xenium

    # One or more individual samples
    python scripts/download_data.py --local_dir /path/to/stp_bench --sample SNU16A --sample SNU16B

    # --dataset and --sample can be combined and repeated freely.
"""
import argparse
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

REPO_ID = "nexgem/STP-Bench"
REPO_ROOT = Path(__file__).resolve().parent.parent
FILE_TEMPLATES = (
    "patches/{sid}.h5",
    "st/{sid}.h5ad",
    "metadata/{sid}.json",
    "wsis/{sid}.tif",
    "wsis/{sid}.tiff",
)


def sample_ids_for(dataset: str) -> list:
    """Read the sample_id column from this repo's own input/<namespace>/<name>/ids.csv."""
    namespace, _, name = dataset.partition("/")
    ids_path = REPO_ROOT / "input" / namespace / name / "ids.csv"
    if not ids_path.is_file():
        raise FileNotFoundError(
            f"No ids.csv found for dataset {dataset!r} at {ids_path} "
            "-- expected format: <namespace>/<name>, e.g. ncche/xenium"
        )
    return pd.read_csv(ids_path)["sample_id"].astype(str).tolist()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--local_dir", required=True, help="Directory to download into")
    parser.add_argument(
        "--dataset", action="append", default=[], metavar="NAMESPACE/NAME",
        help="Download only this dataset's samples (repeatable), e.g. ncche/xenium",
    )
    parser.add_argument(
        "--sample", action="append", default=[], metavar="SAMPLE_ID",
        help="Download only this sample (repeatable)",
    )
    parser.add_argument("--repo_id", default=REPO_ID, help=f"Default: {REPO_ID}")
    args = parser.parse_args()

    sample_ids = list(args.sample)
    for dataset in args.dataset:
        sample_ids.extend(sample_ids_for(dataset))

    allow_patterns = None
    if sample_ids:
        sample_ids = sorted(set(sample_ids))
        allow_patterns = [tpl.format(sid=sid) for sid in sample_ids for tpl in FILE_TEMPLATES]
        print(f"Downloading {len(sample_ids)} sample(s): {', '.join(sample_ids)}")
    else:
        print("No --dataset/--sample given -- downloading the full dataset (~444 GB).")

    local_dir = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=args.local_dir,
        allow_patterns=allow_patterns,
    )
    print(f"Downloaded to {local_dir}")


if __name__ == "__main__":
    main()
