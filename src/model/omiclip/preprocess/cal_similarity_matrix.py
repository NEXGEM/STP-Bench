import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import scanpy as sc
import argparse

from typing import Union
from scipy.sparse import issparse
import torch

try:
    from loki.utils import load_model, encode_images_from_h5, encode_text_df
    from loki.preprocess import generate_gene_df
except ModuleNotFoundError as exc:
    if exc.name == "open_clip":
        raise ModuleNotFoundError(
            "OmiCLIP preprocessing requires open_clip_torch. "
            "Install it with `pip install open_clip_torch==2.26.1` in the STPBench environment."
        ) from exc
    raise

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def encode_text_one_slide(
    model: torch.nn.Module,
    tokenizer: callable,
    st_path: str,
    device: Union[str, torch.device],
    batch_size: int = 32,
):
    ad = sc.read_h5ad(st_path)
    house_keeping_genes = pd.read_csv(
        os.path.join(_SCRIPT_DIR, 'housekeeping_genes.csv'), index_col=0
    )
    top_k_genes_str = generate_gene_df(ad, house_keeping_genes, todense=issparse(ad.X))
    if len(top_k_genes_str) <= batch_size:
        return encode_text_df(model, tokenizer, top_k_genes_str, 'label', device)

    embeddings = []
    for start in range(0, len(top_k_genes_str), batch_size):
        end = min(start + batch_size, len(top_k_genes_str))
        embeddings.append(
            encode_text_df(model, tokenizer, top_k_genes_str.iloc[start:end], 'label', device).detach().cpu()
        )
        if torch.cuda.is_available() and str(device).startswith('cuda'):
            torch.cuda.empty_cache()
    return torch.cat(embeddings, dim=0).to(device)


def _process_fold(
    model,
    preprocess,
    tokenizer,
    data_dir: str,
    meta_dir: str,
    asset_dir: str,
    external_dir: str,
    external_meta_dir: str,
    external_asset_dir: str,
    fold: int,
    device: str,
    overwrite: bool,
    text_batch_size: int,
):
    if external_dir is not None:
        query_asset_dir = external_asset_dir
    else:
        query_asset_dir = asset_dir

    query_meta_dir = meta_dir if external_dir is None else (external_meta_dir or external_dir)
    if external_dir is not None:
        # External evaluation uses ALL of the external dataset's samples for
        # every trained fold's checkpoint (STDataset does the same — see
        # ref_data_dir-driven unfiltered loading in base_dataset.py), not a
        # fold-specific test split of the external data's own (irrelevant)
        # CV folds.
        query_samples = pd.read_csv(os.path.join(query_meta_dir, 'ids.csv')).sample_id.tolist()
    else:
        query_samples = _read_ids(query_meta_dir, 'test', fold).sample_id.tolist()

    if external_dir is not None:
        # Namespace by the training run's own meta_dir, matching
        # EGNDataset's `ref_data` derivation and OmiCLIP.py's forward(),
        # which already looks up similarity_matrix/{ref_data_dir}/fold{N}
        # first (falling back to the flat path) — write to the namespaced
        # location it actually expects, so a similarity matrix built here
        # (using the EXTERNAL data as query and the TRAINING data's text
        # embeddings as key) can never collide with a flat-path file built
        # for `data_dir` used as a standalone/primary dataset.
        train_data = '/'.join(meta_dir.replace('/bench_data', '').split('/')[-2:])
        save_path = os.path.join(query_asset_dir, 'similarity_matrix', train_data, f'fold{fold}')
    else:
        save_path = os.path.join(query_asset_dir, 'similarity_matrix', f'fold{fold}')

    if not overwrite:
        query_samples = [
            s for s in query_samples
            if not os.path.exists(os.path.join(save_path, f'{s}.npy'))
        ]
    if not query_samples:
        print(f"All similarity matrices for fold {fold} already exist. Skipping...")
        return

    key_samples = _read_ids(meta_dir, 'train', fold).sample_id.tolist()

    print("Encoding text of key samples...")
    key_st_embeddings = torch.cat([
        encode_text_one_slide(
            model, tokenizer,
            _st_path(asset_dir, s),
            device,
            batch_size=text_batch_size,
        ).detach().cpu()
        for s in tqdm(key_samples)
    ], dim=0)

    print("Encoding images of query samples...")
    os.makedirs(save_path, exist_ok=True)
    for sample in query_samples:
        query_img_embedding = encode_images_from_h5(
            model, preprocess,
            _patch_path(query_asset_dir, sample),
            device,
        ).detach().cpu()
        dot_similarity = query_img_embedding @ key_st_embeddings.T
        out_path = os.path.join(save_path, f'{sample}.npy')
        print(f'Saving similarity matrix of {sample} to {out_path}')
        np.save(out_path, dot_similarity.numpy())


def main(
    data_dir: str,
    meta_dir: str = None,
    asset_dir: str = None,
    external_dir: str = None,
    external_meta_dir: str = None,
    external_asset_dir: str = None,
    model_path: str = None,
    device: str = 'cuda',
    fold: int = None,
    overwrite: bool = False,
    text_batch_size: int = 32,
):
    if not model_path:
        raise ValueError(
            "OmiCLIP preprocessing requires --model_path pointing to a pretrained OmiCLIP/open_clip checkpoint. "
            "Set DATA.model_path in the data config or pass model_path through the model preprocess config."
        )
    meta_dir = meta_dir or data_dir
    asset_dir = asset_dir or data_dir
    external_meta_dir = external_meta_dir or external_dir
    external_asset_dir = external_asset_dir or external_dir
    device = _resolve_device(device)

    model, preprocess, tokenizer = load_model(model_path, device)

    folds = range(_num_folds(meta_dir)) if fold is None else [fold]

    for f in folds:
        print(f"Processing fold: {f}")
        _process_fold(
            model, preprocess, tokenizer,
            data_dir, meta_dir, asset_dir, external_dir, external_meta_dir, external_asset_dir,
            f, device, overwrite, text_batch_size,
        )


def _read_ids(meta_dir, phase, fold):
    split_path = os.path.join(meta_dir, 'splits', f'{phase}_{fold}.csv')
    if os.path.isfile(split_path):
        return pd.read_csv(split_path)

    ids_path = os.path.join(meta_dir, 'ids.csv')
    ids = pd.read_csv(ids_path)
    fold_col = f'fold_{fold}'
    if phase in {'train', 'test'} and fold_col in ids.columns:
        return ids.loc[ids[fold_col].astype(str).str.lower() == phase].reset_index(drop=True)
    if phase == 'test' and 'sample_id' in ids.columns:
        return ids
    raise FileNotFoundError(f"{split_path} not found and {fold_col} is missing from {ids_path}")


def _num_folds(meta_dir):
    split_dir = os.path.join(meta_dir, 'splits')
    if os.path.isdir(split_dir):
        return len([name for name in os.listdir(split_dir) if name.startswith('train_') and name.endswith('.csv')])

    ids_path = os.path.join(meta_dir, 'ids.csv')
    ids = pd.read_csv(ids_path, nrows=1)
    fold_cols = [col for col in ids.columns if col.startswith('fold_')]
    if not fold_cols:
        raise FileNotFoundError(f"No splits directory or fold_* columns found in {meta_dir}")
    return len(fold_cols)


def _resolve_device(device):
    if device is None:
        device = 'cuda'
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError(
            "OmiCLIP preprocessing requires a CUDA GPU, but CUDA is not available in this process. "
            "Check GPU allocation, CUDA_VISIBLE_DEVICES, and whether the notebook/kernel environment can access the GPU."
        )
    return device


def _st_path(asset_dir, sample):
    for subdir in ('st', 'adata'):
        path = os.path.join(asset_dir, subdir, f'{sample}.h5ad')
        if os.path.isfile(path):
            return path
    return os.path.join(asset_dir, 'st', f'{sample}.h5ad')


def _patch_path(asset_dir, sample):
    path = os.path.join(asset_dir, 'patches', f'{sample}.h5')
    if os.path.isfile(path):
        return path
    return os.path.join(asset_dir, 'patches', f'{sample}_patches.h5')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--meta_dir', type=str, default=None)
    parser.add_argument('--asset_dir', type=str, default=None)
    parser.add_argument('--external_dir', type=str, default=None)
    parser.add_argument('--external_meta_dir', type=str, default=None)
    parser.add_argument('--external_asset_dir', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--fold', type=int, default=None)
    parser.add_argument('--text_batch_size', type=int, default=32)
    parser.add_argument('--overwrite', action='store_true', default=False)
    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        meta_dir=args.meta_dir,
        asset_dir=args.asset_dir,
        external_dir=args.external_dir,
        external_meta_dir=args.external_meta_dir,
        external_asset_dir=args.external_asset_dir,
        model_path=args.model_path,
        device=args.device,
        fold=args.fold,
        overwrite=args.overwrite,
        text_batch_size=args.text_batch_size,
    )
