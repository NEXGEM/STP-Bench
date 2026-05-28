import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import scanpy as sc
import argparse

from typing import Union
from scipy.sparse import issparse
import torch

from loki.utils import load_model, encode_images_from_h5, encode_text_df
from loki.preprocess import generate_gene_df

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def encode_text_one_slide(
    model: torch.nn.Module,
    tokenizer: callable,
    st_path: str,
    device: Union[str, torch.device],
):
    ad = sc.read_h5ad(st_path)
    house_keeping_genes = pd.read_csv(
        os.path.join(_SCRIPT_DIR, 'housekeeping_genes.csv'), index_col=0
    )
    top_k_genes_str = generate_gene_df(ad, house_keeping_genes, todense=issparse(ad.X))
    return encode_text_df(model, tokenizer, top_k_genes_str, 'label', device)


def _process_fold(
    model,
    preprocess,
    tokenizer,
    data_dir: str,
    asset_dir: str,
    external_dir: str,
    external_asset_dir: str,
    fold: int,
    device: str,
    overwrite: bool,
):
    split_path = os.path.join(data_dir, 'splits')

    if external_dir is not None:
        query_path = os.path.join(external_dir, 'ids.csv')
        query_asset_dir = external_asset_dir
    else:
        query_path = os.path.join(split_path, f'test_{fold}.csv')
        query_asset_dir = asset_dir

    query_samples = pd.read_csv(query_path).sample_id.tolist()
    save_path = os.path.join(query_asset_dir, 'similarity_matrix', f'fold{fold}')

    if not overwrite:
        query_samples = [
            s for s in query_samples
            if not os.path.exists(os.path.join(save_path, f'{s}.npy'))
        ]
    if not query_samples:
        print(f"All similarity matrices for fold {fold} already exist. Skipping...")
        return

    key_path = os.path.join(split_path, f'train_{fold}.csv')
    key_samples = pd.read_csv(key_path).sample_id.tolist()

    print("Encoding text of key samples...")
    key_st_embeddings = torch.cat([
        encode_text_one_slide(
            model, tokenizer,
            os.path.join(data_dir, 'adata', f'{s}.h5ad'),
            device,
        ).detach().cpu()
        for s in tqdm(key_samples)
    ], dim=0)

    print("Encoding images of query samples...")
    os.makedirs(save_path, exist_ok=True)
    for sample in query_samples:
        query_img_embedding = encode_images_from_h5(
            model, preprocess,
            os.path.join(query_asset_dir, 'patches', f'{sample}.h5'),
            device,
        ).detach().cpu()
        dot_similarity = query_img_embedding @ key_st_embeddings.T
        out_path = os.path.join(save_path, f'{sample}.npy')
        print(f'Saving similarity matrix of {sample} to {out_path}')
        np.save(out_path, dot_similarity.numpy())


def main(
    data_dir: str,
    asset_dir: str = None,
    external_dir: str = None,
    external_asset_dir: str = None,
    model_path: str = None,
    device: str = 'cuda',
    fold: int = None,
    overwrite: bool = False,
):
    asset_dir = asset_dir or data_dir
    external_asset_dir = external_asset_dir or external_dir

    model, preprocess, tokenizer = load_model(model_path, device)

    split_path = os.path.join(data_dir, 'splits')
    folds = range(len(os.listdir(split_path)) // 2) if fold is None else [fold]

    for f in folds:
        print(f"Processing fold: {f}")
        _process_fold(
            model, preprocess, tokenizer,
            data_dir, asset_dir, external_dir, external_asset_dir,
            f, device, overwrite,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--asset_dir', type=str, default=None)
    parser.add_argument('--external_dir', type=str, default=None)
    parser.add_argument('--external_asset_dir', type=str, default=None)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--fold', type=int, default=None)
    parser.add_argument('--overwrite', action='store_true', default=False)
    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        asset_dir=args.asset_dir,
        external_dir=args.external_dir,
        external_asset_dir=args.external_asset_dir,
        model_path=args.model_path,
        device=args.device,
        fold=args.fold,
        overwrite=args.overwrite,
    )
