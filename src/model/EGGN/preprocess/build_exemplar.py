import os
from tqdm import tqdm

import argparse
import h5py
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import torch


def _read_ids(meta_dir, phase, fold):
    ids_path = os.path.join(meta_dir, "ids.csv")
    ids = pd.read_csv(ids_path)
    fold_col = f"fold_{fold}"
    if fold_col not in ids.columns:
        raise FileNotFoundError(f"{fold_col} is missing from {ids_path}")
    return ids.loc[ids[fold_col].astype(str).str.lower() == phase].reset_index(drop=True)


def _num_folds(meta_dir):
    ids_path = os.path.join(meta_dir, "ids.csv")
    ids = pd.read_csv(ids_path, nrows=1)
    fold_cols = [col for col in ids.columns if col.startswith("fold_")]
    if not fold_cols:
        raise FileNotFoundError(f"No fold_* columns found in {ids_path}")
    return len(fold_cols)


def main(
    data_dir,
    meta_dir=None,
    distance_metric='l1',
    external_dir=None,
    model_name='uni_v2',
    fold_idx=None,
    asset_dir=None,
    external_asset_dir=None,
    external_meta_dir=None,
    overwrite=False,
):
    meta_dir = meta_dir or data_dir
    asset_dir = asset_dir or data_dir
    external_asset_dir = external_asset_dir or external_dir
    external_meta_dir = external_meta_dir or external_dir
    emb_dir = f"{asset_dir}/emb/global/features_{model_name}"
    num_fold = _num_folds(meta_dir)

    if external_dir is not None:
        # Namespace by the training run's own meta_dir, matching EGNDataset's
        # own `ref_data` derivation (ref_data_dir.split('/')[-2:]) — deriving
        # this from data_dir instead breaks whenever data_dir is a root
        # shared across every dataset (the documented convention), since its
        # last two path components no longer identify the training dataset.
        train_data = '/'.join(meta_dir.replace('/bench_data', '').split('/')[-2:])
        ext_emb_dir = f"{external_asset_dir}/emb/global/features_{model_name}"
        save_dir = f"{external_asset_dir}/exemplar/{model_name}/{distance_metric}/{train_data}"
    else:
        ext_emb_dir = None
        save_dir = f"{asset_dir}/exemplar/{model_name}/{distance_metric}"

    output = []
    for fold in range(num_fold):
        print(f"Processing fold {fold}...")

        if fold_idx is not None and fold != fold_idx:
            continue

        train_dataset = _read_ids(meta_dir, "train", fold)
        if external_dir is None:
            test_dataset = _read_ids(meta_dir, "test", fold)
        else:
            test_dataset = pd.read_csv(os.path.join(external_meta_dir, "ids.csv"))

        train_names = train_dataset["sample_id"].tolist()
        train_embs = []
        name_list = []
        flag_list = []
        sid_list = []
        for name in train_names:
            h5_path = os.path.join(emb_dir, f"{name}.h5")
            with h5py.File(h5_path, "r") as f:
                img_emb = f['features'][:]

            train_embs.append(img_emb)

            name_ = np.repeat(name, img_emb.shape[0])
            flag_ = np.repeat('train', img_emb.shape[0])
            sid = np.arange(img_emb.shape[0])

            name_list.append(name_)
            flag_list.append(flag_)
            sid_list.append(sid)

        train_embs = np.concatenate(train_embs, axis=0)
        print(f"Loaded {train_embs.shape[0]} `image embeddings` for training.")

        test_names = test_dataset["sample_id"].tolist()
        test_embs = []
        for name in test_names:
            if external_dir is None:
                h5_path = os.path.join(emb_dir, f"{name}.h5")
            else:
                h5_path = os.path.join(ext_emb_dir, f"{name}.h5")

            with h5py.File(h5_path, "r") as f:
                if 'features' in f:
                    img_emb = f['features'][:]
                elif 'embeddings' in f:
                    img_emb = f['embeddings'][:]
                else:
                    raise ValueError(f"Cannot find 'features' or 'embeddings' in {h5_path}")

            test_embs.append(img_emb)

            name_ = np.repeat(name, img_emb.shape[0])
            flag_ = np.repeat('test', img_emb.shape[0])
            sid = np.arange(img_emb.shape[0])

            name_list.append(name_)
            flag_list.append(flag_)
            sid_list.append(sid)

        test_embs = np.concatenate(test_embs, axis=0)
        print(f"Loaded {test_embs.shape[0]} `image embeddings` for testing.")

        embs = np.concatenate((train_embs, test_embs), axis=0)
        names = np.concatenate(name_list, axis=0)
        flags = np.concatenate(flag_list, axis=0)
        sids = np.concatenate(sid_list, axis=0)

        unique_name = np.unique(names)
        result = {}
        for n in tqdm(unique_name):

            idx_query = names == n
            flag_ = flags[idx_query][0]

            if external_dir is not None and flag_ == 'train':
                continue

            save_path = f"{save_dir}/fold{fold}/{flag_}"
            os.makedirs(save_path, exist_ok=True)

            save_name = f"{save_path}/{n}.h5"
            if os.path.exists(save_name) and not overwrite:
                print(f"{save_name} exists, skip!")
                continue

            idx_key = (~idx_query) & (flags == 'train')
            if idx_key.sum() == 0:
                idx_key = (flags == 'train')

            emb_query = torch.Tensor(embs[idx_query])
            emb_key = torch.Tensor(embs[idx_key])
            p = 1 if distance_metric == 'l1' else 2

            try:
                dist = torch.cdist(emb_query.cuda(), emb_key.cuda(), p=p)
                topk = min(dist.shape[1], 100)
            except RuntimeError as e:
                print(f"GPU computation failed, falling back to CPU: {e}")
                dist = torch.cdist(emb_query, emb_key, p=p)
                topk = min(dist.shape[1], 100)

            knn = dist.topk(topk, dim=1, largest=False)
            knn_indices = knn.indices.cpu().numpy()
            knn_values = knn.values.cpu().numpy()

            ex_sid = []
            ex_name = []
            for i in range(knn_indices.shape[0]):
                indices = knn_indices[i]
                name_ = names[idx_key][indices]
                sid_ = sids[idx_key][indices]

                ex_sid.append(sid_)
                ex_name.append(name_)

            ex_sid = np.stack(ex_sid, axis=0)
            ex_name = np.stack(ex_name, axis=0)

            max_length = max(len(value) for value in unique_name)

            with h5py.File(save_name, "w") as f:
                f.create_dataset("sid", data=ex_sid)
                f.create_dataset("pid", data=ex_name.astype(f'S{max_length}'))

            result[n] = (ex_sid, ex_name)

        output.append(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build exemplar dataset")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--meta_dir", type=str, default=None)
    parser.add_argument("--distance_metric", type=str, default="l1", choices=["l1", "l2"])
    parser.add_argument("--external_dir", type=str, default=None)
    parser.add_argument("--asset_dir", type=str, default=None)
    parser.add_argument("--external_asset_dir", type=str, default=None)
    parser.add_argument("--external_meta_dir", type=str, default=None)
    parser.add_argument("--model_name", type=str, default='uni_v2')
    parser.add_argument("--fold_idx", type=int, default=None)
    parser.add_argument("--overwrite", action='store_true', default=False)

    args = parser.parse_args()
    main(
        args.data_dir,
        meta_dir=args.meta_dir,
        distance_metric=args.distance_metric,
        external_dir=args.external_dir,
        model_name=args.model_name,
        fold_idx=args.fold_idx,
        asset_dir=args.asset_dir,
        external_asset_dir=args.external_asset_dir,
        external_meta_dir=args.external_meta_dir,
        overwrite=args.overwrite,
    )
