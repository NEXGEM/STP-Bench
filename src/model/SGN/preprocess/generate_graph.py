
from tqdm import tqdm
import numpy as np
import torch
import os
from collections import namedtuple
import torch_geometric
import argparse
import pandas as pd
import sys
from torch_geometric.nn import knn_graph

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.insert(0, src_path)

from src.dataset.egn import EGNDataset


def _num_folds(meta_dir):
    ids_path = os.path.join(meta_dir, "ids.csv")
    ids = pd.read_csv(ids_path, nrows=1)
    fold_cols = [col for col in ids.columns if col.startswith("fold_")]
    if not fold_cols:
        raise FileNotFoundError(f"No fold_* columns found in {ids_path}")
    return len(fold_cols)


def _read_ids(meta_dir, phase, fold):
    ids_path = os.path.join(meta_dir, "ids.csv")
    ids = pd.read_csv(ids_path)
    fold_col = f"fold_{fold}"
    if fold_col not in ids.columns:
        raise FileNotFoundError(f"{fold_col} is missing from {ids_path}")
    return ids.loc[ids[fold_col].astype(str).str.lower() == phase].reset_index(drop=True)


def get_edge(x):

     edge_index = torch_geometric.nn.radius_graph(
            x,
            np.sqrt(2),
            None,
            False,
            max_num_neighbors=5,
            flow="source_to_target",
            num_workers=1,
        )
     return edge_index
 
def get_knn_edge(x,num_edge=5):
    edge_index = torch_geometric.nn.knn_graph(
        x,
        k=num_edge,
        )
    return edge_index


def main(
    data_dir,
    external_dir=None,
    model_name='uni_v2',
    gene_type='hmhvg',
    num_genes=200,
    cpm=False,
    overwrite=False,
    fold_idx=None,
    asset_dir=None,
    external_asset_dir=None,
    num_edge=5,
    meta_dir=None,
):
    meta_dir = meta_dir or data_dir
    asset_dir = asset_dir or data_dir
    external_asset_dir = external_asset_dir or external_dir
    num_fold = _num_folds(meta_dir)

    for fold in range(num_fold):
        if fold_idx is not None and fold != fold_idx:
            continue
        print(f"Processing fold {fold}...")

        for phase in ["train", "test"]:

            if external_dir is None:
                dataset = _read_ids(meta_dir, phase, fold)
                
                if cpm:
                    savename = f"{asset_dir}/SGN/cpm/{model_name}/fold{fold}/{phase}"
                else:
                    savename = f"{asset_dir}/SGN/{model_name}/fold{fold}/{phase}"
                os.makedirs(savename,exist_ok=True)
                
                ref_data_dir = None
            else:
                if phase == "train":
                    continue  # skip training phase for exemplar generation
                
                test_split = os.path.join(external_dir, "ids.csv")
                dataset = pd.read_csv(test_split)
                
                train_data = '/'.join(data_dir.replace('/bench_data', '').split('/')[-2:])

                if cpm:
                    savename = f"{external_asset_dir}/SGN/cpm/{model_name}/{train_data}/fold{fold}/{phase}"
                else:
                    savename = f"{external_asset_dir}/SGN/{model_name}/{train_data}/fold{fold}/{phase}"
                os.makedirs(savename, exist_ok=True)
                
                ref_data_dir = data_dir
            
            foldername = f"{savename}/graph"
            os.makedirs(foldername, exist_ok=True) 
                        
            # Load dataset
            dataset = EGNDataset(
                mode='cv',
                phase=phase,
                fold=fold,
                data_dir=data_dir if external_dir is None else external_dir,
                asset_dir=asset_dir if external_dir is None else external_asset_dir,
                distance_metric='l1',
                gene_type=gene_type,
                num_genes=num_genes,
                num_outputs=num_genes,
                normalize=True,
                cpm=cpm,
                model_name=model_name,
                ref_data_dir=ref_data_dir,
                ref_asset_dir=asset_dir if ref_data_dir is not None else None,
                load_level='slide' 
            )
            
            loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1
            )
            
            # img_data = []
            for i, x in tqdm(enumerate(loader)):
                sample_id = dataset.int2id[i]
                save_name = f"{foldername}/{sample_id}.pt"
                if os.path.exists(save_name) and not overwrite:
                    print(f"{save_name} exists, skip!")
                    continue

                pos, p, py, _, _ = x["pos"], x["ei"], x["label"], x["ej"], x["yj"]
                img_data = [pos, p.squeeze(-2), py]
                img_data = [data.squeeze(0) for data in img_data]
                
                window_edge = get_edge((img_data[0]).clone())
                
                knn_edge = get_knn_edge(img_data[1].clone(), num_edge)
                
                data = torch_geometric.data.HeteroData()
            
                data["window"].pos = img_data[0].clone()
                data["window"].x = img_data[1].clone()
                data["window"].y = img_data[2].clone()
                
                assert len(data["window"]["pos"]) == len(data["window"]["x"]) == len(data["window"]["y"])
                
                data['window', 'near', 'window'].edge_index = window_edge
                data['window', 'knn', 'window'].edge_index = knn_edge

                torch.save(data, save_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build exemplar dataset")
    parser.add_argument("--data_dir", type=str, default="input/takano/xenium", help="Path to the data directory")
    parser.add_argument("--num_edge", type=int, default=5)
    parser.add_argument("--external_dir", type=str, default=None, help="Path to the data directory")
    parser.add_argument("--asset_dir", type=str, default=None, help="Path to patches, embeddings, ST files, and generated assets")
    parser.add_argument("--external_asset_dir", type=str, default=None, help="Asset directory for external data")
    parser.add_argument("--model_name", type=str, default='uni_v2', help="Path to the data directory")
    parser.add_argument("--gene_type", type=str, default='hmhvg', help="Type of genes to use")
    parser.add_argument("--num_genes", type=int, default=200, help="Number of genes to use")
    parser.add_argument("--cpm", action='store_true', default=False, help="Whether to use CPM normalization")
    parser.add_argument("--overwrite", action='store_true', default=False, help="Whether to overwrite existing files")
    parser.add_argument("--fold_idx", type=int, default=None, help="If specified, only process this fold")
    parser.add_argument("--meta_dir", type=str, default=None, help="Path to the meta directory containing ids.csv with fold columns")

    args = parser.parse_args()
    data_dir = args.data_dir
    external_dir = args.external_dir
    model_name = args.model_name
    gene_type = args.gene_type
    num_genes = args.num_genes
    cpm = args.cpm
    overwrite = args.overwrite
    fold_idx = args.fold_idx if args.fold_idx is not None else None

    main(
        data_dir,
        external_dir,
        model_name,
        gene_type,
        num_genes,
        cpm,
        overwrite,
        fold_idx,
        args.asset_dir,
        args.external_asset_dir,
        args.num_edge,
        args.meta_dir,
    )
