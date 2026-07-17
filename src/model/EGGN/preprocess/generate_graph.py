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


def _external_genes_override(meta_dir, asset_dir, external_meta_dir, external_asset_dir, gene_type, num_genes):
    """Restrict the training gene panel to genes actually measured in the
    external dataset, mirroring src/api/stpbench.py's _external_gene_overlap.

    This script runs as a standalone subprocess (invoked via extra_preprocess)
    and never goes through STPBench's runtime-config orchestration, so it
    can't just read cfg.DATA.genes_override — it has to recompute the overlap
    itself. Returns None when the full training panel is available in the
    external data (no restriction needed) or the check can't be performed.
    """
    import json
    train_gene_path = None
    for candidate_dir in dict.fromkeys([meta_dir, asset_dir]):
        if not candidate_dir:
            continue
        path = os.path.join(candidate_dir, f"{gene_type}_{num_genes}genes.json")
        if os.path.isfile(path):
            train_gene_path = path
            break
    if train_gene_path is None:
        return None
    with open(train_gene_path) as f:
        train_genes = json.load(f)['genes']

    ids_path = os.path.join(external_meta_dir, "ids.csv")
    if not os.path.isfile(ids_path):
        return None
    sample_ids = pd.read_csv(ids_path)["sample_id"].dropna().astype(str).tolist()

    from dataset.path_utils import st_dir as resolve_st_dir
    st_root = resolve_st_dir(external_asset_dir)

    measured = None
    try:
        import scanpy as sc
        for name in sample_ids:
            path = os.path.join(st_root, f"{name}.h5ad")
            if not os.path.isfile(path):
                continue
            var_names = set(sc.read_h5ad(path, backed='r').var_names)
            measured = var_names if measured is None else (measured & var_names)
    except Exception:
        return None
    if not measured:
        return None

    overlap = [g for g in train_genes if g in measured]
    if len(overlap) == len(train_genes):
        return None
    return overlap


def _num_folds(meta_dir):
    ids_path = os.path.join(meta_dir, "ids.csv")
    ids = pd.read_csv(ids_path, nrows=1)
    fold_cols = [col for col in ids.columns if col.startswith("fold_")]
    if not fold_cols:
        raise FileNotFoundError(f"No fold_* columns found in {ids_path}")
    return len(fold_cols)


def get_edge(x):
    try:
        return torch_geometric.nn.radius_graph(
            x,
            np.sqrt(2),
            None,
            False,
            max_num_neighbors=5,
            flow="source_to_target",
            num_workers=1,
        )
    except ImportError as exc:
        if "torch-cluster" not in str(exc):
            raise
        return _grid_radius_graph(x, radius=np.sqrt(2), max_num_neighbors=5)


def _grid_radius_graph(x, radius=np.sqrt(2), max_num_neighbors=5):
    coords = x.detach().cpu().round().long()
    coord_to_indices = {}
    for idx, coord in enumerate(coords.tolist()):
        coord_to_indices.setdefault(tuple(coord[:2]), []).append(idx)

    offsets = []
    ceil_radius = int(np.ceil(radius))
    for dx in range(-ceil_radius, ceil_radius + 1):
        for dy in range(-ceil_radius, ceil_radius + 1):
            if dx == 0 and dy == 0:
                continue
            if (dx * dx + dy * dy) ** 0.5 <= radius:
                offsets.append((dx, dy))

    sources = []
    targets = []
    for target, coord in enumerate(coords.tolist()):
        neighbors = []
        cx, cy = coord[:2]
        for dx, dy in offsets:
            neighbors.extend(coord_to_indices.get((cx + dx, cy + dy), []))
            if len(neighbors) >= max_num_neighbors:
                break
        for source in neighbors[:max_num_neighbors]:
            sources.append(source)
            targets.append(target)

    if not sources:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)
    return torch.tensor([sources, targets], dtype=torch.long, device=x.device)


def get_knn_graph(x, k=3):
    try:
        return torch_geometric.nn.knn_graph(x, k=k, loop=False)
    except ImportError as exc:
        if "torch-cluster" not in str(exc):
            raise
        return _chunked_knn_graph(x, k=k)


def _chunked_knn_graph(x, k=3, chunk_size=4096):
    if x.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)
    n = x.shape[0]
    if n <= 1:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)
    k = min(k, n - 1)
    x_float = x.float()
    sources = []
    targets = []
    all_indices = torch.arange(n, device=x.device)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        dist = torch.cdist(x_float[start:end], x_float)
        row_indices = torch.arange(start, end, device=x.device)
        dist[torch.arange(end - start, device=x.device), row_indices] = float("inf")
        nn = dist.topk(k, dim=1, largest=False).indices
        sources.append(nn.reshape(-1))
        targets.append(all_indices[start:end].repeat_interleave(k))
    return torch.stack([torch.cat(sources), torch.cat(targets)])


def get_cross_edge(x):
    l = x[0].shape[0]
    # l = len(x)
    source = torch.LongTensor(range(l))

    op = x[3].clone()
    opy = x[4].clone()
    
    # op = torch.cat([i[3] for i in x]).clone()
    # opy = torch.cat([i[4] for i in x]).clone()
    

    b,n,c= op.shape
    source = torch.repeat_interleave(source, n)
    
    ops = torch.cat((op,opy),-1).view(b*n,-1)
    ops,inverse = torch.unique(ops,dim=0, return_inverse=True)
    unique_op = ops[:,:c]
    unique_opy = ops[:,c:]
    
    edge = torch.stack((source,inverse))
    return unique_op, unique_opy, edge


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
    external_meta_dir=None,
    numk=6,
    meta_dir=None,
):
    meta_dir = meta_dir or data_dir
    asset_dir = asset_dir or data_dir
    external_asset_dir = external_asset_dir or external_dir
    external_meta_dir = external_meta_dir or external_dir
    num_fold = _num_folds(meta_dir)

    genes_override = None
    if external_dir is not None:
        genes_override = _external_genes_override(
            meta_dir, asset_dir, external_meta_dir, external_asset_dir, gene_type, num_genes,
        )

    for fold in range(num_fold):
        if fold_idx is not None and fold != fold_idx:
            continue

        print(f"Processing fold {fold}...")
        
        for phase in ["train", "test"]:
        
            # names = train_dataset["sample_id"].tolist()
            
            if external_dir is None:
                # split = os.path.join(split_dir, f"{phase}_{fold}.csv")
                # dataset = pd.read_csv(split)
                if cpm:
                    savename = f"{asset_dir}/EGGN/cpm/{model_name}/fold{fold}/{phase}"
                else:
                    savename = f"{asset_dir}/EGGN/{model_name}/fold{fold}/{phase}"
                os.makedirs(savename,exist_ok=True)
                
                ref_data_dir = None
            else:
                if phase == "train":
                    continue  # skip training phase for exemplar generation
                
                # test_split = os.path.join(external_dir, "ids.csv")
                # dataset = pd.read_csv(test_split)
                # Namespace by meta_dir, matching EGNDataset's own `ref_data`
                # derivation (used below via ref_data_dir=meta_dir) — using
                # data_dir here instead breaks whenever data_dir is a root
                # shared across every dataset (the documented convention).
                train_data = '/'.join(meta_dir.replace('/bench_data', '').split('/')[-2:])
                
                if cpm:
                    savename = f"{external_asset_dir}/EGGN/cpm/{model_name}/{train_data}/fold{fold}/{phase}"
                else:
                    savename = f"{external_asset_dir}/EGGN/{model_name}/{train_data}/fold{fold}/{phase}"
                os.makedirs(savename, exist_ok=True)
                
                ref_data_dir = meta_dir
                # data_dir = external_dir
            
            # temp_arg = namedtuple("arg",["numk","mdim", "index_path", "emb_path", "data"])
            # emb_path = f"{data_dir}/EGGN"
            # index_path = f"{savename}/index"
            foldername = f"{savename}/graph_{numk}"
            os.makedirs(foldername, exist_ok=True) 
                        
            # temp_arg = temp_arg(args.numk, args.mdim, index_path, emb_path, data_dir) 
            
            # Load dataset
            dataset = EGNDataset(
                mode='cv',
                phase=phase,
                fold=fold,
                data_dir=data_dir if external_dir is None else external_dir,
                meta_dir=meta_dir if external_dir is None else external_meta_dir,
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
                genes_override=genes_override if ref_data_dir is not None else None,
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

                pos, p, py, op, opy = x["pos"], x["ei"], x["label"], x["ej"], x["yj"]
                img_data = [pos, p, py, op, opy]
                img_data = [data.squeeze(0) for data in img_data]
                
                window_edge = get_edge((img_data[0]).clone())
                # img_data.append([pos, p, py, op, opy])
                # print(pos.size(), p.size(), py.size(), op.size(), opy.size())
                
                # window_edge = get_edge(torch.cat(([i[0] for i in img_data])).clone())
            
                unique_op, unique_opy, cross_edge = get_cross_edge(img_data)
            
                # print(window_edge.size(), unique_op.size(), unique_opy.size(), cross_edge.size())
                
                data = torch_geometric.data.HeteroData()
            
                data["window"].pos = img_data[0].clone()
                data["window"].x = img_data[1].clone()
                
                # data["window"].pos = torch.cat(([i[0] for i in img_data])).clone()
                # data["window"].x = torch.cat(([i[1] for i in img_data])).clone()
                data["window"].x = data["window"].x.squeeze()
                # data["window"].y = torch.cat(([i[2] for i in img_data])).clone()
                data["window"].y = img_data[2].clone()
                
                assert len(data["window"]["pos"]) == len(data["window"]["x"]) == len(data["window"]["y"])
                
                data["example"].x = torch.cat((unique_op, unique_opy),-1)
                
                data['window', 'near', 'window'].edge_index = window_edge
                data["example", "refer", "window"].edge_index = cross_edge[[1,0]]
                
                
                edge_index = get_knn_graph(data["example"]["x"], k=3)
                data["example", "close", "example"].edge_index = edge_index
                
                # sample_id = dataset.int2id[i]
                torch.save(data, save_name)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build exemplar dataset")
    parser.add_argument("--data_dir", type=str, default="input/takano/xenium", help="Path to the data directory")
    parser.add_argument("--numk",required=True,type=int, default=6)
    parser.add_argument("--external_dir", type=str, default=None, help="Path to the data directory")
    parser.add_argument("--asset_dir", type=str, default=None, help="Path to patches, embeddings, ST files, and generated assets")
    parser.add_argument("--external_asset_dir", type=str, default=None, help="Asset directory for external data")
    parser.add_argument("--external_meta_dir", type=str, default=None, help="Meta directory (ids.csv) for external data")
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
        external_dir=external_dir,
        model_name=model_name,
        gene_type=gene_type,
        num_genes=num_genes,
        cpm=cpm,
        overwrite=overwrite,
        fold_idx=fold_idx,
        asset_dir=args.asset_dir,
        external_asset_dir=args.external_asset_dir,
        external_meta_dir=args.external_meta_dir,
        numk=args.numk,
        meta_dir=args.meta_dir,
    )
