import os
from glob import glob
import argparse
import json

import pandas as pd
import numpy as np
import anndata as ad
import scanpy as sc
from tqdm import tqdm


def load_data(st_dir, id_path=None):
    if id_path:
        ids = pd.read_csv(id_path)['sample_id'].dropna().astype(str).tolist()
        file_list = [f"{st_dir}/{sample_id}.h5ad" for sample_id in ids]
    else:
        file_list = glob(f"{st_dir}/*.h5ad")
    missing = [path for path in file_list if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            "Some ST files listed in ids.csv are missing: "
            + ", ".join(missing[:10])
            + ("..." if len(missing) > 10 else "")
        )
    return [sc.read_h5ad(file) for file in tqdm(file_list, desc="Loading ST data")]


def find_geneset(data_list, n_top_hvg=50, n_top_heg=1000, n_top_hmhvg=200,
                 min_spot_percentage=0.1, method='ALL'):
    """Select gene sets from a list of AnnData objects.

    Args:
        data_list: List of AnnData objects.
        n_top_hvg: Number of top highly variable genes.
        n_top_heg: Number of top highly expressed genes.
        n_top_hmhvg: Number of top highly-mean-highly-variable genes.
        min_spot_percentage: Minimum fraction of spots a gene must appear in.
        method: One of 'HVG', 'HEG', 'HMHVG', or 'ALL'.

    Returns:
        Dict mapping gene-set prefix to list of gene names.
    """
    if method not in ('HMHVG', 'HVG', 'HEG', 'ALL'):
        raise ValueError("method must be one of 'HMHVG', 'HVG', 'HEG', 'ALL'")

    EXCLUDE_PREFIXES = ('NegControlCodeword', 'NegControlProbe', 'UnassignedCodeword')
    common_genes = list(set.intersection(*[
        {g for g in adata.var_names if not g.startswith(EXCLUDE_PREFIXES)}
        for adata in data_list
    ]))
    total_spots = sum(adata.shape[0] for adata in data_list)

    spot_counts = np.array(
        sum((adata[:, common_genes].X > 0).sum(axis=0) for adata in data_list)
    ).squeeze()
    expressed_genes = np.array(common_genes)[spot_counts / total_spots >= min_spot_percentage]

    output = {'total': sorted(common_genes)}

    if method in ('HMHVG', 'ALL'):
        common_genes_sorted = sorted(common_genes)
        data_lst = [adata[:, common_genes_sorted].copy() for adata in data_list]

        union_hvg = set()
        for adata in tqdm(data_lst, desc="Computing per-sample HVGs"):
            tmp = adata.copy()
            sc.pp.filter_cells(tmp, min_genes=1)
            sc.pp.filter_genes(tmp, min_cells=1)
            sc.pp.normalize_total(tmp, inplace=True)
            sc.pp.log1p(tmp)
            sc.pp.highly_variable_genes(tmp, n_top_genes=2000)
            union_hvg |= set(tmp.var_names[tmp.var["highly_variable"]])

        union_hvg = sorted(
            g for g in union_hvg
            if not g.startswith(("MT", "mt", "RPS", "RPL"))
        )

        # Build a spots × genes count matrix across all samples
        frames = [
            pd.DataFrame(
                adata[:, union_hvg].X.toarray(),
                columns=union_hvg,
                index=[f"s{i}_{j}" for j in range(adata.shape[0])],
            )
            for i, adata in enumerate(data_lst)
        ]
        all_counts = pd.concat(frames, axis=0).fillna(0)

        mean_ranks = all_counts.mean(axis=0).rank(ascending=False, method='min')
        std_ranks  = all_counts.std(axis=0).rank(ascending=False, method='min')
        combined   = mean_ranks + std_ranks

        output['hmhvg'] = combined.sort_values().head(n_top_hmhvg).index.tolist()
        # print(f"Selected HMHVG genes: {output['hmhvg']}")

    if method in ('HVG', 'ALL'):
        data_combined = []
        for adata in tqdm(data_list, desc="Normalizing per-sample data"):
            tmp = adata.copy()
            sc.pp.normalize_total(tmp, target_sum=1e4)
            sc.pp.log1p(tmp)
            data_combined.append(tmp[:, expressed_genes])
        data_combined = ad.concat(data_combined, label="batch")
        sc.pp.highly_variable_genes(data_combined, n_top_genes=n_top_hvg, batch_key="batch")
        output['var'] = expressed_genes[data_combined.var['highly_variable']].tolist()
        # print(f"Selected highly variable genes: {output['var']}")

    if method in ('HEG', 'ALL'):
        gene_counts = pd.concat(
            [
                pd.DataFrame(
                    np.array(adata[:, expressed_genes].X.sum(axis=0)).reshape(1, -1)
                )
                for adata in data_list
            ],
            axis=0,
        )
        total_counts = gene_counts.sum(axis=0)
        output['mean'] = expressed_genes[total_counts.argsort()[::-1][:n_top_heg]].tolist()
        # print(f"Selected highly expressed genes: {output['mean']}")

    return output


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--st_dir", type=str, required=True, help="Directory containing ST h5ad files")
    argparser.add_argument("--n_top_hvg", type=int, default=50)
    argparser.add_argument("--n_top_heg", type=int, default=1000)
    argparser.add_argument("--n_top_hmhvg", type=int, default=500)
    argparser.add_argument("--output_dir", type=str, required=True)
    argparser.add_argument("--method", type=str, default='HMHVG', choices=['HVG', 'HEG', 'HMHVG', 'ALL'])
    argparser.add_argument("--id_path", type=str, default=None)

    args = argparser.parse_args()

    data_list = load_data(args.st_dir, id_path=args.id_path)
    geneset = find_geneset(
        data_list,
        method=args.method,
        n_top_hvg=args.n_top_hvg,
        n_top_heg=args.n_top_heg,
        n_top_hmhvg=args.n_top_hmhvg,
    )

    for prefix, genes in geneset.items():
        with open(f"{args.output_dir}/{prefix}_{len(genes)}genes.json", "w") as f:
            json.dump({"genes": genes}, f)
