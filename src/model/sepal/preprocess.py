
import sys
import os
from tqdm import tqdm
os.environ['USE_PYGEOS'] = '0' # To supress a warning from geopandas
import squidpy as sq
import scanpy as sc
import warnings

from glob import glob
import pandas as pd
import numpy as np
import anndata as ad
from scipy import sparse

import torch
import torchvision.transforms as transforms

import json
import h5py
import argparse

from typing import Tuple
from torch_geometric.data import Data as geo_Data
from torch_geometric.utils import from_scipy_sparse_matrix

import torch
from positional_encodings.torch_encodings import PositionalEncoding2D

from backbone import LocalNet

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.train_utils import normalize_adata
from model.linear_prob.model import LinearProb


def _read_ids(meta_dir, phase, fold):
    split_path = os.path.join(meta_dir, "splits", f"{phase}_{fold}.csv")
    if os.path.isfile(split_path):
        return pd.read_csv(split_path)["sample_id"]

    ids_path = os.path.join(meta_dir, "ids.csv")
    ids = pd.read_csv(ids_path)
    fold_col = f"fold_{fold}"
    if fold_col not in ids.columns:
        raise FileNotFoundError(f"{split_path} not found and {fold_col} is missing from {ids_path}")
    return ids.loc[ids[fold_col].astype(str).str.lower() == phase, "sample_id"]


def _num_folds(meta_dir):
    split_dir = os.path.join(meta_dir, "splits")
    if os.path.isdir(split_dir):
        return len(glob(f"{split_dir}/train_*.csv"))

    ids_path = os.path.join(meta_dir, "ids.csv")
    ids = pd.read_csv(ids_path, nrows=1)
    fold_cols = [col for col in ids.columns if col.startswith("fold_")]
    if not fold_cols:
        raise FileNotFoundError(f"No splits directory or fold_* columns found in {meta_dir}")
    return len(fold_cols)


class SepalPreprocess:
    model_embedding_dims = {'uni_v2': 1536, 'ctranspath': 768}
    
    def __init__(self, 
                 ckpt_path: str,
                 dataset_path: str, 
                 backbone: str,
                 local_model: str = 'LocalNet',
                 ref_dataset_path: str = None,
                 ref_meta_dir: str = None,
                 hex_geometry: bool = True,
                 model_name: str = 'uni_v2',
                 num_genes: int = 200,
                 use_pretrained_emb: bool = True
                 ):
        """
        This class preprocesses the data for the SEPAL model. It computes the graphs for each patch in the slide
        and adds the positional encodings to the graph.

        Args:
            adata (ad.AnnData): The AnnData object with the slide data.
            patch_scale (int): The scale of the patches to compute the graphs.
            dataset_path (str): The path to the dataset.
            hex_geometry (bool): Whether the slide has hexagonal geometry or not.
        """
        
        # self.get_ckpt_path(ckpt_path, fold)
        self.ckpt_path = ckpt_path
        self.ref_dataset_path = ref_dataset_path
        self.ref_meta_dir = ref_meta_dir or ref_dataset_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        emb_dim = self.model_embedding_dims[model_name]
        self.model = self.load_model(local_model=local_model, backbone=backbone, emb_dim=emb_dim, num_genes=num_genes, use_pretrained_emb=use_pretrained_emb)
        
        self.dataset_path = dataset_path
        self.hex_geometry = hex_geometry
        
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        
        
        # self.adata = self.adata.to(self.device)
        # self.backbone = backbone
        
    # def get_ckpt_path(self, ckpt_path, fold=None):
    #     """
    #     This function returns the path to the checkpoint file for the model.

    #     Args:
    #         fold (int): The fold number for cross-validation.

    #     Returns:
    #         str: The path to the checkpoint file.
    #     """
        
    #     if fold is None:
    #         self.ckpt_path = ckpt_path
    #     else:
    #         # Get the checkpoint path
    #         ckpt_path = glob(f"{self.ckpt_path}/fold{fold}/*.ckpt")[0]
    #         if os.path.exists(ckpt_path):
    #             self.ckpt_path = ckpt_path
        
    def load_model(self, local_model: str, backbone: str, emb_dim: int = 1536, num_genes: int = 200, use_pretrained_emb=True):
        """
        This function loads the model for the SEPAL model.

        Args:
            model_name (str): The name of the model to load.
            num_genes (int): The number of genes in the dataset.
            fold (int): The fold number for cross-validation.

        Returns:
            nn.Module: The loaded model.
        """
        # Load the model
        if local_model == "LinearProb":
            model = LinearProb(img_embedding_dim=emb_dim, num_genes=num_genes)
            model.use_pretrained_emb = True
        elif local_model == "LocalNet":
            model = LocalNet(backbone=backbone, img_embedding_dim=emb_dim, num_genes=num_genes, use_pretrained_emb=use_pretrained_emb)
        else:
            raise ValueError(f"Unsupported Sepal local model: {local_model}")
        model = model.to(self.device)
        
        # Load the weights
        checkpoint = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint['state_dict']
        for key in list(state_dict.keys()):
            state_dict[key.replace('model.', '', 1)] = state_dict.pop(key)
        
        model.load_state_dict(state_dict)
        
        return model
        
    def compute_predictions(self, slide_name: str, return_emb: bool = False, model_name='uni_v2'):
        """
        This function computes the predictions for the patches in the slide using the model.

        Args:
            adata (ad.AnnData): The AnnData object with the slide data.
            model (nn.Module): The model to use for predictions.
            batch_size (int): The batch size to use for predictions.

        Returns:
            ad.AnnData: The AnnData object with the predictions added to it.
        """
        # Set the model to evaluation mode
        self.model.eval()

        # # Create a DataLoader for the patches
        # dataloader = DataLoader(adata, batch_size=batch_size, shuffle=False)
        # patch_path = os.path.join(self.dataset_path, 'patches', slide_name)
        
        if getattr(self.model, "use_pretrained_emb", True):
            emb_path = f"{self.dataset_path}/emb/global/features_{model_name}/{slide_name}.h5"
            
            with h5py.File(emb_path, 'r') as f:
                # Get the patches and labels
                if 'features' in f:
                    emb = f['features'][:]
                elif 'embeddings' in f:
                    emb = f['embeddings'][:]
                else:
                    raise ValueError(f"Cannot find 'features' or 'embeddings' in {emb_path}")
                emb = torch.Tensor(emb)
                emb = emb.to(self.device)
                
            with torch.no_grad():
                output = self.model(phase='test', return_emb=True, img_emb=emb)
        else:
            patch_path = f"{self.dataset_path}/patches/{slide_name}.h5"
            
            with h5py.File(patch_path, 'r') as f:
                # Get the patches and labels
                img = f['img'][:]
            
            img = torch.stack([self.transforms(im) for im in img], dim=0)
            img = img.to(self.device)
        
            with torch.no_grad():
                output = self.model(img, phase='test', return_emb=True)
            
        predictions = output['logits']
        embeddings = output['embeddings']
    
        return predictions, embeddings


    def get_graphs_one_slide(self, adata: ad.AnnData, n_hops: int, hex_geometry: bool) -> Tuple[dict,int]:
        """
        This function receives an AnnData object with a single slide and for each node computes the graph in an
        n_hops radius in a pytorch geometric format. It returns a dictionary where the patch names are the keys
        and a pytorch geometric graph for each one as values. NOTE: The first node of every graph is the center.

        Args:
            adata (ad.AnnData): The AnnData object with the slide data.
            n_hops (int): The number of hops to compute the graph.
            layer (str): The layer of the graph to predict. Will be added as y to the graph.
            hex_geometry (bool): Whether the slide has hexagonal geometry or not.

        Returns:
            Tuple(dict,int)
            dict: A dictionary where the patch names are the keys and pytorch geometric graph for each one as values.
                NOTE: The first node of every graph is the center.
            int: Max absolute value of d pos in the slide                      
        """
        # Compute spatial_neighbors
        if hex_geometry:
            sq.gr.spatial_neighbors(adata, coord_type='generic', n_neighs=6) # Hexagonal visium case
        else:
            sq.gr.spatial_neighbors(adata, coord_type='grid', n_neighs=8) # Grid STNet dataset case

        # Get the adjacency matrix
        adj_matrix = adata.obsp['spatial_connectivities']

        # Define power matrix
        power_matrix = adj_matrix.copy()
        # Define the output matrix
        output_matrix = adj_matrix.copy()

        # Iterate through the hops
        for i in range(n_hops-1):
            # Compute the next hop
            power_matrix = power_matrix * adj_matrix
            # Add the next hop to the output matrix
            output_matrix = output_matrix + power_matrix

        # Zero out the diagonal
        output_matrix.setdiag(0)
        # Threshold the matrix to 0 and 1
        output_matrix = output_matrix.astype(bool).astype(int)

        # Define dict from index to obs name
        index_to_obs = {i: obs for i, obs in enumerate(adata.obs.index.values)}

        # Define neighbors dicts (one with names and one with indexes)
        neighbors_dict_index = {}
        neighbors_dict_names = {}
        matrices_dict = {}

        # Iterate through the rows of the output matrix
        for i in range(output_matrix.shape[0]):
            # Get the non-zero elements of the row
            non_zero_elements = output_matrix[i].nonzero()[1]
            # Get the names of the neighbors
            non_zero_names = [index_to_obs[index] for index in non_zero_elements]
            # Add the neighbors to the neighbors dicts. NOTE: the first index is the query obs
            neighbors_dict_index[i] = [i] + list(non_zero_elements)
            neighbors_dict_names[index_to_obs[i]] = np.array([index_to_obs[i]] + non_zero_names)
            
            # Subset the matrix to the non-zero elements and store it in the matrices dict
            matrices_dict[index_to_obs[i]] = output_matrix[neighbors_dict_index[i], :][:, neighbors_dict_index[i]]

        
        ### Get pytorch geometric graphs ###
        # patch_names = adata.obs.index.values                                                                        # Get global patch names
        # layers_dict = {key: torch.from_numpy(adata.layers[key]).type(torch.float32) for key in adata.layers.keys()} # Get global layers
        counts = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
        counts = torch.from_numpy(counts).type(torch.float32)                                   # Get global counts
        # patches = torch.from_numpy(adata.obsm[f'patches_scale_{self.patch_scale}'])                                 # Get global patches
        pos = torch.from_numpy(adata.obs[['array_row', 'array_col']].values.astype('int64'))                                        # Get global positions

        # If embeddings and predictions are present in obsm, get them
        embeddings = torch.from_numpy(adata.obsm['embeddings']).type(torch.float32) if 'embeddings' in adata.obsm.keys() else None
        predictions = torch.from_numpy(adata.obsm['predictions']).type(torch.float32) if 'predictions' in adata.obsm.keys() else None

        # If layer contains delta then add a used_mean attribute to the graph
        # used_mean = torch.from_numpy(adata.var[f'{layer}_avg_exp'.replace('deltas', 'log1p')].values).type(torch.float32) if 'deltas' in layer else None

        # Define the empty graph dict
        graph_dict = {}
        max_abs_d_pos=-1

        # Cycle over each obs
        for i in tqdm(range(len(neighbors_dict_index)), leave=False, position=1):
            central_node_name = index_to_obs[i]                                                 # Get the name of the central node
            curr_nodes_idx = torch.tensor(neighbors_dict_index[i])                              # Get the indexes of the nodes in the graph
            curr_adj_matrix = matrices_dict[central_node_name]                                  # Get the adjacency matrix of the graph (precomputed)
            curr_edge_index, curr_edge_attribute = from_scipy_sparse_matrix(curr_adj_matrix)    # Get the edge index and edge attribute of the graph
            # curr_layers = {key: layers_dict[key][curr_nodes_idx] for key in layers_dict.keys()} # Get the layers of the graph filtered by the nodes
            curr_count = counts[curr_nodes_idx]                                                   # Get the counts of the nodes in the graph
            curr_pos = pos[curr_nodes_idx]                                                      # Get the positions of the nodes in the graph
            curr_d_pos = curr_pos - curr_pos[0]                                                 # Get the relative positions of the nodes in the graph

            # Define the graph
            graph_dict[central_node_name] = geo_Data(
                # x=patches[curr_nodes_idx],
                # y=curr_layers[layer],
                y=curr_count,
                edge_index=curr_edge_index,
                # edge_attr=curr_edge_attribute,
                pos=curr_pos,
                d_pos=curr_d_pos,
                # patch_names=patch_names[neighbors_dict_index[i]],
                embeddings=embeddings[curr_nodes_idx] if embeddings is not None else None,
                predictions=predictions[curr_nodes_idx] if predictions is not None else None,
                # used_mean=used_mean if used_mean is not None else None,
                num_nodes=len(curr_nodes_idx),
                # mask=layers_dict['mask'][curr_nodes_idx]
                # **curr_layers
            )

            max_curr_d_pos=curr_d_pos.abs().max()
            if max_curr_d_pos>max_abs_d_pos:
                max_abs_d_pos=max_curr_d_pos

        #cast as int
        max_abs_d_pos=int(max_abs_d_pos)
        
        # Return the graph dict
        return graph_dict, max_abs_d_pos

    def get_sin_cos_positional_embeddings(self, graph_dict: dict, max_d_pos: int) -> dict:
        
        """This function adds the positional embeddings of each node to the graph dict.

        Args:
            graph_dict (dict): A dictionary where the patch names are the keys and pytorch geometric graph for each one as values
            max_d_pos (int): Max absolute value in the relative position matrix.

        Returns:
            dict: The input graph dict with the information of positional encodings for each graph.
        """
        graph_dict_keys = list(graph_dict.keys())
        embedding_dim =graph_dict[graph_dict_keys[0]].embeddings.shape[1]

        # Define the positional encoding model
        p_encoding_model= PositionalEncoding2D(embedding_dim)

        # Define the empty grid with size (batch_size, x, y, channels)
        grid_size = torch.zeros([1, 2*max_d_pos+1, 2*max_d_pos+1, embedding_dim])

        # Obtain the embeddings for each position
        positional_look_up_table = p_encoding_model(grid_size)        

        for key, value in graph_dict.items():
            d_pos = value.d_pos
            grid_pos = d_pos + max_d_pos
            graph_dict[key].positional_embeddings = positional_look_up_table[0,grid_pos[:,0],grid_pos[:,1],:]
        
        return graph_dict
    
    def load_st(self, name: str, genes, normalize: bool = True, cpm=False, smooth=False):
        """Load gene expression data of a sample.

        Args:
            name (str): name of a sample
            normalize (bool): whether to normalize gene expression data.
            cpm (bool): whether to conduct CPM while normalizing gene expression data.
            smooth (bool): whether to smooth gene expression data.
            
        Returns:
            annData: return adata of st data. 
        """
        path = f"{self.dataset_path}/adata/{name}.h5ad"
        if not os.path.isfile(path):
            path = f"{self.dataset_path}/st/{name}.h5ad"
        adata = sc.read_h5ad(path)
        
        
        # common_genes = list(set(genes).intersection(set(adata.var_names)))
        if adata.var_names.isin(genes).sum() < len(genes):
            common_genes = list(set(genes).intersection(set(adata.var_names)))
            warnings.warn(f"Some genes are not found in {name}.h5ad. Use {len(common_genes)} / {len(genes)} genes.")
            adata = adata[:, common_genes]
        else:
            adata = adata[:, genes]
        
        if normalize:
            adata = normalize_adata(adata, cpm=cpm, smooth=smooth)
    
        return adata

    def prepare_graph(self, 
                    slide_name: str,
                    # layer: str = 'y', 
                    n_hops: int = 2,
                    model_name='uni_v2',
                    gene_type: str = 'hmhvg',
                    num_genes: int = 200,
                    cpm: bool = False,
                    smooth: bool = False):
            
        # Get dictionary of parameters to get the graphs
        curr_graph_params = {
            'n_hops': n_hops,
            # 'layer': layer
            # 'backbone': backbone,
            # 'model_path': model_path
        }        

        # adata = sc.read_h5ad(f"{self.dataset_path}/adata/{slide_name}.h5ad")
        if self.ref_meta_dir is not None:
            gene_path = f"{self.ref_meta_dir}/{gene_type}_{num_genes}genes.json"
        else:
            gene_path = f"{self.dataset_path}/{gene_type}_{num_genes}genes.json"

        with open(gene_path, 'r') as f:
            genes = json.load(f)['genes']

        adata = self.load_st(name=slide_name, genes=genes, normalize=True, cpm=cpm, smooth=smooth)
        # pred_path = glob(f"{self.pred_path}/fold*")[0]
        # adata = sc.read_h5ad(f"{pred_path}/{slide_name}.h5ad")
        preds, embs = self.compute_predictions(slide_name=slide_name, return_emb=True, model_name=model_name)
        
        adata.obsm['predictions'] = preds.cpu().numpy()
        adata.obsm['embeddings'] = embs.cpu().numpy()
        
        graph_dict, max_curr_d_pos = self.get_graphs_one_slide(adata, n_hops, self.hex_geometry)
            
        # if max_curr_d_pos>max_global_d_pos:
        #     max_global_d_pos=max_curr_d_pos
        graph_dict = self.get_sin_cos_positional_embeddings(graph_dict, max_curr_d_pos)
        
        # Get graph dicts
        # general_graph_dict = self.get_graphs(n_hops=n_hops, layer=layer)

        # Get the train, validation and test indexes
        # idx_train, idx_val, idx_test = self.adata.obs[self.adata.obs.split == 'train'].index, self.adata.obs[self.adata.obs.split == 'val'].index, self.adata.obs[self.adata.obs.split == 'test'].index

        # # Get list of graphs
        # train_graphs = [general_graph_dict[idx] for idx in idx_train]
        # val_graphs = [general_graph_dict[idx] for idx in idx_val]
        # test_graphs = [general_graph_dict[idx] for idx in idx_test] if len(idx_test) > 0 else None

        print('Saving graphs...')
        # Create graph directory if it does not exist with the current time
        # graph_dir = os.path.join(self.dataset_path, 'graphs', datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
        # os.makedirs(graph_dir, exist_ok=True)

        # Save the graph parameters
        # with open(os.path.join(graph_dir, 'graph_params.json'), 'w') as f:
        #     # Write the json
        #     json.dump(curr_graph_params, f, indent=4)

        # torch.save(train_graphs, os.path.join(graph_dir, 'train_graphs.pt'))
        # torch.save(val_graphs, os.path.join(graph_dir, 'val_graphs.pt'))
        # torch.save(test_graphs, os.path.join(graph_dir, 'test_graphs.pt')) if test_graphs is not None else None
        
        return graph_dict, curr_graph_params


def get_args():

    parser = argparse.ArgumentParser(description='SEPAL Preprocess')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset')
    # parser.add_argument('--ref_dataset_path', type=str, default=None, help='Path to the reference dataset')
    parser.add_argument('--fold_idx', type=int, default=None, help='Fold index')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to the checkpoint file')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train', help='Mode of operation: train or test')
    parser.add_argument("--meta_dir", type=str, default=None, help="Path to ids.csv and gene lists")
    parser.add_argument("--external_dir", type=str, default=None, help="Path to the data directory")
    parser.add_argument("--local_model", type=str, default="LocalNet", choices=["LocalNet", "LinearProb"], help="Local model checkpoint type")
    parser.add_argument("--model_name", type=str, default="uni_v2", help="Name of the model")
    parser.add_argument("--gene_type", type=str, default="hmhvg", help="Type of genes to use")
    parser.add_argument("--num_genes", type=int, default=200, help="Number of genes to use")
    parser.add_argument("--cpm", action="store_true", default=False, help="Whether to use CPM normalization")
    parser.add_argument("--smooth", action="store_true", default=False, help="Whether to use smoothing")
    parser.add_argument("--use_pretrained_emb", action="store_true", default=False, help="Whether to use pretrained embeddings")

    return parser.parse_args()

def main():
    
    args = get_args()
    dataset_path = args.dataset_path
    meta_dir = args.meta_dir or dataset_path
    
    external_dir = args.external_dir if args.external_dir else None
    # ref_dataset_path = args.ref_dataset_path
    ckpt_path = args.ckpt_path
    mode = args.mode
    model_name = args.model_name
    gene_type = args.gene_type
    num_genes = args.num_genes
    cpm = args.cpm
    smooth = args.smooth
    use_pretrained_emb = args.use_pretrained_emb
    model_type = 'modified' if use_pretrained_emb else 'original'
    if args.local_model == "LinearProb":
        model_type = "linear_prob"
        use_pretrained_emb = True
    fold_idx = args.fold_idx
    
    if mode == 'train':
        num_folds = _num_folds(meta_dir)
        for fold in range(num_folds):
            if fold_idx is not None and fold != fold_idx:
                continue
            
            print(f"Processing fold {fold}...")
            ckpt_dir = sorted(glob(f"{args.ckpt_path}/*"))[-1]
            ckpt_path = glob(f"{ckpt_dir}/fold{fold}/*.ckpt")[0]
            
            for phase in ['train', 'test']:
                if external_dir is None:    
                    print(f"Processing phase {phase}...")
                    split = _read_ids(meta_dir, phase, fold)
                    
                    if cpm:
                        save_dir = f"{dataset_path}/sepal/cpm/{model_type}/fold{fold}/{phase}"
                    else:
                        save_dir = f"{dataset_path}/sepal/{model_type}/fold{fold}/{phase}"
                    os.makedirs(save_dir, exist_ok=True)

                    # ref_dataset_path=None
                    
                else:
                    if phase == 'train':
                        continue
                    else:
                        print(f"Processing external phase {phase}...")
                        split_path = f"{external_dir}/ids.csv"
                        split = pd.read_csv(split_path)["sample_id"]
                        
                        train_data = '/'.join(dataset_path.replace('/bench_data', '').split('/')[-2:])
                        
                        # save_dir = f"{external_dir}/sepal/fold{fold}/{phase}"
                        if cpm:
                            save_dir = f"{external_dir}/sepal/cpm/{model_type}/{train_data}/fold{fold}/{phase}"
                        else:
                            save_dir = f"{external_dir}/sepal/{model_type}/{train_data}/fold{fold}/{phase}"
                        os.makedirs(save_dir, exist_ok=True)
                    
                for slide_name in tqdm(split):
                    # Prepare the graph for each slide
                    sepal_preprocess = SepalPreprocess(
                        ckpt_path=ckpt_path,
                        dataset_path=dataset_path if external_dir is None else external_dir,
                        ref_dataset_path=dataset_path,
                        ref_meta_dir=meta_dir,
                        backbone='ViT',
                        local_model=args.local_model,
                        model_name=model_name,
                        num_genes=num_genes,
                        use_pretrained_emb=use_pretrained_emb
                    )
                    
                    graph_dict, curr_graph_params = sepal_preprocess.prepare_graph(
                        slide_name=slide_name,
                        # layer='y',
                        n_hops=2,
                        model_name=model_name,
                        gene_type=gene_type,
                        num_genes=num_genes,
                        cpm=cpm,
                        smooth=smooth
                    )
                    
                    torch.save(graph_dict, f"{save_dir}/{slide_name}.pt")
                    
                    with open(f'{save_dir}/{slide_name}_graph_params.json', 'w') as f:
                        # Write the json
                        json.dump(curr_graph_params, f, indent=4)
                        
                        
    elif mode == 'test':
        print("Testing mode is not implemented yet.")
        
                
if __name__ == '__main__':
    main()
