

import os

import json
import numpy as np
from scipy import sparse
import h5py
import torch

from dataset.base_dataset import STDataset
from dataset.path_utils import emb_dir


class FlowDataset(STDataset):
    """
    Dataset for the Stem model which is a diffusion-based gene expression predictor.
    """
    def __init__(self, 
                mode: str,
                phase: str,
                fold: int,
                data_dir: str,
                meta_dir: str = None,
                ref_data_dir: str = None,
                genes_override: list = None,
                wsi_dir: str = None,
                gene_type: str = 'mean',
                num_genes: int = 1000,
                num_outputs: int = 300,
                normalize: bool = True,
                cpm: bool = False,
                smooth: bool = False,
                data_id: str = None,
                model_name: str = 'uni_v2',
                load_level: str = 'slide'):
        super(FlowDataset, self).__init__(
                                mode=mode,
                                phase=phase,
                                fold=fold,
                                data_dir=data_dir,
                                meta_dir=meta_dir,
                                ref_data_dir=ref_data_dir,
                                genes_override=genes_override,
                                wsi_dir=wsi_dir,
                                gene_type=gene_type,
                                num_genes=num_genes,
                                num_outputs=num_outputs,
                                normalize=normalize,
                                cpm=cpm,
                                smooth=smooth,
                                data_id=data_id,
                                model_name=model_name,
                                load_level=load_level)
        
        # Base directories for embeddings
        self.emb_dir = emb_dir(data_dir)
    
    def __getitem__(self, index):
        
        if self.mode == 'inference':
            
            img_emb, coords = self.load_emb(self.name, return_crds=True)
            
            return {'img_features': img_emb, 'coords': coords}
        else:
            name = self.int2id[index]
            
            # Load pre-computed embeddings for all spots
            img_emb, coords = self.load_emb(name, return_crds=True)
            
            adata = self.load_st(name, self.genes, **self.norm_param)
            expression = adata.X.toarray() if sparse.issparse(adata.X) else adata.X

            return {'img_features': img_emb, 'label': torch.FloatTensor(expression), 'coords': coords}
        