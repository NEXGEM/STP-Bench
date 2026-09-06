

import os

import json
import numpy as np
from scipy import sparse
import h5py
import torch

from dataset.base_dataset import STDataset
from dataset.path_utils import emb_dir


class StemDataset(STDataset):
    """
    Dataset for the Stem model which is a diffusion-based gene expression predictor.
    """
    def __init__(self, 
                mode: str,
                phase: str,
                fold: int,
                data_dir: str,
                meta_dir: str = None,
                gene_type: str = 'mean',
                num_genes: int = 1000,
                num_outputs: int = 300,
                normalize: bool = True,
                cpm: bool = False,
                smooth: bool = False,
                data_id: str = None,
                model_name: str = 'uni_v2',
                load_level: str = 'patch' # 'patch' or 'slide'
                ):
        super(StemDataset, self).__init__(
                                mode=mode,
                                phase=phase,
                                fold=fold,
                                data_dir=data_dir,
                                meta_dir=meta_dir,
                                gene_type=gene_type,
                                num_genes=num_genes,
                                num_outputs=num_outputs,
                                normalize=normalize,
                                cpm=cpm,
                                smooth=smooth,
                                data_id=data_id,
                                model_name=model_name,
                                load_level=load_level
                                )

        # Base directories for embeddings
        self.emb_dir = emb_dir(data_dir)
        # self.uni_emb_dir = f"{emb_dir}/global/uni_v2"
        # self.conch_emb_dir = f"{emb_dir}/global/conch_v1"
        
        # For augmented embeddings
        # self.use_augmented = use_augmented
        # self.aug_ratio = aug_ratio
        
        # self.uni_aug_emb_dirs = [d for d in glob(self.uni_emb_dir + "/*") if os.path.isdir(d)]
        # self.conch_aug_emb_dirs = [d for d in glob(self.conch_emb_dir + "/*") if os.path.isdir(d)]
        
        # self.augmentations = ['hori', 'vert', 'rot_90', 'rot_180', 'rot_270', 'tp', 'tv']
        

    def __getitem__(self, index):
        """Get item implementation compatible with the Stem training pipeline"""
        if self.phase == 'train':
            i = 0
            while index >= self.cumlen[i]:
                i += 1
            idx = index
            if i > 0:
                idx = index - self.cumlen[i-1]

            name = self.int2id[i]
            
            # Get expression data
            adata = self.adata_dict[name]
            expression = adata[idx].X
            expression = expression.toarray().squeeze(0) \
                if sparse.issparse(expression) else expression.squeeze(0)
            
            # Check for NaN/zeros and skip if necessary
            if np.isnan(expression).all() or np.sum(expression) == 0:
                # Return a neighboring valid spot instead
                alternate_idx = (idx + 1) % len(adata)
                expression = adata[alternate_idx].X
                expression = expression.toarray().squeeze(0) \
                    if sparse.issparse(expression) else expression.squeeze(0)
            
            # Load pre-computed embeddings (combined uni and conch)
            if self.model_name =='uni_conch':
                conch_emb = self.load_emb(name, idx=idx, model_name='conch_v1')
                uni_emb = self.load_emb(name, idx=idx, model_name='uni_v1')
                img_emb = torch.cat([uni_emb, conch_emb], dim=0)
                
            else:
                img_emb = self.load_emb(name, idx=idx)
                
            # conch_emb = self.load_emb(name, idx, model_name='conch_v1')
            # y = torch.cat([uni_emb, conch_emb], dim=0)
            
            # Return format expected by the training loop
            # return torch.FloatTensor(expression).unsqueeze(0), y  # (Gene_count, 1), Embedding
            return {'img_emb': img_emb, 'label': torch.FloatTensor(expression)}
            
        elif self.phase == 'test':

            if self.mode == 'inference':
                # Load pre-computed embeddings for all spots
                if self.model_name =='uni_conch':
                    conch_emb = self.load_emb(self.name, model_name='conch_v1')
                    uni_emb = self.load_emb(self.name, model_name='uni_v1')
                    img_emb = torch.cat([uni_emb, conch_emb], dim=1)
                else:
                    img_emb = self.load_emb(self.name)
                # conch_emb = self.load_emb(name, model_name='conch_v1')
                # y = torch.cat([uni_emb, conch_emb], dim=1)
                
                return {'img_emb': img_emb}  # (N_spots, Gene_count, 1), (N_spots, Embedding_dim)
            
            else:
                name = self.int2id[index]
                
                # Load pre-computed embeddings for all spots
                if self.model_name =='uni_conch':
                    conch_emb = self.load_emb(name, model_name='conch_v1')
                    uni_emb = self.load_emb(name, model_name='uni_v1')
                    img_emb = torch.cat([uni_emb, conch_emb], dim=1)
                else:
                    img_emb = self.load_emb(name)
                # conch_emb = self.load_emb(name, model_name='conch_v1')
                # y = torch.cat([uni_emb, conch_emb], dim=1)
                
                
                adata = self.load_st(name, self.genes, **self.norm_param)
                expression = adata.X.toarray() if sparse.issparse(adata.X) else adata.X

                return {'img_emb': img_emb, 'label': torch.FloatTensor(expression)}  # (N_spots, Gene_count, 1), (N_spots, Embedding_dim)
            
            