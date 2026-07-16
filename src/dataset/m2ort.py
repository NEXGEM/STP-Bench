
import os

import json
import numpy as np
from scipy import sparse
import h5py
import torch

from dataset.base_dataset import STDataset


class M2OSTDataset(STDataset):
    def __init__(self,
                 mode: str,
                 phase: str,
                 fold: int,
                 data_dir: str,
                 meta_dir: str = None,
                 ref_data_dir: str = None,
                 gene_type: str = 'var',
                 num_genes: int = 50,
                 num_outputs: int = 50,
                 normalize: bool = True,
                 cpm: bool = False,
                 smooth: bool = False,
                 data_id: str = None,
                 load_level: str = 'patch'):
        super().__init__(mode,
                         phase=phase,
                         fold=fold,
                         data_dir=data_dir,
                         meta_dir=meta_dir,
                         ref_data_dir=ref_data_dir,
                         gene_type=gene_type,
                         num_genes=num_genes,
                         num_outputs=num_outputs,
                         normalize=normalize,
                         cpm=cpm,
                         smooth=smooth,
                         data_id=data_id,
                         load_level=load_level)

    def __getitem__(self, index):
        data = {}

        if self.phase == 'train':
            i = 0
            while index >= self.cumlen[i]:
                i += 1
            idx = index
            if i > 0:
                idx = index - self.cumlen[i - 1]

            name = self.int2id[i]  
            
            img0 = self.load_img(name, idx, level=0)
            img0 = self.transforms(img0)
            
            img2 = self.load_img(name, idx, level=0)
            img2 = self.transforms(img2)
            
            img3 = self.load_img(name, idx, level=0)
            img3 = self.transforms(img3)

            adata = self.adata_dict[name]
            expr = adata[idx].X
            expr = expr.toarray().squeeze(0) if sparse.issparse(expr) else expr.squeeze(0)
            label = torch.FloatTensor(expr)

            data['img'] = img0
            data['img2'] = img2 
            data['img3'] = img3 
            data['label'] = label
            
            return data

        elif self.phase == 'test':
            
            if self.mode == 'inference':
                img0 = self.load_img(self.name, idx=index, level=0)
                img0 = self.transforms(img0)
                
                img2 = self.load_img(self.name, idx=index, level=1)
                img2 = self.transforms(img2)
                
                img3 = self.load_img(self.name, idx=index, level=2)
                img3 = self.transforms(img3)

                data['img'] = img0
                data['img2'] = img2
                data['img3'] = img3
            
            else:
                
                name = self.int2id[index]

                img0 = self.load_img(name, level=0)
                img0 = torch.stack([self.transforms(im) for im in img0], dim=0)
                
                img2 = self.load_img(name, level=1)
                img2 = torch.stack([self.transforms(im) for im in img2], dim=0)
                
                img3 = self.load_img(name, level=2)
                img3 = torch.stack([self.transforms(im) for im in img3], dim=0)

                data['img'] = img0
                data['img2'] = img2
                data['img3'] = img3

                adata = self.load_st(name, self.genes, **self.norm_param)
                expr = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
                data['label'] = torch.FloatTensor(expr)  

            return data
