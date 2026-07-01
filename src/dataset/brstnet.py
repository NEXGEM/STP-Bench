import os

import json
import numpy as np
from scipy import sparse
import h5py
from glob import glob
import torch

from dataset.base_dataset import STDataset


class BrSTNetDataset(STDataset):
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
                 cpm: bool = True,
                 smooth: bool = True,
                 data_id: str = None,
                 model_name: str = 'uni_v2',
                 load_level: str = 'patch'):
        super().__init__(
                            mode=mode,
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
                            model_name=model_name,
                            load_level=load_level
                        )
    
        if mode != 'inference':
            total_gene_path = glob(f"{self.meta_dir}/total_*.json")[0]
            with open(total_gene_path, 'r') as f:
                total_genes = json.load(f)['genes']
            
            self.remaining_genes = list(set(total_genes) - set(self.genes))

        if phase == 'train':
            self.adata_aux_dict = {
                _id: self.load_st(_id, self.remaining_genes, **self.norm_param)
                for _id in self.ids
            }
        # norm_dir = os.path.join(self.data_dir, 'patches', 'patches_norm')
        # if os.path.isdir(norm_dir):
        #     self.img_dir = norm_dir

        # if self.phase != 'train':
        #     if hasattr(self, 'adata_dict'):
        #         del self.adata_dict
        #     if hasattr(self, 'lengths'):
        #         del self.lengths, self.cumlen

        #     self.lengths = []  
        #     for sid in self.ids:
        #         adata = sc.read_h5ad(os.path.join(self.st_dir, f"{sid}.h5ad"), backed='r')
        #         self.lengths.append(adata.n_obs)
        #         adata.file.close()
            
        #     self.cumlen = np.cumsum(self.lengths)

    def __getitem__(self, index):
        data = {}
        
        if self.phase == 'train':
            i = 0
            while index >= self.cumlen[i]:
                i += 1
            idx = index
            if i > 0:
                idx = index - self.cumlen[i-1]

            name = self.int2id[i]
            img = self.load_img(name, idx)
            img = self.transforms(img)
            
            img_emb = self.load_emb(name, emb_name='global', idx=idx)
            
            adata = self.adata_dict[name]
            expression = adata[idx].X
            expression = expression.toarray().squeeze(0) \
                if sparse.issparse(expression) else expression.squeeze(0)
            
            adata_aux = self.adata_aux_dict[name]
            aux = adata_aux[idx].X
            aux = aux.toarray().squeeze(0) \
                if sparse.issparse(aux) else aux.squeeze(0)
            
            data['img'] = img
            data['img_emb'] = img_emb
            data['label'] = torch.FloatTensor(expression) 
            data['aux'] = torch.FloatTensor(aux)
            
        elif self.phase == 'test':
            
            if self.mode == 'inference':
                img = self.load_img(self.name, idx=index)
                # img = self.img[index]
                img = self.transforms(img)
                img_emb = self.load_emb(self.name, emb_name='global', idx=index)
            else:
                name = self.int2id[index]
                img = self.load_img(name)
                img = torch.stack([self.transforms(im) for im in img], dim=0)
                
                img_emb = self.load_emb(name, emb_name='global')
                
                if os.path.isfile(f"{self.st_dir}/{name}.h5ad"):
                    adata = self.load_st(name, self.genes, **self.norm_param)
                    expression = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
                    data['label'] = torch.FloatTensor(expression)

                    adata_aux = self.load_st(name, self.remaining_genes, **self.norm_param)
                    aux = adata_aux.X.toarray() if sparse.issparse(adata_aux.X) else adata_aux.X
                    data['aux'] = torch.FloatTensor(aux)
            
            data['img'] = img
            data['img_emb'] = img_emb
            
        return data
    
