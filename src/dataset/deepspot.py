
import os

import json
import numpy as np
from scipy import sparse
import h5py
import torch

from dataset.base_dataset import STDataset


class DeepSpotDataset(STDataset):
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
                load_level: str = 'patch' # 'patch' or 'slide'
                ):
        super(DeepSpotDataset, self).__init__(
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
                                load_level=load_level
                            )

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
            spot_emb = self.load_emb(name, emb_name='global', idx=idx)
            sub_spot_emb = self.load_emb(name, emb_name='target', idx=idx)
            neighbor_emb, _ = self.load_emb(name, emb_name='neighbor', idx=idx)
            
            adata = self.adata_dict[name]
            expression = adata[idx].X
            expression = expression.toarray().squeeze(0) \
                if sparse.issparse(expression) else expression.squeeze(0)
            
            data['spot_emb'] = spot_emb
            data['sub_spot_emb'] = sub_spot_emb
            data['neighbor_emb'] = neighbor_emb
            data['label'] = torch.FloatTensor(expression)
            # data['pid'] = torch.LongTensor([i])
            # data['sid'] = torch.LongTensor([idx])
            
        elif self.phase == 'test':
            if self.mode == 'inference':
                # img = self.img[index]
                # img = self.transforms(img)
                spot_emb = self.load_emb(self.name, emb_name='global', idx=index)
                sub_spot_emb = self.load_emb(self.name, emb_name='target', idx=index)
                neighbor_emb, _ = self.load_emb(self.name, emb_name='neighbor', idx=index)
                
                # neighbor_emb, mask = self.load_emb(self.name, emb_name='neighbor', idx=index)
                # global_emb = self.load_emb(self.name, emb_name='global')
                # pos = np.load(f"{self.data_dir}/pos/{self.name}.npy")
                data['spot_emb'] = spot_emb
                data['sub_spot_emb'] = sub_spot_emb
                data['neighbor_emb'] = neighbor_emb
                # data['sid'] = torch.LongTensor([index])
            else:
                name = self.int2id[index]
                # img = self.load_img(name)
                # img = torch.stack([self.transforms(im) for im in img], dim=0)
                
                # neighbor_emb, mask = self.load_emb(name, emb_name='neighbor')
                spot_emb = self.load_emb(name, emb_name='global')
                sub_spot_emb = self.load_emb(name, emb_name='target')
                neighbor_emb, _ = self.load_emb(name, emb_name='neighbor')
            
                if os.path.isfile(f"{self.st_dir}/{name}.h5ad"):
                    adata = self.load_st(name, self.genes, **self.norm_param)
                    expression = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
                    data['label'] = torch.FloatTensor(expression)
                    
                
                data['spot_emb'] = spot_emb
                data['sub_spot_emb'] = sub_spot_emb
                data['neighbor_emb'] = neighbor_emb
                    
            # data['img'] = img
            # data['mask'] = mask
            # data['neighbor_emb'] = neighbor_emb
            
        return data
