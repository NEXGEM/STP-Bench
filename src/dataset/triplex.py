
import os

import json
import numpy as np
from scipy import sparse
import h5py
import torch

from dataset.base_dataset import STDataset
from dataset.path_utils import emb_dir


class TriDataset(STDataset):
    def __init__(self, 
                mode: str,
                phase: str,
                fold: int,
                data_dir: str,
                meta_dir: str = None,
                ref_data_dir: str = None,
                wsi_dir: str = None,
                gene_type: str = 'mean',
                num_genes: int = 1000,
                num_outputs: int = 300,
                normalize: bool = True,
                cpm: bool = False,
                smooth: bool = False,
                data_id: str = None,
                model_name: str = 'uni_v2',
                load_level: str = 'patch'  # 'patch' or 'slide'
                ):
        super(TriDataset, self).__init__(
                                mode=mode,
                                phase=phase,
                                fold=fold,
                                data_dir=data_dir,
                                meta_dir=meta_dir,
                                ref_data_dir=ref_data_dir,
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

        self.emb_dir = emb_dir(data_dir)
        
        if phase == 'train':

            self.global_data = {_id: self.load_emb(_id, emb_name='global', return_crds=True) \
                for _id in self.ids}
            self.pos_dict = {_id: data[1] \
                for _id, data in self.global_data.items()}
            self.global_embs = {_id: data[0] \
                for _id, data in self.global_data.items()}
        
        self.gene_ids = None
        if mode == 'inference':
            self.global_emb, self.position = self.load_emb(self.name, emb_name='global', return_crds=True)
            # self.gene_ids = None
        # else:    
        #     with open('/home/shared/chungym/hier_st/GenePT_emebdding_v2/gene2id.json', 'r') as f:
        #         gene2id = json.load(f)
        #
        #     self.gene_ids = [gene2id[gene] for gene in self.genes]
            
            
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
            
            spot_emb = self.load_emb(name, emb_name='global', idx=idx)
            # sub_spot_emb = self.load_emb(name, emb_name='target', idx=idx)
            neighbor_emb, mask = self.load_emb(name, emb_name='neighbor', idx=idx)
            
            # slide_emb = self.load_emb(name, emb_name='global', model_name='titan')

            adata = self.adata_dict[name]
            expression = adata[idx].X
            expression = expression.toarray().squeeze(0) \
                if sparse.issparse(expression) else expression.squeeze(0)
            
            data['img'] = img
            data['mask'] = mask
            data['spot_emb'] = spot_emb
            # data['sub_spot_emb'] = sub_spot_emb
            data['neighbor_emb'] = neighbor_emb
            # data['slide_emb'] = slide_emb
            data['label'] = torch.FloatTensor(expression)
            data['pid'] = torch.LongTensor([i])
            data['sid'] = torch.LongTensor([idx])
            if self.gene_ids is not None:
                data['gene_idx'] = torch.LongTensor(self.gene_ids)
            
        elif self.phase == 'test':
            if self.mode == 'inference':
                # img = self.img[index]
                img = self.load_img(self.name, idx=index)
                img = self.transforms(img)
                spot_emb = self.load_emb(self.name, emb_name='global', idx=index)
                # try:
                #     sub_spot_emb = self.load_emb(self.name, emb_name='target', idx=index)
                # except:
                #     sub_spot_emb = spot_emb
                neighbor_emb, mask = self.load_emb(self.name, emb_name='neighbor', idx=index)
                # slide_emb = self.load_emb(self.name, emb_name='global', model_name='titan')
                # global_emb = self.load_emb(self.name, emb_name='global')
                # pos = np.load(f"{self.data_dir}/pos/{self.name}.npy")
                data['sid'] = torch.LongTensor([index])
            else:
                name = self.int2id[index]
                img = self.load_img(name)
                img = torch.stack([self.transforms(im) for im in img], dim=0)
                
                spot_emb = self.load_emb(name, emb_name='global')
                # try:
                #     sub_spot_emb = self.load_emb(name, emb_name='target')
                # except:
                #     sub_spot_emb = spot_emb
                neighbor_emb, mask = self.load_emb(name, emb_name='neighbor')
                global_emb, pos = self.load_emb(name, emb_name='global', return_crds=True)
                # slide_emb = self.load_emb(name, emb_name='global', model_name='titan')
            
                if os.path.isfile(f"{self.st_dir}/{name}.h5ad"):
                    adata = self.load_st(name, self.genes, **self.norm_param)

                    expression = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
                    data['label'] = torch.FloatTensor(expression)

                data['position'] = pos
                data['global_emb'] = global_emb
                    
            
            data['img'] = img
            data['spot_emb'] = spot_emb
            # data['sub_spot_emb'] = sub_spot_emb
            data['mask'] = mask
            data['neighbor_emb'] = neighbor_emb
            # data['slide_emb'] = slide_emb
            if self.gene_ids is not None:
                data['gene_idx'] = torch.LongTensor(self.gene_ids)
            
        return data
        
    # def load_emb(self, name: str, emb_name: str = 'global', idx: int = None, model_name=None, return_crds=False):
    #     if emb_name not in ['global', 'neighbor', 'target']:
    #         raise ValueError(f"emb_name must be 'global' or 'neighbor' or 'target', but got {emb_name}")
        
    #     if model_name is None:
    #         model_name = self.model_name

    #     if model_name == 'titan':
    #         emb_name = f"{emb_name}/slide_features_{model_name}/20x_512px_0px_overlap"
    #         path = f"{self.emb_dir}/{emb_name}/slide_features_{model_name}/{name}.h5"
    #     else:
    #         path = f"{self.emb_dir}/{emb_name}/features_{model_name}/{name}.h5"
    #     # path = f"{self.emb_dir}/{emb_name}/features_{model_name}/{name}.h5"

    #     with h5py.File(path, 'r') as f:
    #         if 'embeddings'in f:
    #             emb = f['embeddings'][idx] if idx is not None else f['embeddings'][:]
    #         else:
    #             emb = f['features'][idx] if idx is not None else f['features'][:]
                
    #         emb = torch.Tensor(emb)
            
    #         if emb_name == 'neighbor':
    #             mask = f['mask_tb'][idx] if idx is not None else f['mask_tb'][:]
    #             mask = torch.LongTensor(mask)
    #             # return emb, mask
    #             if return_crds:
    #                 coords = f['coords'][idx] if idx is not None else f['coords'][:]
    #                 coords = torch.Tensor(coords)
    #                 return emb, mask, coords
    #             else:
    #                 return emb, mask
    #         else:
    #             if return_crds:
    #                 coords = f['coords'][idx] if idx is not None else f['coords'][:]
    #                 coords = torch.Tensor(coords)
    #                 return emb, coords
    #             else:
    #                 return emb
                
                
class GlobalDataset(STDataset):
    def __init__(self, 
                mode: str,
                phase: str,
                fold: int,
                data_dir: str,
                meta_dir: str = None,
                ref_data_dir: str = None,
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
        super(GlobalDataset, self).__init__(
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
    
        self.emb_dir = emb_dir(data_dir)
        
        # if phase == 'train':
            
        #     self.global_data = {_id: self.load_emb(_id, emb_name='global', model_name=model_name, return_crds=True) \
        #         for _id in self.ids}
        #     self.pos_dict = {_id: data[1] \
        #         for _id, data in self.global_data.items()}
        #     self.global_embs = {_id: data[0] \
        #         for _id, data in self.global_data.items()}
        
        # if mode == 'inference':
        #     self.global_emb, self.position = self.load_emb(self.name, emb_name='global', return_crds=True)
            
        
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
            img_emb = self.load_emb(name, emb_name='global', idx=idx)
            
            adata = self.adata_dict[name]
            expression = adata[idx].X
            expression = expression.toarray().squeeze(0) \
                if sparse.issparse(expression) else expression.squeeze(0)
            
            data['img_emb'] = img_emb
            data['label'] = torch.FloatTensor(expression)
            data['pid'] = torch.LongTensor([i])
            data['sid'] = torch.LongTensor([idx])
            
        elif self.phase == 'test':
            if self.mode == 'inference':
                img = self.load_img(self.name, idx=index)
                img = self.transforms(img)
                # img = self.img[index]
                # img = self.transforms(img)
                img_emb = self.load_emb(self.name, emb_name='global', idx=index)
                # neighbor_emb, mask = self.load_emb(self.name, emb_name='neighbor', idx=index)
                # global_emb = self.load_emb(self.name, emb_name='global')
                # pos = np.load(f"{self.data_dir}/pos/{self.name}.npy")
                data['img_emb'] = img_emb
                data['sid'] = torch.LongTensor([index])
            else:
                name = self.int2id[index]
                img = self.load_img(name)
                img = torch.stack([self.transforms(im) for im in img], dim=0)
                
                # neighbor_emb, mask = self.load_emb(name, emb_name='neighbor')
                img_emb = self.load_emb(name, emb_name='global')
            
                if os.path.isfile(f"{self.st_dir}/{name}.h5ad"):
                    adata = self.load_st(name, self.genes, **self.norm_param)

                    expression = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
                    data['label'] = torch.FloatTensor(expression)

                data['img_emb'] = img_emb
                    
            
            # data['img'] = img
            # data['mask'] = mask
            # data['neighbor_emb'] = neighbor_emb
            
        return data
        
    # def load_emb(self, name: str, emb_name: str = 'global', idx: int = None, return_crds=False):
    #     if emb_name not in ['global', 'neighbor']:
    #         raise ValueError(f"emb_name must be 'global' or 'neighbor', but got {emb_name}")
        
    #     path = f"{self.emb_dir}/{emb_name}/features_{self.model_name}/{name}.h5"
        
    #     with h5py.File(path, 'r') as f:
    #         if 'embeddings'in f:
    #             emb = f['embeddings'][idx] if idx is not None else f['embeddings'][:]
    #         else:
    #             emb = f['features'][idx] if idx is not None else f['features'][:]
                
    #         emb = torch.Tensor(emb)
            
    #         if emb_name == 'neighbor':
    #             mask = f['mask_tb'][idx] if idx is not None else f['mask_tb'][:]
    #             mask = torch.LongTensor(mask)
    #             # return emb, mask
    #             if return_crds:
    #                 coords = f['coords'][idx] if idx is not None else f['coords'][:]
    #                 coords = torch.Tensor(coords)
    #                 return emb, mask, coords
    #             else:
    #                 return emb, mask
    #         else:
    #             if return_crds:
    #                 coords = f['coords'][idx] if idx is not None else f['coords'][:]
    #                 coords = torch.Tensor(coords)
    #                 return emb, coords
    #             else:
    #                 return emb
                

class StrideDataset(STDataset):
    def __init__(self, 
                mode: str,
                phase: str,
                fold: int,
                data_dir: str,
                meta_dir: str = None,
                wsi_dir: str = None,
                gene_type: str = 'mean',
                num_genes: int = 1000,
                num_outputs: int = 300,
                normalize: bool = True,
                cpm: bool = False,
                smooth: bool = False,
                data_id: str = None,
                model_name: str = 'uni_v2',
                load_level: str = 'patch'
                ):
        super(StrideDataset, self).__init__(
                                mode=mode,
                                phase=phase,
                                fold=fold,
                                data_dir=data_dir,
                                meta_dir=meta_dir,
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

        self.emb_dir = emb_dir(data_dir)
            
        
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
            
            spot_emb = self.load_emb(name, emb_name='global', idx=idx)
            sub_spot_emb = self.load_emb(name, emb_name='target', idx=idx)
            neighbor_emb, mask = self.load_emb(name, emb_name='neighbor', idx=idx)
            adata = self.adata_dict[name]
            expression = adata[idx].X
            expression = expression.toarray().squeeze(0) \
                if sparse.issparse(expression) else expression.squeeze(0)
            
                
            data['img'] = img
            data['mask'] = mask
            data['spot_emb'] = spot_emb
            data['sub_spot_emb'] = sub_spot_emb
            data['neighbor_emb'] = neighbor_emb
            data['label'] = torch.FloatTensor(expression)
            # data['pid'] = torch.LongTensor([i])
            # data['sid'] = torch.LongTensor([idx])
            
        elif self.phase == 'test':
            if self.mode == 'inference':
                # img = self.img[index]
                img = self.load_img(self.name, idx=index)
                img = self.transforms(img)
                spot_emb = self.load_emb(self.name, emb_name='global', idx=index)
                try:
                    sub_spot_emb = self.load_emb(self.name, emb_name='target', idx=index)
                except:
                    sub_spot_emb = spot_emb
                neighbor_emb, mask = self.load_emb(self.name, emb_name='neighbor', idx=index)
                # global_emb = self.load_emb(self.name, emb_name='global')
                # pos = np.load(f"{self.data_dir}/pos/{self.name}.npy")
                data['sid'] = torch.LongTensor([index])
            else:
                name = self.int2id[index]
                img = self.load_img(name)
                img = torch.stack([self.transforms(im) for im in img], dim=0)
                
                spot_emb = self.load_emb(name, emb_name='global')
                try:
                    sub_spot_emb = self.load_emb(name, emb_name='target')
                except:
                    sub_spot_emb = spot_emb
                neighbor_emb, mask = self.load_emb(name, emb_name='neighbor')
                # global_emb, pos = self.load_emb(name, emb_name='global', return_crds=True)
            
                if os.path.isfile(f"{self.st_dir}/{name}.h5ad"):
                    adata = self.load_st(name, self.genes, **self.norm_param)

                    expression = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
                    data['label'] = torch.FloatTensor(expression)

                # data['position'] = pos
                # data['global_emb'] = global_emb
                    
            
            data['img'] = img
            data['spot_emb'] = spot_emb
            data['sub_spot_emb'] = sub_spot_emb
            data['mask'] = mask
            data['neighbor_emb'] = neighbor_emb
            
        return data
        
    # def load_emb(self, name: str, emb_name: str = 'global', idx: int = None, return_crds=False):
    #     if emb_name not in ['global', 'neighbor', 'target']:
    #         raise ValueError(f"emb_name must be 'global' or 'neighbor' or 'target', but got {emb_name}")

    #     path = f"{self.emb_dir}/{emb_name}/features_{self.model_name}/{name}.h5"

    #     with h5py.File(path, 'r') as f:
    #         if 'embeddings'in f:
    #             emb = f['embeddings'][idx] if idx is not None else f['embeddings'][:]
    #         else:
    #             emb = f['features'][idx] if idx is not None else f['features'][:]
                
    #         emb = torch.Tensor(emb)
            
    #         if emb_name == 'neighbor':
    #             mask = f['mask_tb'][idx] if idx is not None else f['mask_tb'][:]
    #             mask = torch.LongTensor(mask)
    #             # return emb, mask
    #             if return_crds:
    #                 coords = f['coords'][idx] if idx is not None else f['coords'][:]
    #                 coords = torch.Tensor(coords)
    #                 return emb, mask, coords
    #             else:
    #                 return emb, mask
    #         else:
    #             if return_crds:
    #                 coords = f['coords'][idx] if idx is not None else f['coords'][:]
    #                 coords = torch.Tensor(coords)
    #                 return emb, coords
    #             else:
    #                 return emb
                
                
