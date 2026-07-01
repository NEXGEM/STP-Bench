
import os

import json
import numpy as np
from scipy import sparse
import h5py
import torch

from dataset.base_dataset import STDataset
from dataset.path_utils import emb_dir as resolve_emb_dir
from dataset.path_utils import st_dir as resolve_st_dir


class BleepDataset(STDataset):
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
                ref_data_dir: str = None,
                ref_asset_dir: str = None,
                load_level: str = 'patch', # 'patch' or 'slide',
                ids: list = None
                ):
        super(BleepDataset, self).__init__(
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
                                load_level=load_level)

        # Load reference DB when: non-CV mode (eval/inference), or CV test phase
        # (val inference during training needs the reference expression bank).
        if mode != 'cv' or phase == 'test':

            if ref_data_dir is not None:
                ref_asset_dir = ref_asset_dir or ref_data_dir
                self.ids_ref = self._get_ids(phase='train', fold=fold, ids_dir=ref_data_dir)

                if not os.path.isfile(f"{ref_data_dir}/{gene_type}_{num_genes}genes.json"):
                    raise ValueError(f"{gene_type}_{num_genes}genes.json is not found in {ref_data_dir}")

                with open(f"{ref_data_dir}/{gene_type}_{num_genes}genes.json", 'r') as f:
                    genes = json.load(f)['genes']
                if gene_type in ['mean', 'hmhvg', 'all']:
                    self.genes = genes[:num_outputs]
                else:
                    self.genes = genes
            else:
                self.ids_ref = self._get_ids(phase='train', fold=fold)

            spot_expressions_ref = []
            positions_ref = []
            
            st_dir = resolve_st_dir(ref_asset_dir) if ref_data_dir is not None else None
            emb_dir = resolve_emb_dir(ref_asset_dir) if ref_data_dir is not None else None
            
            for _id in self.ids_ref:
                expression = self.load_st(_id, self.genes, st_dir=st_dir, **self.norm_param).X
                expression = expression.toarray() if sparse.issparse(expression) else expression
                expression = torch.FloatTensor(expression)
                spot_expressions_ref.append(expression)
                
                _, pos = self.load_emb(_id, emb_name='global', emb_dir=emb_dir, return_crds=True)
                grid_pos = self.get_normalized_pos(pos, rounding_factor=20)
                positions_ref.append(grid_pos)
                
            self.spot_expressions_ref = torch.cat(spot_expressions_ref, dim=0) 
            self.positions_ref = torch.cat(positions_ref, dim=0)
            
            self.ref_data_dir = ref_data_dir if ref_data_dir is not None else data_dir
            self.ref_asset_dir = ref_asset_dir if ref_data_dir is not None else data_dir
            self.fold = fold

        if mode == 'inference':
            self.ids = ids
            self.int2id = dict(enumerate(self.ids))
            self.id2int = {v:k for k,v in self.int2id.items()}
            # self.pid = 0
            
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
            
            img_emb, pos = self.load_emb(name, emb_name='global', return_crds=True)
            img_emb = img_emb[idx]
            grid_pos = self.get_normalized_pos(pos, rounding_factor=20)
            grid_pos = grid_pos[idx]
            
            adata = self.adata_dict[name]
            expression = adata[idx].X
            expression = expression.toarray().squeeze(0) \
                if sparse.issparse(expression) else expression.squeeze(0)
            
            data['img'] = img
            data['img_emb'] = img_emb
            data['position'] = grid_pos
            data['label'] = torch.FloatTensor(expression)
            
        elif self.phase == 'test':
            if self.mode == 'inference':
                # img = self.img[index]
                img = self.load_img(self.name, idx=index)
                img = self.transforms(img)
                # neighbor_emb, mask = self.load_emb(self.name, emb_name='neighbor', idx=index)
                img_emb, pos = self.load_emb(self.name, emb_name='global', return_crds=True)
                img_emb = img_emb[index]
                grid_pos = self.get_normalized_pos(pos, rounding_factor=20)
                grid_pos = grid_pos[index]
                # pos = np.load(f"{self.data_dir}/pos/{self.name}.npy")
                
                if getattr(self, 'current_name', None) is None:
                    self.current_name = self.name

                if self.name != self.current_name:
                    self.pid += 1
                    self.current_name = self.name
                # data['pid'] = torch.LongTensor([self.pid])
                data['pid'] = torch.LongTensor([self.id2int[self.name]])
                # data['sid'] = torch.LongTensor([index])
                
            else:
                name = self.int2id[index]
                img = self.load_img(name)
                img = torch.stack([self.transforms(im) for im in img], dim=0)
                
                img_emb, pos = self.load_emb(name, emb_name='global', return_crds=True)
                grid_pos = self.get_normalized_pos(pos, rounding_factor=20)
            
                if os.path.isfile(f"{self.st_dir}/{name}.h5ad"):
                    adata = self.load_st(name, self.genes, **self.norm_param)
                    expression = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
                    data['label'] = torch.FloatTensor(expression)
                
                data['pid'] = torch.LongTensor([index])
                        
            data['img'] = img
            data['img_emb'] = img_emb
            data['position'] = grid_pos
            
            
        return data
    
    def get_normalized_pos(self, pos, rounding_factor=None):
        W,H = self.infer_grid_size(pos, rounding_factor=rounding_factor)

        pos_min = pos.min(dim=0, keepdim=True)[0]
        pos_max = pos.max(dim=0, keepdim=True)[0]
        pos_norm = (pos - pos_min) / (pos_max - pos_min + 1e-5)

        grid_pos = pos_norm * torch.tensor([W - 1, H - 1])
        grid_pos = grid_pos.round().long()
        
        return grid_pos

    def infer_grid_size(self, pos, rounding_factor=None):
        """
        pos: (N, 2) tensor
        """
        if rounding_factor is None:
            rounding_factor = self.dynamic_rounding_factor(pos)
        
        pos_rounded = (pos / rounding_factor).round() * rounding_factor
        unique_x = torch.unique(pos_rounded[:, 0])
        unique_y = torch.unique(pos_rounded[:, 1])
        W = unique_x.numel()
        H = unique_y.numel()
        return (W, H)
