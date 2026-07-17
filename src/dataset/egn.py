
import os
from glob import glob
import json
import numpy as np
from scipy import sparse
import h5py
import torch

from dataset.base_dataset import STDataset
from dataset.path_utils import emb_dir as resolve_emb_dir
from dataset.path_utils import st_dir as resolve_st_dir


class EGNDataset(STDataset):
    def __init__(self,
                mode: str,
                phase: str,
                fold: int,
                data_dir: str,
                meta_dir: str = None,
                wsi_dir: str = None,
                asset_dir: str = None,
                distance_metric: str = 'l1',
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
                genes_override: list = None,
                load_level: str = 'patch'
                ):
        # Treat same-directory ref as no-ref: internal evaluation always sets
        # ref_data_dir == data_dir, but the super().__init__ must see None so
        # STDataset loads only the fold-specific test split (not all samples).
        if ref_data_dir is not None and os.path.normpath(ref_data_dir) == os.path.normpath(data_dir):
            ref_data_dir = None

        super(EGNDataset, self).__init__(
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

        self.num_outputs = num_outputs
        self.asset_dir = asset_dir or self.asset_dir
        self.img_dir = f"{self.asset_dir}/patches"
        self.st_dir = resolve_st_dir(self.asset_dir)
        self.emb_dir = f"{self.asset_dir}/emb"

        # self.exemplar_dir = f"{data_dir}/exemplar/{model_name}/{distance_metric}/fold{fold}/{phase}"
        self.model_name = model_name
        self.ref_data_dir = ref_data_dir
        ref_asset_dir = ref_asset_dir or ref_data_dir

        if ref_data_dir is not None:
            ref_data = '/'.join(ref_data_dir.replace('/bench_data', '').split('/')[-2:])
            self.exemplar_dir = f"{self.asset_dir}/exemplar/{model_name}/{distance_metric}/{ref_data}/fold{fold}/{phase}"
            
            ids_ref = self._get_ids(phase='train', fold=fold, ids_dir=ref_data_dir)

            # The reference bank (spot_expressions_ref/global_embs_ref) is
            # built from TRAINING data — always use the full training gene
            # panel for it, never genes_override, which only restricts the
            # current (external) sample's own label via self.genes.
            if not os.path.isfile(f"{ref_data_dir}/{gene_type}_{num_genes}genes.json"):
                raise ValueError(f"{gene_type}_{num_genes}genes.json is not found in {ref_data_dir}")

            with open(f"{ref_data_dir}/{gene_type}_{num_genes}genes.json", 'r') as f:
                genes = json.load(f)['genes']
            ref_genes = genes[:num_genes] if gene_type in ['mean', 'hmhvg', 'all'] else genes

            self.genes = list(genes_override) if genes_override is not None else ref_genes
            # get_exemplars_batch pre-allocates exp_exemplars using
            # self.num_outputs, and exemplar values come from
            # spot_expressions_ref (built from ref_genes above, the full
            # training panel) — not from self.genes, which may be narrower
            # for external evaluation.
            self.num_outputs = len(ref_genes)

            ref_emb_dir = resolve_emb_dir(ref_asset_dir)
            ref_st_dir = resolve_st_dir(ref_asset_dir)

            adata_dict = {_id: self.load_st(_id, ref_genes, st_dir=ref_st_dir, **self.norm_param)
                for _id in ids_ref}
            self.spot_expressions_ref = {_id: adata.X.toarray() if sparse.issparse(adata.X) else adata.X
                for _id, adata in adata_dict.items()}
            self.global_embs_ref = {_id: self.load_emb(_id, emb_dir=ref_emb_dir)
                for _id in ids_ref}

        else:
            self.exemplar_dir = f"{self.asset_dir}/exemplar/{model_name}/{distance_metric}/fold{fold}/{phase}"

            ids_ref = self._get_ids(phase='train', fold=fold)

            adata_dict = {_id: self.load_st(_id, self.genes, **self.norm_param)
                for _id in ids_ref}
            self.spot_expressions_ref = {_id: adata.X.toarray() if sparse.issparse(adata.X) else adata.X
                for _id, adata in adata_dict.items()}
            self.global_embs_ref = {_id: self.load_emb(_id)
                for _id in ids_ref}
    
    def __getitem__(self, index):
        data = {}

        if self.load_level == 'patch':
            
            if self.mode == 'inference':
                # img = self.img[index]
                img = self.load_img(self.name, idx=index)
                img = self.transforms(img)
                
                # img_emb = self.load_emb(self.name, idx=index, emb_name='global')
                
                global_emb, coord = self.load_emb(self.name, idx=index, return_crds=True)
                # global_embs = global_embs.unsqueeze(0)
                
                window = img.shape[-1]
                pos = coord // window
                
                with h5py.File(f"{self.exemplar_dir}/{self.name}.h5", 'r') as f:
                    pid = f['pid'][:].astype('str')
                    sid = f['sid'][:]
                    
                pid_i = pid[index]
                sid_i = sid[index]
                img_exemplars, exp_exemplars = self.get_exemplars(pid_i, sid_i)
                
            else:
                i = 0
                while index >= self.cumlen[i]:
                    i += 1
                idx = index
                if i > 0:
                    idx = index - self.cumlen[i-1]

                name = self.int2id[i]
                img = self.load_img(name, idx)
                img = self.transforms(img)
                
                adata = self.adata_dict[name]
                expression = adata[idx].X
                expression = expression.toarray().squeeze(0) \
                    if sparse.issparse(expression) else expression.squeeze(0)
                
                # global_embs = self.global_embs[name]
                global_emb = self.load_emb(name, idx=idx)
                
                with h5py.File(f"{self.exemplar_dir}/{name}.h5", 'r') as f:
                    pid = f['pid'][:].astype('str')
                    sid = f['sid'][:]
                
                pid_i = pid[idx]
                sid_i = sid[idx]
                img_exemplars, exp_exemplars = self.get_exemplars(pid_i, sid_i)
                
                data['label'] = torch.FloatTensor(expression) 
            
            data['img'] = img
            data['ei'] = global_emb.unsqueeze(0)
            data['ej'] = img_exemplars
            data['yj'] = exp_exemplars
            
        elif self.load_level == 'slide':
            
            if self.mode == 'inference':
                # img = self.img[index]
                img = self.load_img(self.name, idx=index)
                img = self.transforms(img)
                
                # img_emb = self.load_emb(self.name, idx=index, emb_name='global')
                
                global_embs, coords = self.load_emb(self.name, return_crds=True)
                # global_embs = global_embs.unsqueeze(0)
                
                window = img.shape[-1]
                pos = coords // window
                
                with h5py.File(f"{self.exemplar_dir}/{self.name}.h5", 'r') as f:
                    pid = f['pid'][:].astype('str')
                    sid = f['sid'][:]
                    
                # pid_i = pid[index]
                # sid_i = sid[index]
                # img_exemplars, exp_exemplars = self.get_exemplars(pid_i, sid_i)
                # img_exemplars, exp_exemplars = self.get_exemplars_batch(pid, sid, D_img=global_embs.shape[-1])
                img_exemplars, exp_exemplars = self.get_exemplars_batch(pid, sid, D_img=global_embs.shape[-1])
                global_embs = global_embs.unsqueeze(1)
            
            else:
                name = self.int2id[index]
                img = self.load_img(name)
                img = torch.stack([self.transforms(im) for im in img], dim=0)
                
                if os.path.isfile(f"{self.st_dir}/{name}.h5ad"):
                    adata = self.load_st(name, self.genes, **self.norm_param)
                    expression = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
                    data['label'] = torch.FloatTensor(expression)
                
                global_embs, coords = self.load_emb(name, return_crds=True)
                window = img.shape[-1]
                pos = coords // window
                
                with h5py.File(f"{self.exemplar_dir}/{name}.h5", 'r') as f:
                    pid = f['pid'][:].astype('str')
                    sid = f['sid'][:]
                img_exemplars, exp_exemplars = self.get_exemplars_batch(pid, sid, D_img=global_embs.shape[-1])
                global_embs = global_embs.unsqueeze(1)
            
            data['img'] = img
            data['ei'] = global_embs
            data['ej'] = img_exemplars
            data['yj'] = exp_exemplars
            data['pos'] = pos.long()
            
        return data
    
    def get_exemplars(self, pid_i, sid_i, num_exemplars=9):
        
        pid_i = pid_i[:num_exemplars]
        sid_i = sid_i[:num_exemplars]
        
        # Retrieve and assign embeddings
        img_exemplars = np.array([self.global_embs_ref[p][s] for p, s in zip(pid_i, sid_i)], dtype=np.float32)
        exp_exemplars = np.array([self.spot_expressions_ref[p][s] for p, s in zip(pid_i, sid_i)], dtype=np.float32)

        return img_exemplars, exp_exemplars
    
    def get_exemplars_batch(self, pid, sid, num_exemplars=9, D_img=1024):
        """
        Optimized extraction of image and expression exemplars.

        Parameters:
        - pid (np.ndarray): Array of participant IDs with shape (batch_size, 100)
        - sid (np.ndarray): Array of session IDs with shape (batch_size, 100)

        Returns:
        - img_exemplars (np.ndarray): Stacked image embeddings with shape (batch_size, 9, D_img)
        - exp_exemplars (np.ndarray): Stacked expression embeddings with shape (batch_size, 9, D_exp)
        """
        batch_size = pid.shape[0] 

        # Preallocate arrays
        img_exemplars = np.empty((batch_size, num_exemplars, D_img), dtype=np.float32)
        exp_exemplars = np.empty((batch_size, num_exemplars, self.num_outputs), dtype=np.float32)

        for i in range(batch_size):
            pid_i = pid[i]
            sid_i = sid[i]
            
            img_exemplar, exp_exemplar = self.get_exemplars(pid_i, sid_i, num_exemplars)
            
            img_exemplars[i] = img_exemplar
            exp_exemplars[i] = exp_exemplar

        return img_exemplars, exp_exemplars
    
    
class EGGNDataset(STDataset):
    def __init__(self,
                mode: str,
                phase: str,
                fold: int,
                data_dir: str,
                meta_dir: str = None,
                ref_data_dir: str = None,
                genes_override: list = None,
                gene_type: str = 'mean',
                num_genes: int = 1000,
                num_outputs: int = 300,
                normalize: bool = True,
                cpm: bool = False,
                smooth: bool = False,
                data_id: str = None,
                model_name: str = 'uni_v2',
                load_level: str = 'slide'
                ):
        super(EGGNDataset, self).__init__(
                                mode=mode,
                                phase=phase,
                                fold=fold,
                                data_dir=data_dir,
                                meta_dir=meta_dir,
                                ref_data_dir=ref_data_dir,
                                genes_override=genes_override,
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
        
        if cpm:
            if ref_data_dir is not None:
                ref_data = '/'.join(ref_data_dir.replace('/bench_data', '').split('/')[-2:])
                all_files = glob(f"{self.asset_dir}/EGGN/cpm/{model_name}/{ref_data}/fold{fold}/{phase}/graph_6/*.pt")
            else:
                all_files = glob(f"{self.asset_dir}/EGGN/cpm/{model_name}/fold{fold}/{phase}/graph_6/*.pt")
        else:
            if ref_data_dir is not None:
                ref_data = '/'.join(ref_data_dir.replace('/bench_data', '').split('/')[-2:])
                all_files = glob(f"{self.asset_dir}/EGGN/{model_name}/{ref_data}/fold{fold}/{phase}/graph_6/*.pt")
            else:
                all_files = glob(f"{self.asset_dir}/EGGN/{model_name}/fold{fold}/{phase}/graph_6/*.pt")   
        
        self.selected_files = {}
        for i in all_files:
            name = i.split('/')[-1].split('.')[0]
            graph = torch.load(i)
            self.selected_files[name] = graph
            # self.selected_files.append(graph)
        
    def __getitem__(self, index):
        name = self.int2id[index] if self.mode != 'inference' else self.name
        return self.selected_files[name]
    
    
class SGNDataset(STDataset):
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
                load_level: str = 'slide'
                ):
        super(SGNDataset, self).__init__(
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

        if cpm:
            if ref_data_dir is not None:
                ref_data = '/'.join(ref_data_dir.replace('/bench_data', '').split('/')[-2:])
                all_files = glob(f"{self.asset_dir}/SGN/cpm/{model_name}/{ref_data}/fold{fold}/{phase}/graph/*.pt")
            else:
                all_files = glob(f"{self.asset_dir}/SGN/cpm/{model_name}/fold{fold}/{phase}/graph/*.pt")
        else:
            if ref_data_dir is not None:
                ref_data = '/'.join(ref_data_dir.replace('/bench_data', '').split('/')[-2:])
                all_files = glob(f"{self.asset_dir}/SGN/{model_name}/{ref_data}/fold{fold}/{phase}/graph/*.pt")
            else:
                all_files = glob(f"{self.asset_dir}/SGN/{model_name}/fold{fold}/{phase}/graph/*.pt")   

        self.selected_files = {}
        for i in all_files:
            name = i.split('/')[-1].split('.')[0]
            graph = torch.load(i)
            self.selected_files[name] = graph
            # self.selected_files.append(graph)
        
    def __getitem__(self, index):
        name = self.int2id[index] if self.mode != 'inference' else self.name
        return self.selected_files[name]   
