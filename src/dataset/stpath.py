
import os

import json
import numpy as np
from scipy import sparse
import h5py
import torch

from dataset.base_dataset import STDataset


class STPathDataset(STDataset):
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
                load_level: str = 'patch', # 'patch' or 'slide',
                organ: str = None,
                tech: str = None
                ):
        super(STPathDataset, self).__init__(
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
                                load_level=load_level)
        if data_id == 'TENX138': # Temporary - Since tech is mixed in this datasets
            tech = 'Xenium'
            
        self.organ_voc = ["<pad>", "<mask>", "<unk>", "Spinal cord", "Brain", "Breast", "Bowel", "Skin", "Heart", "Kidney", "Prostate", 
             "Lung", "Liver", "Uterus", "Bone", "Muscle", "Eye", "Pancreas", "Mouth", "Ovary", "Glioma", "Glioblastoma",
             "Stomach", "Colon", "Others"]
        
        organ2idx = {v:k for k,v in enumerate(self.organ_voc)}
        self.organ_idx = organ2idx[organ] if organ in organ2idx else 0
        self.tech_voc = ["<pad>", "Spatial Transcriptomics", "Visium", "Xenium", "Visium HD"]
        self.tech2idx = {v:k for k,v in enumerate(self.tech_voc)}
        self.tech_idx = self.tech2idx[tech] if tech in self.tech2idx else 0
            
    def __getitem__(self, index):
        data = {}
        
        if self.mode == 'inference':
            # img = self.img[index]
            # img = self.load_img(self.name, idx=index)
            # img = self.transforms(img)
            # neighbor_emb, mask = self.load_emb(self.name, emb_name='neighbor', idx=index)
            img_emb, pos = self.load_emb(self.name, emb_name='global', return_crds=True)
            # img_emb = img_emb[index]
            
        else:
            name = self.int2id[index]
            # img = self.load_img(name)
            # img = torch.stack([self.transforms(im) for im in img], dim=0)
            img_emb, pos = self.load_emb(name, emb_name='global', return_crds=True)
            

        
            if os.path.isfile(f"{self.st_dir}/{name}.h5ad"):
                adata = self.load_st(name, self.genes, **self.norm_param)
                expression = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
                data['label'] = torch.FloatTensor(expression)
                
        # data['img'] = img
        data['img_emb'] = img_emb
        data['coord'] = pos
        data['organ_idx'] = self.organ_idx
        data['tech_idx'] = self.tech_idx

        return data