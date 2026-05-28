
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{CURRENT_DIR}/STPath")
from stpath.app.pipeline.inference import STPathInference
# import STPathInference


class STPathModule(nn.Module):
    
    def __init__(self, gene_path=None):
        super(STPathModule, self).__init__()

        self.agent = STPathInference(
            gene_voc_path='./src/model/stpath/STPath/utils_data/symbol2ensembl.json',
            model_weight_path='./src/model/stpath/weight/stfm.pth', 
            device='cpu'
        )

    def forward(self, img_emb, coord, organ_idx, tech_idx, label=None, **kwargs):
        device = kwargs.get('device', 'cuda')
        dataset = kwargs.get('dataset', None)
        if dataset is None:
            raise ValueError("Please provide dataset in kwargs for organ and tech vocabularies.")
        
        organ_voc = dataset.organ_voc
        tech_voc = dataset.tech_voc
        genes = dataset.genes
        
        organ = organ_voc[organ_idx]
        tech = tech_voc[tech_idx]

        coord = coord.cpu().numpy()
        img_emb = img_emb.cpu().numpy()
        
        pred_adata = self.agent.inference(
            coords=coord, # [number_of_spots, 2]
            img_features=img_emb,  # [number_of_spots, 1536], the image features extracted using Gigapath
            organ_type=organ,  # Default is None
            tech_type=tech,  # Default is None
            save_gene_names=genes  # a list of gene names to save in the adata, e.g., ['GATA3', 'UBLE2C', ...]. None will save all genes in the model.
        )
        
        output = torch.clamp(torch.Tensor(pred_adata.X).to(device), min=0)

        if label is not None:
            loss = F.mse_loss(output, label)
            return {'loss': loss, 'logits': output}
        else:
            return {'logits': output}
