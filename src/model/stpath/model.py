
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))
sys.path.append(f"{CURRENT_DIR}/STPath")
from stpath.app.pipeline.inference import STPathInference
# import STPathInference


def _resolve_stpath_file(path, default_relative_path, label):
    path = path or default_relative_path
    path = os.path.expanduser(path)
    candidates = []
    if os.path.isabs(path):
        candidates.append(path)
    else:
        candidates.append(os.path.join(CURRENT_DIR, path))
        candidates.append(os.path.join(REPO_ROOT, path))

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        f"STPath {label} file is not found. Checked: {candidates}. "
        "Relative paths are resolved from src/model/stpath. "
        "For the pretrained STPath weight, download tlhuang/STPath stfm.pth "
        "and place it at src/model/stpath/weight/stfm.pth, or set MODEL.model_weight_path."
    )


class STPathModule(nn.Module):
    
    def __init__(self, gene_path=None, gene_voc_path=None, model_weight_path=None, device='cpu'):
        super(STPathModule, self).__init__()
        gene_voc_path = _resolve_stpath_file(
            gene_voc_path,
            os.path.join("STPath", "utils_data", "symbol2ensembl.json"),
            "gene vocabulary",
        )
        model_weight_path = _resolve_stpath_file(
            model_weight_path,
            os.path.join("weight", "stfm.pth"),
            "model weight",
        )

        self.agent = STPathInference(
            gene_voc_path=gene_voc_path,
            model_weight_path=model_weight_path,
            device=device,
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
