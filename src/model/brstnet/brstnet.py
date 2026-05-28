from glob import glob
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from typing import Optional


class AuxNet(nn.Module):
    def __init__(self, in_features: int, main_outputs: int, aux_outputs: int):
        super().__init__()
        
        self.fc_main = nn.Linear(in_features, main_outputs)
        self.fc_aux  = nn.Linear(in_features, aux_outputs)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y    = self.fc_main(x)
        auxy = self.fc_aux(x)
        return y, auxy
    

class BrSTNet(nn.Module):
    def __init__(
        self,
        backbone: str = 'efficientnet_b4',
        pretrained: bool = True,
        num_genes: int = 50,
        data_dir: str = None,
        ref_data_dir: str = None,
        aux_weight: float = 1.0,
        image_embedding_dim = 1024,
        use_pretrained_emb=False,
        max_batch_size: int = 1024,
        non_negative_output: bool = True
    ):
        super().__init__()
        
        if ref_data_dir is None:
            total_gene_paths = glob(f"{data_dir}/total_*.json")
        else:
            total_gene_paths = glob(f"{ref_data_dir}/total_*.json")
            data_dir = ref_data_dir
        
        if len(total_gene_paths) == 0:
            raise ValueError(f"No total genes file found in {data_dir}.")
        elif len(total_gene_paths) > 1:
            raise ValueError(f"Multiple total genes files found in {data_dir}: {total_gene_paths}")
            
        total_gene_path = total_gene_paths[0]
        with open(total_gene_path, 'r') as f:
            total_genes = json.load(f)['genes']
        
        aux_outputs = len(total_genes) - num_genes
        
        self.use_pretrained_emb = use_pretrained_emb
        self.max_batch_size = max_batch_size
        self.non_negative_output = non_negative_output

        if not use_pretrained_emb:
            if backbone not in torchvision.models.__dict__:
                raise ValueError(f"Unknown backbone: {backbone}")
            full_backbone = torchvision.models.__dict__[backbone](pretrained=pretrained)
            self.backbone = full_backbone

            if hasattr(self.backbone, 'classifier'):
                in_features = self.backbone.classifier[1].in_features
                self.backbone.classifier = nn.Identity()
            elif hasattr(self.backbone, 'fc'):
                in_features = self.backbone.fc.in_features
                self.backbone.fc = nn.Identity()
            else:
                raise ValueError("classifier/fc cannot be found in Backbone.")
        else:
            if image_embedding_dim > 1536:
                self.mapping = nn.Linear(image_embedding_dim, 1536)
                image_embedding_dim = 1536
                
            in_features = image_embedding_dim
        
        self.head_aux = AuxNet(
            in_features= in_features, 
            main_outputs = num_genes,
            aux_outputs  = aux_outputs
        )
        self.aux_weight = aux_weight

    def forward(self, img: torch.Tensor, label: torch.Tensor = None, aux: torch.Tensor = None, **kwargs) -> dict:
        """
        img: (N, C, H, W)  N = #patches (train:1, test: thousands)
        """
        
        phase = kwargs.get('phase', 'train')
        
        if self.use_pretrained_emb:
            img = kwargs.get('img_emb', None)
            if img is None:
                raise ValueError("Image embeddings must be provided when use_pretrained_emb is True.")
        
        if phase == 'train':
            # output = self.model(img)
            if self.use_pretrained_emb:
                features = img
            else:
                features = self.backbone(img)           # (N, C, H, W) → (N, D_feat,…)
            logits, aux_logits = self.head_aux(features)
            
        else:
            if img.shape[0] > self.max_batch_size:
                imgs = img.split(self.max_batch_size, dim=0)
                logits, aux_logits = [], []
                for _img in imgs:
                    if self.use_pretrained_emb:
                        features = _img
                    else:
                        features = self.backbone(_img)           # (N, C, H, W) → (N, D_feat,…)
                    logit, aux_logit = self.head_aux(features)
                    logits.append(logit)
                    aux_logits.append(aux_logit)
                logits = torch.cat(logits, dim=0)
                aux_logits = torch.cat(aux_logits, dim=0)
            else:
                if self.use_pretrained_emb:
                    features = img
                else:
                    features = self.backbone(img)           # (N, C, H, W) → (N, D_feat,…)
                logits, aux_logits = self.head_aux(features)

        # logits = torch.clamp(logits, 0) 
        if self.non_negative_output:
            logits = F.softplus(logits)
        
        result_dict = {'logits': logits, 'aux_logits': aux_logits}

        if label is not None:
            loss_main = F.mse_loss(logits, label)
            aux = aux.squeeze(0) if len(aux.shape) == 3 else aux
            loss_aux  = F.mse_loss(aux_logits, aux)
            result_dict['aux_loss'] = loss_aux
            result_dict['loss']     = loss_main + self.aux_weight * loss_aux

        return result_dict