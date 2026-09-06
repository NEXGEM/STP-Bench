
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Path2Space(nn.Module):
    
    def __init__(self, num_genes=300, emb_dim=1024, max_batch_size=1024,
                 non_negative_output: bool = True):
        super(Path2Space, self).__init__()

        if emb_dim > 1536:
            self.mapping = nn.Linear(emb_dim, 1536)
            emb_dim = 1536

        self.max_batch_size = max_batch_size
        self.non_negative_output = non_negative_output
        
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, num_genes)
        )

    def forward(self, img_emb, label=None, **kwargs):
        phase = kwargs.get('phase', 'train')

        if getattr(self, 'mapping', None) is not None:
            img_emb = self.mapping(img_emb)

        if phase == 'train':
            output = self.mlp(img_emb)
            
        else:
            if img_emb.shape[0] > self.max_batch_size:
                imgs = img_emb.split(self.max_batch_size, dim=0)
                output = [self.mlp(img) for img in imgs]
                output = torch.cat(output, dim=0)
            else:
                output = self.mlp(img_emb)
        
        # output = torch.clamp(output, 0) 
        if self.non_negative_output:
            output = F.softplus(output)
        
        if label is not None:
            loss = F.mse_loss(output, label)
            return {'loss': loss, 'logits': output}
        else:
            return {'logits': output}
