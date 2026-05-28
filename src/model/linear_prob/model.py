
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models      


class LinearProb(nn.Module):
    
    def __init__(self, 
        num_genes=200, 
        img_embedding_dim=1536,
        max_batch_size=1024,
        non_negative_output: bool = True
        ):

        super(LinearProb, self).__init__()

        self.non_negative_output = non_negative_output

        self.max_batch_size = max_batch_size
        self.linear = nn.Linear(img_embedding_dim, num_genes)
            # self.mlp = nn.Sequential(
            #     nn.Linear(img_embedding_dim, 512),
            #     nn.ReLU(),
            #     nn.Linear(512, num_genes)
            # )

    def forward(self, img=None, label=None, **kwargs):
        phase = kwargs.get('phase', 'train')
        return_emb = kwargs.get('return_emb', False)

        
        img = kwargs.get('img_emb', None)
        if img is None:
            raise ValueError("Image embeddings must be provided for LinearProb.")
    
        if phase == 'train':
            output = self.linear(img)
            
        else:
            if img.shape[0] > self.max_batch_size:
                imgs = img.split(self.max_batch_size, dim=0)
                if return_emb:
                    embs = img
                    output = [self.linear(im) for im in imgs]
                    output = torch.cat(output, dim=0)
                    
                else:
                    output = [self.linear(im) for im in imgs]
                    output = torch.cat(output, dim=0)
                
            else:
                embs = img
                output = self.linear(img)
        
        # output = torch.clamp(output, 0) 
        if self.non_negative_output:
            output = F.softplus(output)  # Ensure non-negativity
        
        if label is not None:
            loss = F.mse_loss(output, label)
            result_dict = {'loss': loss, 'logits': output}
        else:
            result_dict = {'logits': output}
        
        if return_emb:
            result_dict['embeddings'] = embs
        return result_dict
        
        
        
        
        