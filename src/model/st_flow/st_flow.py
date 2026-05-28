
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model.denoiser import Denoiser
from .flow.interpolant import Interpolant


class StFlow(nn.Module):
    def __init__(self, 
                num_genes=200,
                feature_dim=1024,
                hidden_dim=128,
                pairwise_hidden_dim=128,
                n_layers=4,
                n_heads=4,
                dropout=0.2,
                attn_dropout=0.2,
                n_neighbors=8,
                activation='swiglu',
                prior_sampler='zinb',
                zinb_total_count=1,
                zinb_logits=0.1,
                zinb_zi_logits=0.,
                n_sample_steps=5,
                max_batch_size=1024):
        super(StFlow, self).__init__()

        if feature_dim > 1536:
            self.mapping = nn.Linear(feature_dim, 1536)
            feature_dim = 1536
        
        self.num_genes = num_genes
        self.n_sample_steps = n_sample_steps
        self.max_batch_size = max_batch_size
        
        # Initialize the flow model
        self.diffusier = Interpolant(
            prior_sampler, 
            total_count=torch.tensor([zinb_total_count]),
            logits=torch.tensor([zinb_logits]),
            zi_logits=zinb_zi_logits,
            normalize=prior_sampler != "gaussian",
        )

        # Initialize the denoiser
        self.model = Denoiser(
                num_genes,
                feature_dim,
                hidden_dim,
                pairwise_hidden_dim,
                n_layers,
                n_heads,
                dropout,
                attn_dropout,
                n_neighbors,
                activation,
            )

        
    def _forward(self, img_features, coords, label):
        
        # Forward pass through the flow model and denoiser
        noisy_exp, t_steps = self.diffusier.corrupt_exp(label)
        
        pred_exp, loss = self.model(
            exp=noisy_exp, 
            img_features=img_features, 
            coords=coords, 
            labels=label, 
            t_steps=t_steps
        )
        
        return pred_exp, loss

    def forward(self, img_features, coords, label=None, **kwargs):
        phase = kwargs.get('phase', 'test')
        assert phase in ['train', 'val', 'test'], "phase must be either 'train', 'val', or 'test'"
        
        if getattr(self, 'mapping', None) is not None:
            img_features = self.mapping(img_features)

        if label is not None and len(label.shape) == 2:
            label = label.unsqueeze(0)
            
        if phase in ('test', 'val'):
            predictions = self.sample(img_features, coords).squeeze(0)

            result_dict = {
                "logits": torch.clamp(predictions, min=0)
            }

        else:
            _, loss = self._forward(img_features, coords, label)

            result_dict = {
                "loss": loss,
            }
            
        return result_dict
    
    def sample(self, img_features, coords):
        assert img_features.shape[0] == 1, "Batch size must be 1 for inference"

        exp_t1 = self.diffusier.sample_from_prior((img_features.shape[0], img_features.shape[1], self.num_genes))
        ts = torch.linspace(
            0.01, 1.0, self.n_sample_steps, device=img_features.device
        )[:, None].expand(self.n_sample_steps, exp_t1.shape[0])

        for step, (t1, t2) in enumerate(zip(ts[:-1], ts[1:])):
            pred = self.model.inference(
                exp_t1, img_features, coords, 
                t1, predict=True
            )
            d_t = t2 - t1

            if step == self.n_sample_steps - 2:
                break
            else:
                exp_t1 = self.diffusier.denoise(pred, exp_t1, t1, d_t)

        return pred
        
        
    
