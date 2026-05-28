from typing import List, Tuple, Union
from torch.backends import cudnn
from enum import Enum
import lightning as L
from torch import nn
import numpy as np
import random
import torch
import os
import torch.nn.functional as F


from .loss import (
    loss_cosine_function,
    loss_pearson_function,
    loss_mse_function,
    loss_poisson_function,
    loss_mse_pearson_function,
    loss_mse_cosine_function
)


class Operation(str, Enum):
    SUM = "sum"
    MAX = "max"
    NONE = "none"
    MEAN = "mean"


class Phi(nn.Module):
    def __init__(self, input_size: int, output_size: int, p: float = 0.0):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.Dropout(p=p),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Rho(nn.Module):
    def __init__(self, input_size: int, output_size: int, p: float = 0.0):
        super().__init__()
        self.model = nn.Sequential(
            nn.Dropout(p=p),
            nn.ReLU(inplace=True),
            nn.Linear(input_size, output_size),
            nn.Dropout(p=0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DeepSpot(nn.Module):
    
    def __init__(self,
                 emb_dim: int,
                 num_genes: int,
                 loss_func: str = "mse",
                 p: float = 0.3,
                 p_phi: Union[None, float] = None,
                 p_rho: Union[None, float] = None,
                 n_ensemble: int = 10,
                 n_ensemble_phi: Union[None, int] = None,
                 n_ensemble_rho: Union[None, int] = None,
                 phi2rho_size: int = 512,
                 scaler=None,
                 spot_context: str = 'spot_subspot_neighbors',
                 max_batch_size: int = 1024,
                 non_negative_output: bool = True):
        super().__init__()

        if emb_dim > 1536:
            self.spot_mapping = nn.Linear(emb_dim, 1536)
            self.sub_spot_mapping = nn.Linear(emb_dim, 1536)
            self.neighbor_mapping = nn.Linear(emb_dim, 1536)
            emb_dim = 1536

        self.scaler = scaler
        self.non_negative_output = non_negative_output

        if loss_func == "mse":
            self.loss_func = loss_mse_function
        elif loss_func == "cos":
            self.loss_func = loss_cosine_function
        elif loss_func == "mse_cos":
            self.loss_func = loss_mse_cosine_function
        elif loss_func == "pearson":
            self.loss_func = loss_pearson_function
        elif loss_func == "mse_pearson":
            self.loss_func = loss_mse_pearson_function
        elif loss_func == "poisson":
            self.loss_func = loss_poisson_function

        self.p_phi = p_phi or p
        self.p_rho = p_rho or p
        self.n_ensemble_phi = n_ensemble_phi or n_ensemble
        self.n_ensemble_rho = n_ensemble_rho or n_ensemble
        self.emb_dim = emb_dim

        self.training_loss = []
        self.validation_loss = []

        self.phi2rho_size = phi2rho_size
        self.phi_spot = nn.ModuleList([Phi(emb_dim, phi2rho_size, self.p_phi) for _ in range(self.n_ensemble_phi)])

        self.rho = nn.ModuleList([Rho(phi2rho_size * self._get_phi_multiplier(spot_context),
                                 num_genes, self.p_rho) for _ in range(self.n_ensemble_rho)])
        
        self.max_batch_size = max_batch_size

        self._forward_fn = self._get_forward_function(spot_context)

        
    def forward(self, spot_emb, sub_spot_emb=None, neighbor_emb=None, label=None, **kwargs):
        
        spot_emb = spot_emb.unsqueeze(1) if spot_emb.ndim == 2 else spot_emb
        spot_emb = self.spot_mapping(spot_emb) if getattr(self, 'spot_mapping', None) else spot_emb
    
        x = spot_emb
        if sub_spot_emb is not None:
            sub_spot_emb = self.sub_spot_mapping(sub_spot_emb) if getattr(self, 'sub_spot_mapping', None) else sub_spot_emb
            x = [spot_emb, sub_spot_emb]
        if neighbor_emb is not None:
            neighbor_emb = self.neighbor_mapping(neighbor_emb) if getattr(self, 'neighbor_mapping', None) else neighbor_emb
            if isinstance(x, list):
                x.append(neighbor_emb)
            else:
                x = [spot_emb, neighbor_emb]
        
        phase = kwargs.get('phase', 'train')
        
        if phase == 'train':
            output = self._forward_fn(x)
            
        else:
            if spot_emb.shape[0] > self.max_batch_size:
                if isinstance(x, list):
                    num_splits = len(x[0].split(self.max_batch_size, dim=0))
                    imgs = [img.split(self.max_batch_size, dim=0) for img in x]
                    
                    xs = []
                    for i in range(num_splits):
                        tmp = []
                        for j in range(len(imgs)):
                            tmp.append(imgs[j][i])
                        xs.append(tmp)
                            
                    output = [self._forward_fn(x) for x in xs]
                    output = torch.cat(output, dim=0)    
                else:
                    imgs = x.split(self.max_batch_size, dim=0)
                    output = [self._forward_fn(img) for img in imgs]
                    output = torch.cat(output, dim=0)
                    
            else:
                output = self._forward_fn(x)
        
        if self.non_negative_output:
            output = F.softplus(output)
        
        if label is not None:
            loss = self.loss_func(output, label)
            return {'loss': loss, 'logits': output}
        else:
            return {'logits': output}
        
        
    def _get_phi_multiplier(self, spot_context: str) -> int:
        context_multipliers = {
            'spot': 1,
            'spot_subspot': 2,
            'spot_neighbors': 2,
            'spot_subspot_neighbors': 3
        }
        return context_multipliers.get(spot_context, 1)

    def _get_forward_function(self, spot_context: str):
        forward_functions = {
            'spot': self._forward_spot,
            'spot_subspot': self._forward_spot_subspot,
            'spot_neighbors': self._forward_spot_neighbors,
            'spot_subspot_neighbors': self._forward_spot_subspot_neighbors,
        }
        return forward_functions.get(spot_context, self._forward_spot)

    def inverse_transform(self, x) -> torch.Tensor:
        if self.scaler is not None:
            x = self.scaler.inverse_transform(x)
        return x

    def _forward_spot(self, x: torch.Tensor) -> torch.Tensor:
        x_phi = self._apply_phi(x, self.phi_spot, operation=Operation.SUM)
        return self._apply_rho(x_phi)

    def _forward_spot_subspot(self, x: List[torch.Tensor]) -> torch.Tensor:
        x_spot, x_subspot = x
        x_subspot = self._apply_phi(x_subspot, self.phi_spot, operation=Operation.SUM)
        x_spot = self._apply_phi(x_spot, self.phi_spot)
        x_phi = torch.cat((x_spot, x_subspot), dim=1)
        return self._apply_rho(x_phi)

    def _forward_spot_neighbors(self, x: List[torch.Tensor]) -> torch.Tensor:
        x_spot, x_neighbors = x
        x_spot = self._apply_phi(x_spot, self.phi_spot)
        x_neighbors = self._apply_phi(x_neighbors, self.phi_spot, operation=Operation.MAX)
        x_phi = torch.cat((x_spot, x_neighbors), dim=1)
        return self._apply_rho(x_phi)

    def _forward_spot_subspot_neighbors(self, x: List[torch.Tensor]) -> torch.Tensor:
        x_spot, x_subspot, x_neighbors = x
        x_subspot = self._apply_phi(x_subspot, self.phi_spot, operation=Operation.SUM)
        x_spot = self._apply_phi(x_spot, self.phi_spot)
        x_neighbors = self._apply_phi(x_neighbors, self.phi_spot, operation=Operation.MAX)
        x_phi = torch.cat((x_spot, x_subspot, x_neighbors), dim=1)
        return self._apply_rho(x_phi)

    def _apply_phi(self, x: torch.Tensor, phi_modules: nn.ModuleList,
                   operation: Operation = Operation.NONE) -> torch.Tensor:
        batch_size = x.shape[0]
        sample_size = x.shape[1]
        x = x.view(-1, self.emb_dim)
        x_phi = torch.stack([phi(x) for phi in phi_modules], dim=1)
        x_phi, _ = torch.median(x_phi, dim=1)

        x_phi = x_phi.view(batch_size, sample_size, -1)

        if operation == Operation.SUM:
            x_phi = x_phi.sum(dim=1)
        elif operation == Operation.MEAN:
            x_phi = x_phi.mean(dim=1)
        elif operation == Operation.MAX:
            x_phi, _ = x_phi.max(dim=1)
        elif operation == Operation.NONE:
            x_phi = x_phi.view(batch_size, -1)

        return x_phi

    def _apply_rho(self, x_phi: torch.Tensor) -> torch.Tensor:
        x_rho = torch.stack([rho(x_phi) for rho in self.rho], dim=1)
        x_rho = torch.mean(x_rho, dim=1)
        return x_rho
