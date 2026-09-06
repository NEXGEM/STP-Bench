import os
import pickle
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch_geometric.nn as pyg
from typing import Optional, Type, List
from timm.layers.helpers import to_2tuple    
from functools import partial
from timm.layers import DropPath
from timm.models.vision_transformer import init_weights_vit_timm
from timm.models._manipulate import named_apply
import torch.nn.functional as F

from .preprocess.extra import EXTRA

def decompose(x):
    if x.startswith("__ambiguous"):
        x = x.replace("__ambiguous[","").replace("]","").split("+")
    return x


def load_pickle(name):
    with open(name,'rb') as f:
        file = pickle.load(f)
    return file


class SGN(nn.Module):
    def __init__(self, hidden_channels, embed_dim, out_channels, 
                 num_layers = 4, feature_dim=1536, name_dim=4096,
                 gene_path=None, gene2name_path='./preprocess', name_feature_path=None,
                 non_negative_output: bool = True):
        super().__init__()

        if embed_dim > 1536:
            self.mapping = nn.Linear(embed_dim, 1536)
            embed_dim = 1536

        # self.gene = np.array(load_pickle(f"{gene_path}/gene.pkl"))
        # self.gene = np.array(load_pickle(f"{gene_path}/gene.pkl"))
        self.non_negative_output = non_negative_output
        
        with open(gene_path, 'r') as f:
            genes = json.load(f)['genes']
        self.genes = np.array(genes)
            
        self.gene2name = load_pickle(f"{gene2name_path}/gene2name.pkl")
        self.gene2name.update(EXTRA)

        self.gene_mapping = pd.read_csv(f"{gene2name_path}/hgnc-symbol-check.csv")
        
        self.gene_emb, self.size = self.load_gene_emb(name_feature_path)

        self.pretransform = pyg.Linear(feature_dim, hidden_channels,bias=False)
        self.leaklyrelu = nn.LeakyReLU(0.2)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = pyg.HeteroConv({
                ('window', 'near', 'window'): pyg.SAGEConv(hidden_channels,hidden_channels), 
                ('window', 'knn', 'window'): pyg.SAGEConv(hidden_channels,hidden_channels), 
            }, aggr='mean')
            self.convs.append(conv)

        self.lin = pyg.Linear(hidden_channels, out_channels)
        
        self.weight_generator = Transformer(in_dim = name_dim, embed_dim=embed_dim, init_values=0.1)


    def forward(self, data, phase='test'):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        label = data["window"].y
        device = x_dict["window"].device

        window = self.mapping(x_dict["window"]) if getattr(self, 'mapping', None) else x_dict["window"]
        x_dict["window"]  = self.pretransform(window)
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: self.leaklyrelu(x) for key, x in x_dict.items()}
        
        gene_emb = [emb.to(device) for emb in self.gene_emb]
        weight, bias = self.weight_generator(gene_emb, self.size)
        pred = self.lin(x_dict["window"]) @ weight.T + bias.T
        # pred = torch.clamp(pred, 0) 
        if self.non_negative_output:
            pred = F.softplus(pred)  # Ensure non-negativity
        
        result_dict = {'logits': pred}
        if phase == 'train':
            loss = F.mse_loss(pred, label)
            result_dict['loss'] = loss
        
        return result_dict
    
    def load_gene_emb(self, name_feature_path):
        if hasattr(self,"gene_emb"):
            return self.gene_emb, self.test_mask, self.size 
        
        gene_emb = []
        size = []
        for gene in self.genes:
            gene = decompose(gene) if gene.startswith("__ambiguous") else [gene]
            current_emb = []
            
            try:
                for j in gene:
                    if j in self.gene2name:
                        symbol = self.gene2name[j]
                    else:
                        if j in self.gene_mapping['Input'].values:
                            gene_mapped = self.gene_mapping.loc[self.gene_mapping['Input'] == j, 'Approved symbol'].values[0]
                            if gene_mapped in self.gene2name:
                                symbol = self.gene2name[gene_mapped]
                            else:
                                print(f"Warning: Mapped gene {gene_mapped} for input {j} not found in gene2name mapping. Skipping...")
                                continue
                        else:
                            print(f"Warning: Gene {j} not found in gene2name mapping. Skipping...")
                            continue
                    if 'symbol' in symbol:
                        symbol = symbol['symbol']
                        current_emb.append(load_pickle(os.path.join(name_feature_path,f"{symbol}.pkl"))[1][0].cpu())
                        
                        size.append([i.size(1) for i in current_emb])
                        gene_emb.append(torch.cat(current_emb,1))
                    else:
                        print(f"Warning: No symbol found for gene {j}.")
                    
                    
            except FileNotFoundError:
                for j in gene:
                    if j in self.gene_mapping['Input'].values:
                        gene_mapped = self.gene_mapping.loc[self.gene_mapping['Input'] == j, 'Approved symbol'].values[0]
                        
                        if gene_mapped in self.gene2name:
                            symbol = self.gene2name[gene_mapped]
                        else:
                            print(f"Warning: Gene {j} not found in gene2name mapping. Skipping...")
                            continue
                        if 'symbol' in symbol:
                            symbol = symbol['symbol']
                            current_emb.append(load_pickle(os.path.join(name_feature_path,f"{symbol}.pkl"))[1][0].cpu())
                            
                            size.append([i.size(1) for i in current_emb])
                            gene_emb.append(torch.cat(current_emb,1))
                        else:
                            print(f"Warning: No symbol found for gene {j}.")
                    else:
                        raise FileNotFoundError(f"Gene {gene} not found in {name_feature_path}. Please check the path and ensure all gene embeddings are available.")
                # print(f"Gene {gene} not found in {name_feature_path}. Skipping...")
                # continue

        return gene_emb, size


def padding(x, CLSREG = True):
    x = [i.squeeze(0) for i in x]
    MAX_LENGTH = max([len(i) for i in x])
    X_PAD = []
    MASK  = []
    for i in x:
        pad = torch.zeros((MAX_LENGTH - i.size(0),i.size(1))).to(i)
        X_PAD.append(torch.cat((i,pad)))
        MASK.append([0] * i.size(0) + [1] * (MAX_LENGTH - i.size(0)))
    
    X_PAD = torch.stack((X_PAD))
    MASK = torch.BoolTensor(MASK).to(X_PAD.device)
    
    if CLSREG:
        MASK = torch.cat((torch.zeros((MASK.size(0),2)).bool().to(MASK.device),MASK),1)
    MASK = MASK.unsqueeze(1).unsqueeze(1)
    return X_PAD, MASK
        
class Attention(nn.Module):

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.masked_fill_(mask,-float('inf'))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class Mlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, mask:  torch.BoolTensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x),mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
        
class Transformer(nn.Module):

    def __init__(
            self,
            in_dim: int = 3,
            embed_dim: int = 256,
            depth: int = 2,
            num_heads: int = 4,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            init_values: Optional[float] = None,
            class_token: bool = True,
            reg_tokens: int = 1,
            pre_norm: bool = False,
            fc_norm: Optional[bool] = None,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            norm_layer: Optional[nn.Module] = None,
            act_layer: Optional[nn.Module] = None,
            block_fn: Type[nn.Module] = Block,
            mlp_layer: Type[nn.Module] = Mlp,
    ) -> None:
        
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        act_layer = nn.GELU

        self.num_features = self.embed_dim = embed_dim 
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token
        
        self.patch_embed =  nn.Linear(in_dim, embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, reg_tokens, embed_dim)) if reg_tokens else None
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()
        self.segment = nn.Embedding(5, embed_dim)
        torch.nn.init.zeros_(self.segment.weight)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)] 
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(self.embed_dim, embed_dim + 1) 

        self.init_weights()

    def init_weights(self) -> None:
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def _cat_token(self, x: torch.Tensor) -> torch.Tensor:
        
        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))
        x = torch.cat(to_cat + [x], dim=1)    
        return x

    def forward_features(self, x: torch.Tensor, mask: torch.BoolTensor, size:List[List]) -> torch.Tensor:
        x = self.patch_embed(x)
        
        MAX_LEN = x.size(1)
        SEGMENT = []
        for i in size:
            current_size = []
            for k,j in enumerate(i):
                k+=1 
                current_size.extend([k]*j)
            current_size.extend([0] *  (MAX_LEN-len(current_size)))
            SEGMENT.append(current_size)
        
        SEGMENT = self.segment(torch.LongTensor(SEGMENT).to(x.device))
        x = x + SEGMENT
        
        x = self._cat_token(x)
        x = self.norm_pre(x)
        
        x = x[:,:512]
        mask = mask[...,:512]
        
        for block in self.blocks:
            x = block(x,mask)
        x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = x[:, 0] 
        return x if pre_logits else self.head(x)

    def forward(self, x: List[torch.Tensor], size: List[List]) -> torch.Tensor:
        x, mask = padding(x,True)
        x = self.forward_features(x,mask, size)
        x = self.forward_head(x)
        weight, bias = x[:,:-1],x[:,[-1]]
        weight = torch.nn.functional.normalize(weight,p=2,dim=1)
        return weight,bias