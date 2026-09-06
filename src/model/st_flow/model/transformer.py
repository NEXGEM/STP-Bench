import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp, SwiGLUPacked
from torch_geometric.utils import to_dense_batch
from einops import rearrange, einsum

from .fa import FrameAveraging


def get_activation(activation="gelu"):
    return {
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "relu": nn.ReLU,
    }[activation]


class GeneUpdate(nn.Module):
    def __init__(
            self, 
            d_model, 
            n_genes,
            proj_drop=0.,
        ):
        super(GeneUpdate, self).__init__()    

        self.output = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(proj_drop),
            nn.Linear(d_model, n_genes),
            nn.Dropout(proj_drop),
        )
    
    def forward(self, features):
        update = self.output(features) 
        return update


class MLPAttnEdgeAggregation(FrameAveraging):
    def __init__(
            self, 
            d_model, 
            d_edge_model,
            n_heads=1,
            proj_drop=0.,
            attn_drop=0.,
            activation='gelu',
            n_genes=50
        ):
        super(MLPAttnEdgeAggregation, self).__init__(dim=2)
        
        self.d_head, self.d_edge_head, self.n_heads = d_model // n_heads, d_edge_model // n_heads, n_heads

        self.layernorm_qkv = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 3),
        )

        if activation == "swiglu":
            self.mlp_attn = SwiGLUPacked(
                in_features=self.d_head*2+self.d_edge_head+n_genes, hidden_features=d_model, 
                out_features=1, drop=proj_drop, norm_layer=nn.LayerNorm
            )
            self.edge_trans = SwiGLUPacked(
                in_features=self.dim+1, hidden_features=d_edge_model, 
                out_features=d_edge_model, drop=proj_drop, norm_layer=nn.LayerNorm
            )
            self.W_output = SwiGLUPacked(
                in_features=d_model+d_edge_model, hidden_features=d_model, 
                out_features=d_model, drop=proj_drop, norm_layer=nn.LayerNorm
            )
        else:
            self.mlp_attn = Mlp(
                in_features=self.d_head*2+self.d_edge_head+n_genes, hidden_features=d_model, 
                out_features=1, drop=proj_drop, norm_layer=nn.LayerNorm
            )
            self.edge_trans = Mlp(
                in_features=self.dim+1, hidden_features=d_edge_model, out_features=d_edge_model, 
                act_layer=get_activation(activation), drop=proj_drop, norm_layer=nn.LayerNorm
            )
            self.W_output = Mlp(
                in_features=d_model+d_edge_model, hidden_features=d_model, out_features=d_model, 
                act_layer=get_activation(activation), drop=proj_drop, norm_layer=nn.LayerNorm
            )

        self.attn_dropout = nn.Dropout(attn_drop)

    def forward(self, gene_exp, token_embs, coords, neighbor_indices, neighbor_masks=None):
        # gene_exp: [N, N_genes], token_embs: [N, -1], geo_token_embs: [N, 3]
        # neighbor_indices: [N, N_neighbor], neighbor_masks: [N, N_neighbor]
        n_tokens, n_neighbors = token_embs.size(0), neighbor_indices.size(1)
        n_heads, d_head, d_edge_head = self.n_heads, self.d_head, self.d_edge_head

        q_s, k_s, v_s = self.layernorm_qkv(token_embs).chunk(3, dim=-1)
        q_s, k_s, v_s = map(lambda x: rearrange(x, 'n (h d) -> n h d', h=n_heads), (q_s, k_s, v_s))

        """build pairwise representation with FA"""
        radial_coords = coords[neighbor_indices] - coords.unsqueeze(dim=1)  # [N, N_neighbor, 2]
        radial_coord_norm = radial_coords.norm(dim=-1).unsqueeze(-1)  # [N, N_neighbor, 1]

        frame_feats, _, _ = self.create_frame(radial_coords, neighbor_masks)  # [N*8, N_neighbors, 3]
        frame_feats = frame_feats.view(n_tokens, self.n_frames, n_neighbors, -1)  # [N, 8, N_neighbors, d_model]

        radial_coord_norm = radial_coord_norm.unsqueeze(dim=1).expand(n_tokens, self.n_frames, n_neighbors, -1)
        frame_feats = self.edge_trans(torch.cat([frame_feats, radial_coord_norm], dim=-1)).mean(dim=1)  # [N, N_neighbors, d_edge_model]

        """gene expression features"""
        gene_exp_diff = gene_exp[neighbor_indices] - gene_exp.unsqueeze(dim=1)  # [N, N_neighbor, N_genes]
        gene_exp_feats_expand = gene_exp_diff[..., None, :].expand(n_tokens, n_neighbors, n_heads, -1)  # [N, N_neighbor, n_heads, N_genes+1]

        """attention map"""
        q_s = q_s.unsqueeze(dim=1).expand(n_tokens, n_neighbors, n_heads, d_head)
        frame_feats = frame_feats.view(n_tokens, n_neighbors, n_heads, d_edge_head)
        message = torch.cat([q_s, k_s[neighbor_indices], frame_feats, gene_exp_feats_expand], dim=-1)
        
        attn_map = self.mlp_attn(message).squeeze(-1)
        if neighbor_masks is not None:
            attn_map.masked_fill_(neighbor_masks.unsqueeze(dim=-1), -1e9)
        attn_map = self.attn_dropout(nn.Softmax(dim=-1)(attn_map.transpose(1, 2)))  # [N, n_heads, N_neighbor]

        """context aggregation"""
        v_s_neighs = v_s[neighbor_indices].view(n_tokens, -1, n_heads, d_head)  # [N, n_heads, N_neighbor, D]
        scalar_context = einsum(attn_map, v_s_neighs, 'n h m, n m h d -> n h d').view(n_tokens, -1)  # [N, n_heads*D]
        edge_context = einsum(attn_map, frame_feats, 'n h m, n m h d -> n h d').view(n_tokens, -1)  # [N, n_heads*D]
        return self.W_output(torch.cat([scalar_context, edge_context], dim=-1))


class TransformerBlock(nn.Module):
    def __init__(            
            self,
            d_model,
            d_edge_model,
            n_genes,
            n_heads=1,
            activation="gelu",
            attn_drop=0.,
            proj_drop=0.,
            mlp_ratio=4.0,
        ):
        super(TransformerBlock, self).__init__()

        self.attn = MLPAttnEdgeAggregation(
            d_model=d_model, d_edge_model=d_edge_model, n_heads=n_heads, 
            proj_drop=proj_drop, attn_drop=attn_drop, activation=activation, n_genes=n_genes
        )

        if activation == "swiglu":
            self.mlp = SwiGLUPacked(
                in_features=d_model, hidden_features=int(d_model * mlp_ratio), drop=proj_drop, norm_layer=nn.LayerNorm
            )
        else:
            self.mlp = Mlp(
                in_features=d_model, hidden_features=int(d_model * mlp_ratio), 
                act_layer=get_activation(activation), drop=proj_drop, norm_layer=nn.LayerNorm
            )
        
        self.gene_updater = GeneUpdate(d_model, n_genes, proj_drop=proj_drop)

    def forward(self, gene_exp, token_embs, coords, neighbor_indices):
        context_token_embs = self.attn(gene_exp, token_embs, coords, neighbor_indices)
        token_embs = token_embs + context_token_embs

        token_embs = token_embs + self.mlp(token_embs)
        gene_exp = self.gene_updater(token_embs)

        return gene_exp, token_embs


class SpatialTransformer(nn.Module):
    def __init__(self, config):
        super(SpatialTransformer, self).__init__()

        self.n_neighbors = config.n_neighbors

        self.blks = nn.ModuleList([
            TransformerBlock(config.d_model, config.d_edge_model, 
                          n_genes=config.n_genes, n_heads=config.n_heads, 
                              activation=config.act, attn_drop=config.attn_dropout, 
                              proj_drop=config.dropout, 
                            ) \
                for i in range(config.n_layers)
        ])

    def _build_graph(self, coords, batch_idx, n_neighbors, exclude_self=True):
        # coords: [N, 2], batch_idx: [N], n_neighbors: int
        exclude_self_mask = torch.eye(coords.shape[0], dtype=torch.bool, device=coords.device)  # 1: diagonal elements
        batch_mask = batch_idx.unsqueeze(0) == batch_idx.unsqueeze(1)  # [N, N], True if the token is in the same batch

        # calculate relative distance
        rel_pos = rearrange(coords, 'n d -> n 1 d') - rearrange(coords, 'n d -> 1 n d')
        rel_dist = rel_pos.norm(dim = -1).detach()  # [N, N]
        if exclude_self:
            rel_dist.masked_fill_(exclude_self_mask | ~batch_mask, 1e9)
        else:
            rel_dist.masked_fill_(~batch_mask, 1e9)

        dist_values, nearest_indices = rel_dist.topk(n_neighbors, dim = -1, largest = False)
        return nearest_indices

    def forward(self, gene_exp, features, coords):
        # gene_exp: [B, N_cells, N_genes], features: [B, N_cells, -1], coords: [B, N_cells, 2]
        B, N_cells, N_genes = gene_exp.shape[0], gene_exp.shape[1], gene_exp.shape[-1]
        device = features.device
        
        pad_mask = features.sum(dim=-1) == 0  # [B, N_cells], True if the token is padding
        batch_idx = torch.arange(B, device=device).unsqueeze(-1).repeat(1, N_cells)[~pad_mask]

        features = features[~pad_mask]  # [-1, 1024]
        coords = coords[~pad_mask]  # [-1, 3]
        gene_exp = gene_exp[~pad_mask]  # [-1, N_genes]

        nearest_indices = self._build_graph(
            coords, batch_idx, min(self.n_neighbors, N_cells), exclude_self=True
        )

        # forward pass
        all_gene_exp = []
        for blk in self.blks:
            gene_exp, features = blk(gene_exp, features, coords, nearest_indices)
            all_gene_exp.append(gene_exp)
        gene_exp = torch.stack(all_gene_exp, dim=0).mean(dim=0)  # [B, N_cells, N_genes]
        
        # average the gene expression among the neighbors
        gene_exp, _ = to_dense_batch(gene_exp, batch=batch_idx, fill_value=0, max_num_nodes=N_cells)  # [B, N_cells, N_genes]
        return gene_exp