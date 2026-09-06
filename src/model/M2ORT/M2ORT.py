import torch
from torch import nn

import torch.nn.functional as F

from torchvision.transforms.functional import center_crop
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.1):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_qkv1 = nn.Linear(dim//3, inner_dim, bias = False)
        self.to_qkv2 = nn.Linear(dim//3, inner_dim, bias = False)
        self.to_qkv3 = nn.Linear(dim//3, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # x = self.norm(x)
        x = self.norm(x).chunk(3, dim = -1)
        qkv1=self.to_qkv1(x[0])
        qkv2=self.to_qkv2(x[1])
        qkv3=self.to_qkv3(x[2])
        qkv=(qkv1, qkv2, qkv3)
        # qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class M2ORT(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        num_genes=50,
        dim=768,
        depth=8,
        heads=12,
        mlp_dim=512,
        pool='cls',
        channels=3,
        dim_head=64,
        dropout=0.,
        emb_dropout=0.,
        max_batch_size=64,
        non_negative_output: bool = True
    ):
        super().__init__()
        self.max_batch_size = max_batch_size
        self.non_negative_output = non_negative_output

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        self.pool = pool
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding1 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim//3),
            nn.LayerNorm(dim//3),
        )
        self.to_patch_embedding2 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height//2, p2=patch_width//2),
            nn.LayerNorm(patch_dim//4),
            nn.Linear(patch_dim//4, dim),
            nn.GELU(),
            nn.Linear(dim, dim//3),
            nn.LayerNorm(dim//3),
        )
        self.to_patch_embedding3 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height//4, p2=patch_width//4),
            nn.LayerNorm(patch_dim//16),
            nn.Linear(patch_dim//16, dim),
            nn.GELU(),
            nn.Linear(dim, dim//3),
            nn.LayerNorm(dim//3),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Linear(dim, num_genes)
        self.loss_fn = nn.MSELoss()
        
    def forward(self, img, img2, img3, label=None, **kwargs):
        phase = kwargs.get('phase', 'train')
        if phase == 'train':
            pred = self._forward(img, img2, img3)
        else:
            if img.shape[0] > self.max_batch_size:
                imgs = img.split(self.max_batch_size, dim=0)
                imgs2 = img2.split(self.max_batch_size, dim=0)
                imgs3 = img3.split(self.max_batch_size, dim=0)
                pred = [self._forward(imgs[i], imgs2[i], imgs3[i]) for i in range(len(imgs))]
                pred = torch.cat(pred, dim=0)
            else:
                pred = self._forward(img, img2, img3)
        
        # pred = torch.clamp(pred, 0) 
        if self.non_negative_output:
            pred = F.softplus(pred)
        
        result_dict = {'logits': pred}

        if label is not None:
            loss = F.mse_loss(pred, label)
            result_dict['loss'] = loss
                
        return result_dict

    def _forward(self, img, img2, img3):
        
        # img_crop
        img_crop = img
        img2_crop = center_crop(img2, [112, 112]) 
        img3_crop = center_crop(img3, [56, 56])
        
        x1 = self.to_patch_embedding1(img_crop)
        x2 = self.to_patch_embedding2(img2_crop)
        x3 = self.to_patch_embedding3(img3_crop)

        x = torch.cat([x1, x2, x3], dim=-1)  # (B, L, dim)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, L+1, dim)
        x = x + self.pos_embedding[:, :n+1]
        x = self.dropout(x)

        x = self.transformer(x)  # (B, L+1, dim)

        feat = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        
        return self.mlp_head(feat)