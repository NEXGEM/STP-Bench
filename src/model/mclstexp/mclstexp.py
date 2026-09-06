import math
import timm
import torch
from torch import nn, einsum
from einops import rearrange
from torchvision.models import DenseNet121_Weights, ResNet18_Weights
import torchvision.models as models
import torch.nn.functional as F

import torchvision

def load_pretrained_resnet18(path: str):       
        """Load pretrained ResNet18 model without final fc layer.

        Args:
            path (str): path_for_pretrained_weight

        Returns:
            torchvision.models.resnet.ResNet: ResNet model with pretrained weight
        """
        
        resnet = torchvision.models.__dict__['resnet18'](weights=None)
        
        state = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = state['state_dict']
        for key in list(state_dict.keys()):
            state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)
        
        model_dict = resnet.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        if state_dict == {}:
            print('No weight could be loaded..')
        model_dict.update(state_dict)
        resnet.load_state_dict(model_dict)
        resnet.fc = nn.Identity()

        return resnet
    
    
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads  # 64*8 = 512
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class attn_block(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.attn = PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))
        self.ff = PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))

    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        for p in self.model.parameters():
            p.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return x


class ImageEncoder_Resnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        for p in self.model.parameters():
            p.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return x


class ImageEncoder_VIT(nn.Module):
    def __init__(
            self, model_name="vit_base_patch32_224", pretrained=True, trainable=True
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)


class ImageEncdoer_res18(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        for p in self.model.parameters():
            p.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return x


class ImageEncdoer_res101(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet101(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        for p in self.model.parameters():
            p.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return x


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim, dropout=0.):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)

        return x

class FourierPositionalEncoding(nn.Module):
    def __init__(self, coord_dim=2, embed_dim=128, scale=10.0):
        super().__init__()
        self.coord_dim = coord_dim
        self.embed_dim = embed_dim
        self.scale = scale

        assert embed_dim % (2 * coord_dim) == 0
        self.freqs = nn.Parameter(torch.randn(embed_dim // 2, coord_dim) * scale)

    def forward(self, coord):
        # coord: [B, N, coord_dim] assumed to be normalized to [-1, 1]
        coord = coord.unsqueeze(-2)  # [B, N, 1, coord_dim]
        freqs = self.freqs.view(1, 1, -1, self.coord_dim)  # [1, 1, embed_dim//2, coord_dim]
        x_proj = 2 * math.pi * torch.sum(coord * freqs, dim=-1)  # [B, N, embed_dim//2]
        pe = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)  # [B, N, embed_dim]
        return pe

class mclSTExp(nn.Module):
    def __init__(self,
                 encoder_name, 
                 temperature, 
                 image_dim, 
                 num_genes, 
                 projection_dim, 
                 heads_num, 
                 heads_dim, 
                 head_layers, 
                 dropout=0.,
                 weight="weights/cigar/tenpercent_resnet18.ckpt",
                 num_k=600,
                 n_pos=3000,
                 use_pretrained_emb=False,
                 max_batch_size=1024):
        super().__init__()
        self.max_batch_size = max_batch_size
        self.use_pretrained_emb = use_pretrained_emb
        
        self.num_k = num_k
        # self.pos_emb = FourierPositionalEncoding(coord_dim=2, embed_dim=dim)
        # self.pos_emb = nn.Sequential(
        #     FourierPositionalEncoding(coord_dim=2, embed_dim=32),
        #     nn.Linear(32, num_genes),
        #     nn.ReLU(),
        #     nn.Linear(num_genes, num_genes)
        # )
        self.x_embed = nn.Embedding(n_pos, num_genes)
        self.y_embed = nn.Embedding(n_pos, num_genes)
        
        if not use_pretrained_emb:
            if encoder_name == "resnet50":
                self.image_encoder = ImageEncoder_Resnet()
            if encoder_name == "densenet121":
                self.image_encoder = ImageEncoder()
            if encoder_name == "vit":
                self.image_encoder = ImageEncoder_VIT()
            if encoder_name == "res18":
                self.image_encoder = ImageEncdoer_res18()
            if encoder_name == "res101":
                self.image_encoder = ImageEncdoer_res101()

        else:
            if image_dim > 1536:
                self.mapping = nn.Linear(image_dim, 1536)
                image_dim = 1536
            # self.image_encoder = load_pretrained_resnet18(weight)
        # self.image_encoder = ImageEncoder(model_name=model_name, pretrained=pretrained, trainable=trainable, weight=weight)
        
        self.spot_encoder = nn.Sequential(
            *[attn_block(num_genes, heads=heads_num, dim_head=heads_dim, mlp_dim=num_genes, dropout=dropout) for _ in
              range(head_layers)])

        self.image_projection = ProjectionHead(embedding_dim=image_dim, projection_dim=projection_dim)
        self.spot_projection = ProjectionHead(embedding_dim=num_genes, projection_dim=projection_dim)

        self.temperature = temperature
        
    def forward(self, img, position, label=None, **kwargs):
        phase = kwargs.get('phase', 'train')

        if self.use_pretrained_emb:
            img = kwargs.get('img_emb', None)
            if img is None:
                raise ValueError("Image embeddings must be provided when use_pretrained_emb is True.")
            
            if getattr(self, 'mapping', None) is not None:
                img = self.mapping(img)
        
        if phase == 'train':
            return self._process_training_batch(img, label, position)
        
        elif phase in ('val', 'test'):
            if 'dataset' not in kwargs:
                raise ValueError("Inference mode requires a dataset to be passed in.")
            return self._process_inference_batch(img, kwargs['dataset'])
        
    def _process_training_batch(self, img, spot_features, position):
        img_embeddings = self.get_img_embeddings(img)
        
        spot_embeddings = self.get_spot_embeddings(spot_features, position)
        
        loss = self.calculate_loss(img_embeddings, spot_embeddings)
        
        return {'loss': loss}
        
    def _process_inference_batch(self, img, dataset):
        if img.shape[0] > self.max_batch_size:
            imgs = img.split(self.max_batch_size, dim=0)
            img_embeddings = [self.get_img_embeddings(img) for img in imgs]
            img_embeddings = torch.cat(img_embeddings, dim=0)
        else:
            img_embeddings = self.get_img_embeddings(img)
            
        device = img.device
        spot_expressions_ref = dataset.spot_expressions_ref.clone().to(device)
        positions_ref = dataset.positions_ref.clone().to(device)
        
        if spot_expressions_ref.shape[0] > self.max_batch_size:
            spot_expression_refs = spot_expressions_ref.split(self.max_batch_size, dim=0)
            positions_refs = positions_ref.split(self.max_batch_size, dim=0)
            spot_embeddings_ref = [self.get_spot_embeddings(spot_expression_ref, positions_refs[i]) for i, spot_expression_ref in enumerate(spot_expression_refs)]
            spot_embeddings_ref = torch.cat(spot_embeddings_ref, dim=0)
        else:
            spot_embeddings_ref = self.get_spot_embeddings(spot_expressions_ref, positions_ref)
        
        indices = self.find_matches(spot_embeddings_ref, img_embeddings, top_k=self.num_k)
        
        matched_spot_expression_pred = torch.zeros((indices.shape[0], spot_expressions_ref.shape[1])).to(device)

        for i in range(indices.shape[0]):
            dis = torch.norm(spot_embeddings_ref[indices[i, :], :] - img_embeddings[i, :], dim=1)
            weights = 1.0 / (dis ** 2 + 1e-8)  
            weights = weights / torch.sum(weights)
            matched_spot_expression_pred[i, :] = torch.sum(
                spot_expressions_ref[indices[i, :], :] * weights.unsqueeze(1), dim=0
            )

        return {'logits': matched_spot_expression_pred}    
    
    def calculate_loss(self, img_embeddings, spot_embeddings):
        
        cos_smi = (spot_embeddings @ img_embeddings.T) / self.temperature
        label = torch.eye(cos_smi.shape[0], cos_smi.shape[1]).cuda()
        spots_loss = F.cross_entropy(cos_smi, label)
        images_loss = F.cross_entropy(cos_smi.T, label.T)
        loss = (images_loss + spots_loss) / 2.0
        
        return loss.mean()
    
    @staticmethod
    def find_matches(spot_embeddings, query_embeddings, top_k=1):
        #find the closest matches 
        # spot_embeddings = torch.tensor(spot_embeddings)
        # query_embeddings = torch.tensor(query_embeddings)
        query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
        spot_embeddings = F.normalize(spot_embeddings, p=2, dim=-1)
        dot_similarity = query_embeddings @ spot_embeddings.T   #2277x2265
        print(dot_similarity.shape)
        _, indices = torch.topk(dot_similarity.squeeze(0), k=top_k)
        
        if indices.dim() == 1:
            indices = indices.unsqueeze(0)
        return indices
    
    def get_img_embeddings(self, img):
        if not self.use_pretrained_emb:
            image_features = self.image_encoder(img)
            img_embeddings = self.image_projection(image_features)  
        else:
            img_embeddings = self.image_projection(img)  
        
        return img_embeddings
    
    def get_spot_embeddings(self, spot_features, position):
        # pos =  self.pos_emb(position).squeeze()
        x = position[:, 0].long()
        y = position[:, 1].long()
        centers_x = self.x_embed(x)
        centers_y = self.y_embed(y)

        spot_features = spot_features + centers_x + centers_y
        
        # spot_features = spot_features + pos
        
        spot_features = spot_features.unsqueeze(dim=0)
        spot_embeddings = self.spot_encoder(spot_features)
        spot_embeddings = self.spot_projection(spot_embeddings)
        spot_embeddings = spot_embeddings.squeeze(dim=0)
        
        return spot_embeddings


