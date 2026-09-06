
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import timm

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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
    

class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name, pretrained, trainable, weight="weights/tenpercent_resnet18.ckpt"
    ):
        super().__init__()
        # self.model = timm.create_model(
        #     model_name, pretrained, num_classes=0, global_pool="avg"
        # )
        self.model = load_pretrained_resnet18(weight)
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)
    
class ImageEncoder_resnet50(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name='resnet50', pretrained=True, trainable=True
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)
    
class ImageEncoder_resnet101(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name='resnet101', pretrained=True, trainable=True
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)
    
class ImageEncoder_resnet152(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name='resnet152', pretrained=True, trainable=True
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)
    
class ImageEncoder_ViT(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name="vit_base_patch32_224", pretrained=False, trainable=True
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)
    

class ImageEncoder_CLIP(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name="vit_base_patch32_224_clip_laion2b", pretrained=True, trainable=True
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)

class ImageEncoder_ViT_L(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name="vit_large_patch32_224_in21k", pretrained=False, trainable=True
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)
    

#  'vit_base_patch32_224',
#  'vit_base_patch32_224_clip_laion2b',
#  'vit_base_patch32_224_in21k',
#  'vit_base_patch32_224_sam',
    


# class SpotEncoder(nn.Module):
#     #to change...
#     def __init__(self, model_name=CFG.spot_encoder_model, pretrained=CFG.pretrained, trainable=True):
#         super().__init__()
#         if pretrained:
#             self.model = DistilBertModel.from_pretrained(model_name)
#         else:
#             self.model = DistilBertModel(config=DistilBertConfig())
            
#         for p in self.model.parameters():
#             p.requires_grad = trainable

#         # we are using the CLS token hidden representation as the sentence's embedding
#         self.target_token_idx = 0

#     def forward(self, input_ids, attention_mask):
#         output = self.model(input_ids=input_ids, attention_mask=attention_mask)
#         last_hidden_state = output.last_hidden_state
#         return last_hidden_state[:, self.target_token_idx, :]



class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim,
        dropout
    ):
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


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


class BLEEP(nn.Module):
    def __init__(
        self,
        temperature,
        image_embedding,
        num_genes,
        projection_dim,
        model_name='resnet50',
        pretrained=True,
        trainable=True,
        dropout=0.1,
        weight="weights/cigar/tenpercent_resnet18.ckpt",
        infer_method='average',
        use_pretrained_emb=False,
        max_batch_size=1024
    ):
        super().__init__()

        self.use_pretrained_emb = use_pretrained_emb
        
        if not use_pretrained_emb:
            self.image_encoder = ImageEncoder(model_name=model_name, pretrained=pretrained, trainable=trainable, weight=weight)
        else:
            if image_embedding > 1536:
                self.mapping = nn.Linear(image_embedding, 1536)
                image_embedding = 1536

        self.image_projection = ProjectionHead(embedding_dim=image_embedding, projection_dim=projection_dim, dropout=dropout) #aka the input dim, 2048 for resnet50        
        self.spot_projection = ProjectionHead(embedding_dim=num_genes, projection_dim=projection_dim, dropout=dropout)

        self.temperature = temperature
        
        self.infer_method = infer_method
        self.num_k = 1 if infer_method == 'simple' else 50
        self.max_batch_size = max_batch_size

        
    def forward(self, img, label=None, **kwargs):
        phase = kwargs.get('phase', 'train')
        
        if self.use_pretrained_emb:
            img = kwargs.get('img_emb', None)
            if img is None:
                raise ValueError("Image embeddings must be provided when use_pretrained_emb is True.")
            
            if getattr(self, 'mapping', None) is not None:
                img = self.mapping(img)
        
        if phase == 'train':
            return self._process_training_batch(img, label)
        
        elif phase == 'val':
            if 'dataset' not in kwargs:
                raise ValueError("val inference requires a dataset to be passed in.")
            return self._process_inference_batch(img, kwargs['dataset'])
        
        elif phase == 'test':
            if 'dataset' not in kwargs:
                raise ValueError("Inference mode requires a dataset to be passed in.")
            return self._process_inference_batch(img, kwargs['dataset'])
        
    def _process_training_batch(self, img, spot_features):
        img_embeddings = self.get_img_embeddings(img)
        
        # spot_embeddings = self.spot_projection(spot_features)
        # spot_embeddings = self.get_spot_embeddings(spot_features)
        spot_embeddings = self.spot_projection(spot_features)
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
        
        if spot_expressions_ref.shape[0] > self.max_batch_size:
            spot_expression_refs = spot_expressions_ref.split(self.max_batch_size, dim=0)
            spot_embeddings_ref = [self.spot_projection(spot_expression_ref) for spot_expression_ref in spot_expression_refs]
            spot_embeddings_ref = torch.cat(spot_embeddings_ref, dim=0)
        else:
            spot_embeddings_ref = self.spot_projection(spot_expressions_ref)
            
        # spot_embeddings_ref = self.get_spot_embeddings(spot_expressions_ref)
        indices = self.find_matches(spot_embeddings_ref, img_embeddings, top_k=self.num_k)
        
        if self.infer_method == 'simple':    
            matched_spot_expression_pred = spot_expressions_ref[indices[:,0],:]
            
        elif self.infer_method == 'average':
            matched_spot_expression_pred = torch.zeros((indices.shape[0], spot_expressions_ref.shape[1])).to(device)
            for i in range(indices.shape[0]):
                matched_spot_expression_pred[i,:] = torch.mean(spot_expressions_ref[indices[i,:],:], dim=0)
                
        elif self.infer_method == 'weighted_average':
            matched_spot_expression_pred = torch.zeros((indices.shape[0], spot_expressions_ref.shape[1])).to(device)
            for i in range(indices.shape[0]):
                a = torch.sum((spot_embeddings_ref[indices[i,0],:] - img_embeddings[i,:])**2) #the smallest MSE 
                weights = torch.exp(-(torch.sum((spot_embeddings_ref[indices[i,:],:] - img_embeddings[i,:])**2, dim=1)-a+1))
                matched_spot_expression_pred[i,:] = torch.sum(spot_expressions_ref[indices[i,:],:] * weights.unsqueeze(1), dim=0) / weights.sum()

        return {'logits': matched_spot_expression_pred}    
    
    def calculate_loss(self, image_embeddings, spot_embeddings):
              
        # Calculating the Loss
        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T
        targets = F.softmax(
            (images_similarity + spots_similarity) / 2 * self.temperature, dim=-1
        )
        spots_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + spots_loss) / 2.0 # shape: (batch_size)
        
        return loss.mean()
    
    @staticmethod
    def find_matches(spot_embeddings, query_embeddings, top_k=1):
        #find the closest matches 
        # spot_embeddings = torch.tensor(spot_embeddings)
        # query_embeddings = torch.tensor(query_embeddings)
        query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
        spot_embeddings = F.normalize(spot_embeddings, p=2, dim=-1)
        dot_similarity = query_embeddings @ spot_embeddings.T   #2277x2265
        # print(dot_similarity.shape)
        _, indices = torch.topk(dot_similarity.squeeze(0), k=top_k)
        
        if indices.dim() == 1:
            indices = indices.unsqueeze(0)
        return indices
    
    def get_img_embeddings(self, img):
        if not self.use_pretrained_emb:
            image_features = self.image_encoder(img)
            image_embeddings = self.image_projection(image_features)  
        else:
            image_embeddings = self.image_projection(img)  
        
        return image_embeddings