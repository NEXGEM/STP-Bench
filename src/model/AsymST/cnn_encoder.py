import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import DenseNet121_Weights, densenet121


class DenseNet121Branch(nn.Module):
    """DenseNet-121 CNN branch."""

    def __init__(self, pretrained: bool = True, train_backbone: bool = True):
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = densenet121(weights=weights)
        self.features = backbone.features
        self.out_dim = backbone.classifier.in_features

        if not train_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = F.relu(x, inplace=False)
        cnn_gap = F.adaptive_avg_pool2d(x, output_size=1).flatten(1)
        cnn_tokens = x.flatten(2).transpose(1, 2).contiguous()
        return cnn_tokens, cnn_gap
