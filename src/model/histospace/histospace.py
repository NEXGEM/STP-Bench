import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv_transpose1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.conv_transpose2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.conv_transpose3 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_transpose1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv_transpose2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv_transpose3(x)
        x = self.sigmoid(x)
        return x


class Base_AE(nn.Module):
    def __init__(self):
        super(Base_AE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class HistoSPACE(nn.Module):
    def __init__(self, num_genes, max_batch_size=1024,
                 non_negative_output: bool = True):
        
        super(HistoSPACE, self).__init__()
        
        self.max_batch_size = max_batch_size
        self.non_negative_output = non_negative_output
        
        self.AE = Base_AE()
        
        self.conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)  # Input channels should match encoder output channels
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 28 * 28, 256)  # Adjust the input size based on the conv1 output size
        self.fc2 = nn.Linear(256, num_genes)

    def _forward(self, x, return_rec=False):
        
        # Forward pass through the encoder
        x = self.AE.encoder(x)
        
        if return_rec:
            rec_img = self.AE.decoder(x)
        
        # Additional convolutional layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        # Reshape for FC layers
        x = x.view(x.size(0), -1)
        # FC layers
        x = self.fc1(x)
        x = self.fc2(x)
        
        # Clamp the output to ensure non-negative values
        # x = torch.clamp(x, 0)
        if self.non_negative_output:
            x = F.softplus(x)  # Apply softplus to ensure non-negativity
        if return_rec:
            return x, rec_img
        else:
            return x
        
    def forward(self, img, label=None, **kwargs):
        phase = kwargs.get('phase', 'train')
        
        if phase == 'train':
            pred, rec_img = self._forward(img, return_rec=True)
            ae_loss = F.mse_loss(rec_img, img)
            expr_loss = F.mse_loss(pred, label) if label is not None else 0
            loss = expr_loss + ae_loss
            
            result_dict ={
                "logits": pred,
                "reconstruction": rec_img,
                "loss": loss
            }
        else:
            if img.shape[0] > self.max_batch_size:
                imgs = img.split(self.max_batch_size, dim=0)
                pred = [self._forward(img) for img in imgs]
                pred = torch.cat(pred, dim=0)
            else:
                pred = self._forward(img)
                
            result_dict ={
                "logits": pred
            }
            
        return result_dict

