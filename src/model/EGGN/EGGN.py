import torch
import torch.nn as nn
import torch_geometric.nn as pyg
from .geb import EGBBlock
from .csra import CSRA
from .heteroconv import HeteroConv
import torch.nn.functional as F

'''
code is based on https://pytorch-geometric.readthedocs.io/en/latest/

'''


def pearson_R(x, y):
    xm = x - x.mean()
    ym = y - y.mean()
    r_val = xm.dot(ym) / (torch.norm(xm, 2) * torch.norm(ym, 2) + 1e-8)
    return torch.nan_to_num(r_val, nan=-1)


class EGGN(torch.nn.Module):
    def __init__(self, num_layers = 4, hidden_channels=512, mdim=1536, num_genes=200, non_negative_output: bool = True):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_genes = num_genes
        self.non_negative_output = non_negative_output

        self.pretransform_win = pyg.Linear(mdim,hidden_channels,bias=False)
        self.pretransform_exp = pyg.Linear(mdim+num_genes,hidden_channels,bias=False)
        self.post_transform = nn.Sequential(
            nn.LeakyReLU(0.2,True),
            pyg.Linear(hidden_channels,hidden_channels,bias=False),
            nn.LeakyReLU(0.2,True),
            )
        self.pretransform_ey = pyg.Linear(num_genes,hidden_channels,bias=False)
        self.leaklyrelu = nn.LeakyReLU(0.2)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('window', 'near', 'window'): pyg.SAGEConv(hidden_channels,hidden_channels),
                ('example', 'close', 'example'): pyg.SAGEConv(hidden_channels,hidden_channels), 
                ('example', 'refer', 'window'): EGBBlock((hidden_channels, hidden_channels,hidden_channels), hidden_channels, hidden_channels, add_self_loops = False), 
            }, aggr='mean')
            self.convs.append(conv)

        self.pool = CSRA(hidden_channels)
        self.lin = pyg.Linear(hidden_channels, num_genes)
        
    def forward(self, data, phase='test'):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        label = data["window"].y

        example = x_dict["example"]  # [N, mdim + num_genes] = [image emb | exemplar expression]
        x_dict["example_y"] = self.pretransform_ey(example[:, -self.num_genes:])
        x_dict["example"]  = self.post_transform(self.pretransform_exp(example))
        x_dict['window'] = self.post_transform(self.pretransform_win(x_dict["window"]))
        
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: self.leaklyrelu(x) for key, x in x_dict.items()}
        
        pred_count = self.lin(self.pool(x_dict, edge_index_dict))
        # pred_count = torch.clamp(pred_count, 0) 
        if self.non_negative_output:
            pred_count = F.softplus(pred_count)
        
        result_dict = {'logits': pred_count}
        
        if phase == 'train':
            loss = F.mse_loss(pred_count, label)
            corrloss = self.correlationMetric(pred_count, label)

            final_loss = loss + corrloss * 0.5
            result_dict['loss'] = final_loss
        
        return result_dict
    
    def correlationMetric(self, x, y):
        corr = 0
        for idx in range(x.size(1)):
            corr += pearson_R(x[:, idx], y[:, idx])
        corr /= (idx + 1)
        return (1 - corr).mean()
