
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn as gnn
from torchvision import models      

from .backbone import MLP


class Sepal(nn.Module):
    def __init__(self, 
                 num_genes=200,
                 act='ReLU', 
                 graph_operator='GCNConv', 
                 h_preprocess="1536,512", 
                 h_graph="512,256,128", 
                 h_pred_head="128,256", 
                 pooling='SAGPooling', 
                 sum_positions=False,
                 non_negative_output: bool = True) -> None:
        """_summary_

        Args:
            act (str): Activation function used in the MLP
            graph_operator (str): Graph convolutional operator (i.e. chebconv)
            h_preprocess (str): List of channels for the preprocessing MLP
            h_graph (str): List of channels for the GNN
            h_pred_head (str): List of channels for the prediction head MLP
            pooling (str): Pooling operator (i.e. global_mean_pool)
        """
        super(Sepal, self).__init__()
        
        self.non_negative_output = non_negative_output
        
        h_preprocess = [int(x) for x in h_preprocess.split(',')] if isinstance(h_preprocess, str) else h_preprocess
        h_graph = [int(x) for x in h_graph.split(',')] if isinstance(h_graph, str) else h_graph
        h_pred_head = [int(x) for x in h_pred_head.split(',')] if isinstance(h_pred_head, str) else h_pred_head
        h_pred_head[-1] = num_genes  # Ensure the last layer matches the number of genes
        
        self.act = act
        self.graph_operator = graph_operator
        self.h_preprocess = h_preprocess
        self.h_graph = h_graph
        self.h_pred_head = h_pred_head
        self.pooling = pooling
        self.sum_positions = sum_positions

        self.act_fn = getattr(nn, act)()

        self.graph_operator_fn = getattr(gnn, self.graph_operator)
        self.pooling_fn = getattr(gnn, self.pooling)(in_channels=self.h_graph[-1], ratio=1) if self.pooling == "SAGPooling" else getattr(gnn, self.pooling)

        self.preprocess_layer = MLP(self.h_preprocess, self.act) if self.h_preprocess is not [-1] else nn.Identity()

        # Convolution definitions
        self.layers = nn.ModuleList()
        for i in range(len(self.h_graph)-1):
            self.layers.append(self.graph_operator_fn(self.h_graph[i], self.h_graph[i+1]))
        
        self.prediction_layer = MLP(self.h_pred_head, self.act) if self.h_pred_head is not [-1] else nn.Identity()

    def forward(self, graph, **kwargs):
        """Forward pass of the Sepal model.

        Args:
            graph (torch_geometric.data.Data): Input graph data containing embeddings and positional embeddings.

        Returns:
            torch.Tensor: Output logits after processing the graph.
        """
        phase = kwargs.get('phase', 'train')
        
        if phase == 'train':    
            batch_pred = graph.predictions[graph.ptr[:-1]]
            gnn_pred = self.gnn_forward(graph)
            
            pred = gnn_pred + batch_pred
            # pred = torch.clamp(pred, 0) 
            if self.non_negative_output:
                pred = F.softplus(pred)  # Ensure non-negativity
            label = graph.y[graph.ptr[:-1]] 
            
            loss = F.mse_loss(pred, label) if phase == 'train' else None
            
            result_dict = {'logits': pred,
                        'loss': loss}
        
        else:
            device = kwargs.get('device', 'cuda')
            labels = []
            batch_preds = []
            gnn_preds = []
            for g in graph:
                g = g.to(device)
                batch_pred = g.predictions[g.ptr[:-1]]
                gnn_pred = self.gnn_forward(g)
                label = g.y[g.ptr[:-1]] 
                
                batch_preds.append(batch_pred)
                gnn_preds.append(gnn_pred)
                labels.append(label)
                
            pred = torch.cat(batch_preds, dim=0) + torch.cat(gnn_preds, dim=0)
            # pred = torch.clamp(pred, 0) 
            if self.non_negative_output:
                pred = F.softplus(pred)
            labels = torch.cat(labels, dim=0)
                
            result_dict = {'logits': pred, 'label': labels}        
        
        return result_dict
    
    def gnn_forward(self, graph):
        
        emb_matrix = graph.embeddings
        pos_embs_matrix = graph.positional_embeddings

        if self.sum_positions:
            ftr_matrix = torch.add(emb_matrix, pos_embs_matrix)

        else:
            ftr_matrix = torch.cat((emb_matrix, pos_embs_matrix), dim=1)
        
        x = self.preprocess_layer(ftr_matrix)
        
        for layer in self.layers:
            x = layer(x, graph.edge_index)
            x = self.act_fn(x)

        if self.pooling == "SAGPooling":
            x, _, _, _, _, _ = self.pooling_fn(x, edge_index=graph.edge_index, batch=graph.batch) 
        else:
            x = self.pooling_fn(x, batch=graph.batch)

        out = self.prediction_layer(x)

        return out
