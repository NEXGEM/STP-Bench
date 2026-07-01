
import scanpy as sc
import pandas as pd
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from model.omiclip.loki.predex import predict_st_gene_expr
sc.settings.set_figure_params(dpi=80, facecolor="white")


class OmiCLIP(nn.Module):

    def __init__(self):
        super(OmiCLIP, self).__init__()

    def forward(self, pid, **kwargs):
        # phase = kwargs.get('phase', 'train')
        device = kwargs.get('device', 'cuda')
        if isinstance(pid, torch.Tensor):
            pid = int(pid.detach().cpu().view(-1)[0].item())
        
        dataset = kwargs['dataset']
        train_data = dataset.spot_expressions_ref.clone().numpy()
        
        test_ids = dataset.ids
        ref_data_dir = dataset.ref_data_dir
        ref_data_dir = ref_data_dir.replace('bench_data/', '')
        ref_data_dir = '/'.join(ref_data_dir.split('/')[-2:])
        data_dir = dataset.data_dir
        fold = dataset.fold

        similarity_path = os.path.join(data_dir, 
                                       'similarity_matrix', 
                                       ref_data_dir,
                                       f'fold{fold}')
        if not os.path.isdir(similarity_path):
            similarity_path = os.path.join(data_dir, 'similarity_matrix', f'fold{fold}')

        test_id = test_ids[pid]
        image_text_similarity = np.load(f"{similarity_path}/{test_id}.npy")
        # for _id in test_ids:
        #     image_text_similarity = np.load(f"{similarity_path}/{_id}.npy")
        predicted_image_text_matrix = predict_st_gene_expr(image_text_similarity, train_data)
        output = torch.FloatTensor(predicted_image_text_matrix).to(device)

        return {'logits': output}
