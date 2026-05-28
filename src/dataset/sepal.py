
import os

import json
import numpy as np
from scipy import sparse
import h5py
import torch
from torch_geometric.loader import DataLoader as PygDataLoader

from dataset.base_dataset import STDataset


class SepalDataset(STDataset):
    def __init__(self, 
                mode: str,
                phase: str,
                fold: int,
                data_dir: str,
                meta_dir: str = None,
                ref_data_dir: str = None,
                gene_type: str = 'mean',
                num_genes: int = 1000,
                num_outputs: int = 300,
                normalize: bool = True,
                cpm: bool = False,
                smooth: bool = False,
                data_id: str = None,
                model_name: str = 'uni_v2',
                load_level: str = 'slide'  # 'patch' or 'slide'
                ):
        super(SepalDataset, self).__init__(
                                mode=mode,
                                phase=phase,
                                fold=fold,
                                data_dir=data_dir,
                                meta_dir=meta_dir,
                                gene_type=gene_type,
                                num_genes=num_genes,
                                num_outputs=num_outputs,
                                normalize=normalize,
                                cpm=cpm,
                                smooth=smooth,
                                data_id=data_id,
                                model_name=model_name,
                                load_level=load_level
                                )
        if (model_name is None) or (model_name == 'None'):
            model_type = 'original'
        else:
            model_type = 'modified'

        if cpm:
            if ref_data_dir is not None:
                ref_data = '/'.join(ref_data_dir.replace('/bench_data', '').split('/')[-2:])
                self.graph_dir = f"{data_dir}/sepal/cpm/{model_type}/{ref_data}/fold{fold}/{phase}"
            else:
                self.graph_dir = f"{data_dir}/sepal/cpm/{model_type}/fold{fold}/{phase}"
        else:
            if ref_data_dir is not None:
                ref_data = '/'.join(ref_data_dir.replace('/bench_data', '').split('/')[-2:])
                self.graph_dir = f"{data_dir}/sepal/{model_type}/{ref_data}/fold{fold}/{phase}"
            else:
                self.graph_dir = f"{data_dir}/sepal/{model_type}/fold{fold}/{phase}"

        if phase == 'train':
            self.graph_dicts = {_id: self.load_graph(_id) \
                for _id in self.ids}
        
    def __getitem__(self, index):
        data = {}
        
        if self.phase == 'train':
            i = 0
            while index >= self.cumlen[i]:
                i += 1
            idx = index
            if i > 0:
                idx = index - self.cumlen[i-1]

            name = self.int2id[i]
            
            graph_data = self.graph_dicts[name]
            graph = graph_data[idx]
            
            data['graph'] = graph
            
        elif self.phase == 'test':
            
            if self.mode == 'inference':
                graph_data = self.load_graph(self.name)
                
            else:
                name = self.int2id[index]
                graph_data = self.load_graph(name)
                
            graph_data_batch = PygDataLoader(graph_data,
                                            batch_size=1024)
            data['graph'] = graph_data_batch
                
        return data
    
    def load_graph(self, name: str):
        """Load graph data of a sample.

        Args:
            name (str): name of a sample

        Returns:
            torch_geometric.data.Data: return graph data.
        """
        path = f"{self.graph_dir}/{name}.pt"
        
        if os.path.isfile(path):
            graph_dict = torch.load(path)
            graph_data = list(graph_dict.values())
            return graph_data
        else:
            raise FileNotFoundError(f"{path} is not found.")