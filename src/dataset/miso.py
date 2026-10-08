import numpy as np
import torch
from scipy import sparse
from sklearn.neighbors import kneighbors_graph

from dataset.base_dataset import STDataset


class MisoDataset(STDataset):
    """STDataset arranged slide by slide for MISO (one item = one slide).

    Each item is `img_emb` (N, D) per-spot embeddings, `neighbors` (N, K) and `label` (N, G). The neighbors
    are built exactly like the reference `compute_neighbors.py`: `kneighbors_graph(pos, K, include_self=True)`
    on the spot coordinates (here the top-left coordinates stored next to the embeddings), with the K indices
    of each row in ascending order. A slide with fewer than K spots uses all of its spots.

    Training also iterates slides (the model attends across the spots of a slide); a slide is never padded
    or truncated.
    """

    def __init__(self,
                 mode: str,
                 phase: str,
                 fold: int,
                 data_dir: str,
                 meta_dir: str = None,
                 ref_data_dir: str = None,
                 wsi_dir: str = None,
                 gene_type: str = 'mean',
                 num_genes: int = 1000,
                 num_outputs: int = 300,
                 normalize: bool = True,
                 cpm: bool = False,
                 smooth: bool = False,
                 data_id: str = None,
                 model_name: str = 'uni_v2',
                 load_level: str = 'patch',
                 use_emb: bool = True,
                 genes_override: list = None,
                 num_neighbors: int = 37,
                 ):
        super().__init__(mode=mode,
                         phase=phase,
                         fold=fold,
                         data_dir=data_dir,
                         meta_dir=meta_dir,
                         ref_data_dir=ref_data_dir,
                         wsi_dir=wsi_dir,
                         gene_type=gene_type,
                         num_genes=num_genes,
                         num_outputs=num_outputs,
                         normalize=normalize,
                         cpm=cpm,
                         smooth=smooth,
                         data_id=data_id,
                         model_name=model_name,
                         load_level=load_level,
                         use_emb=use_emb,
                         genes_override=genes_override)
        self.num_neighbors = num_neighbors
        self._ctx = {}  # slide name -> (embeddings (N, D), neighbors (N, K))
        if mode == 'inference':
            self.load_level = 'slide'

    def _slide_context(self, name):
        if name not in self._ctx:
            emb, coords = self.load_emb(name, return_crds=True)
            pos = coords.numpy()
            n = len(pos)
            k = min(self.num_neighbors, n)
            graph = kneighbors_graph(pos, n_neighbors=k, include_self=True).toarray()
            neighbors = torch.as_tensor(graph.nonzero()[1].reshape(n, k), dtype=torch.long)
            self._ctx[name] = (emb.float(), neighbors)
        return self._ctx[name]

    def __len__(self):
        if self.mode == 'inference':
            return 1
        return len(self.int2id)  # train and test both iterate slides

    def __getitem__(self, index):
        if self.mode == 'inference':
            emb, neighbors = self._slide_context(self.name)
            return {'img_emb': emb, 'neighbors': neighbors}

        name = self.int2id[index]
        emb, neighbors = self._slide_context(name)
        if self.phase == 'train':
            expression = self.adata_dict[name].X
        else:
            expression = self.load_st(name, self.genes, **self.norm_param).X
        expression = expression.toarray() if sparse.issparse(expression) else np.asarray(expression)
        return {'img_emb': emb, 'neighbors': neighbors, 'label': torch.FloatTensor(expression)}
