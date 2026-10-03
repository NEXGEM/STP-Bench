import numpy as np
import torch
from scipy import sparse
from scipy.spatial import cKDTree

from dataset.base_dataset import STDataset


class NH2STDataset(STDataset):
    """STDataset for NH2ST: raw augmented patches as `img`, plus (training only) the K nearest spots of each spot.

    A training item adds `nbr_emb` (K, 1536) -- the cached `uni_v2` embeddings of the spot and its K-1 nearest
    spots -- and `nbr_exp` (K, G) -- their expression, as `NGHist2ST.create_hypergraph` (the K nearest spots by
    centre distance, the spot itself included). Evaluation items are the plain image / label of STDataset: the
    model predicts from the image alone, so no neighbor expression is ever read at test time.
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
                 use_emb: bool = False,
                 genes_override: list = None,
                 k_neighbors: int = 9,
                 ):
        # use_emb stays False for the base class: the model needs no per-item embedding of the target spot,
        # the neighbors' embeddings are read once per slide below.
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
                         use_emb=False,
                         genes_override=genes_override)
        self.k_neighbors = k_neighbors
        self._slide = {}  # slide name -> (embeddings (N, D) tensor, neighbor indices (N, K))

    def _slide_neighbors(self, name):
        if name not in self._slide:
            emb, coords = self.load_emb(name, emb_name='global', return_crds=True)
            coords = np.asarray(coords)
            n = len(coords)
            k = min(self.k_neighbors, n)
            _, idx = cKDTree(coords).query(coords, k=k)
            idx = np.asarray(idx).reshape(n, k)
            if k < self.k_neighbors:  # a slide smaller than K spots: repeat the nearest ones so K stays fixed
                idx = idx[:, np.arange(self.k_neighbors) % k]
            self._slide[name] = (emb.float(), idx)
        return self._slide[name]

    def __getitem__(self, index):
        data = super().__getitem__(index)
        if self.phase != 'train':
            return data

        i = 0
        while index >= self.cumlen[i]:
            i += 1
        idx = index - self.cumlen[i - 1] if i > 0 else index
        name = self.int2id[i]

        emb, nbrs = self._slide_neighbors(name)
        nbr = nbrs[idx]
        expression = self.adata_dict[name].X[nbr]
        expression = expression.toarray() if sparse.issparse(expression) else np.asarray(expression)

        data['nbr_emb'] = emb[nbr]
        data['nbr_exp'] = torch.FloatTensor(expression)
        return data
