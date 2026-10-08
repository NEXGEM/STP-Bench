import numpy as np
import torch
from scipy import sparse
from scipy.spatial import KDTree

from dataset.base_dataset import STDataset


class HisDiffDataset(STDataset):
    """STDataset + HisDiff's hierarchical spot context.

    For every spot, the embeddings of its `num_neighbors` (9) and `num_global` (49) nearest spots of the
    same slide (spatial kNN on the spot coordinates, the spot itself included), exactly as the reference
    `preprocess.ipynb` builds `neighbor_ebd` / `global_ebd` from `local_ebd`. They are derived from the
    slide's own per-spot embeddings (`emb/global/features_<model_name>`, which also hold the coordinates),
    so no extra preprocessing artifact is needed.

    * train (patch level): `img_emb` (D), `neighbor_emb` (9, D), `global_emb` (49, D), `label`.
      Spots whose expression is all zero/NaN are dropped, as in the reference.
    * test / inference (whole slide): `img_emb` (N, D), `neighbor_idx` (N, 9), `global_idx` (N, 49)
      (indices into `img_emb`; the model gathers them on the device), plus `label` when available.
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
                 num_neighbors: int = 9,
                 num_global: int = 49,
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
        self.num_global = num_global
        self._ctx = {}  # slide name -> (embeddings (N, D), neighbor idx (N, Kn), global idx (N, Kg))

        if mode == 'inference':
            # the context needs the whole slide, so an inference "item" is the whole slide
            self.load_level = 'slide'

        if phase == 'train':
            names = [self.int2id[i] for i in range(len(self.int2id))]
            valid = []
            for si, name in enumerate(names):
                X = self.adata_dict[name].X
                rows = np.asarray(X.sum(axis=1)).ravel() if sparse.issparse(X) else np.nansum(X, axis=1)
                has_nan_only = np.isnan(X.toarray() if sparse.issparse(X) else X).all(axis=1)
                keep = np.where((rows != 0) & ~has_nan_only)[0]
                valid.append(np.stack([np.full(len(keep), si), keep], axis=1))
                self._slide_context(name)
            self._index = np.concatenate(valid, axis=0)
            dropped = int(self.cumlen[-1]) - len(self._index)
            if dropped:
                print(f"HisDiffDataset: dropped {dropped} spots with all-zero/NaN expression")

    # ---------------------------------------------------------------- context

    def _knn(self, coords, k):
        """k nearest spots (self first), padded by repeating the farthest one if the slide has < k spots."""
        n = len(coords)
        kk = min(k, n)
        _, idx = KDTree(coords).query(coords, k=kk)
        idx = np.asarray(idx).reshape(n, kk)
        if kk < k:
            idx = np.concatenate([idx, np.repeat(idx[:, -1:], k - kk, axis=1)], axis=1)
        return torch.as_tensor(idx, dtype=torch.long)

    def _slide_context(self, name):
        if name not in self._ctx:
            emb, coords = self.load_emb(name, return_crds=True)
            coords = coords.numpy()
            # same two separate queries as the reference: ties at the k-th distance (common on the Visium
            # lattice) can resolve differently for k=9 and for the first 9 of k=49
            self._ctx[name] = (emb.float(), self._knn(coords, self.num_neighbors), self._knn(coords, self.num_global))
        return self._ctx[name]

    # ------------------------------------------------------------------- data

    def __len__(self):
        if self.phase == 'train' and self.load_level == 'patch':
            return len(self._index)
        return super().__len__()

    def __getitem__(self, index):
        if self.phase == 'train':
            si, idx = self._index[index]
            name = self.int2id[int(si)]
            emb, nb_idx, gl_idx = self._slide_context(name)
            expression = self.adata_dict[name][int(idx)].X
            expression = expression.toarray().squeeze(0) if sparse.issparse(expression) else expression.squeeze(0)
            return {'img_emb': emb[int(idx)],
                    'neighbor_emb': emb[nb_idx[int(idx)]],
                    'global_emb': emb[gl_idx[int(idx)]],
                    'label': torch.FloatTensor(expression)}

        name = self.name if self.mode == 'inference' else self.int2id[index]
        emb, nb_idx, gl_idx = self._slide_context(name)
        data = {'img_emb': emb, 'neighbor_idx': nb_idx, 'global_idx': gl_idx}
        if self.mode != 'inference':
            adata = self.load_st(name, self.genes, **self.norm_param)
            expression = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
            data['label'] = torch.FloatTensor(expression)
        return data
