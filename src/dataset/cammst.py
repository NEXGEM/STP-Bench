import numpy as np
import torch
from scipy import sparse
from scipy.spatial import cKDTree

from dataset.base_dataset import STDataset


def local_bio_salience_score(counts, coords, k_neighbors=8):
    """Local bio-salience score of every spot (port of CAMMST `UNIFeatureDataset._compute_bio_scores`,
    https://github.com/Kyyle2114/CAMMST, `utils/data.py`): the L2 deviation of a spot's gene-wise
    z-scored expression from the mean of its `k_neighbors` nearest spots, gated by a sigmoid of the
    spot's total expression. It is the target of the bio-salience sampling loss (training only).
    """
    mean = counts.mean(axis=0, keepdims=True)
    std = counts.std(axis=0, keepdims=True)
    reg_term = 0.1 * np.percentile(std, 50) + 1e-8
    normalized = (counts - mean) / (std + reg_term)

    effective_k = min(k_neighbors, len(coords) - 1)
    if effective_k > 0:
        _, indices = cKDTree(coords).query(coords, k=effective_k + 1)  # first column is the spot itself
        local_means = normalized[indices[:, 1:]].mean(axis=1)
        deviation = np.linalg.norm(normalized - local_means, axis=1)
    else:  # single-spot slide: fall back to the deviation from the slide mean
        deviation = np.linalg.norm(normalized - normalized.mean(axis=0, keepdims=True), axis=1)

    total = counts.sum(axis=1)
    weights = 1 / (1 + np.exp(-0.05 * (total - np.percentile(total, 20))))
    return torch.tensor(deviation * weights, dtype=torch.float32)


class CAMMSTDataset(STDataset):
    """STDataset arranged slide by slide for CAMMST (one item = one slide).

    Each item is `img_emb` (N, D) UNI2-h embeddings, `coord` (N, 2) spot coordinates and `label` (N, G);
    training items also carry `bio_salience_score` (N,). A slide is never padded or truncated. The
    embeddings and coordinates are read once per slide and cached.
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
                 k_neighbors: int = 8,
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
        self.k_neighbors = k_neighbors
        self._ctx = {}  # slide name -> (embeddings (N, D), coords (N, 2))
        if mode == 'inference':
            self.load_level = 'slide'

    def _slide_context(self, name):
        if name not in self._ctx:
            emb, coords = self.load_emb(name, return_crds=True)
            self._ctx[name] = (emb.float(), torch.as_tensor(coords).float())
        return self._ctx[name]

    def __len__(self):
        if self.mode == 'inference':
            return 1
        return len(self.int2id)  # train and test both iterate slides

    def __getitem__(self, index):
        if self.mode == 'inference':
            emb, coords = self._slide_context(self.name)
            return {'img_emb': emb, 'coord': coords}

        name = self.int2id[index]
        emb, coords = self._slide_context(name)
        if self.phase == 'train':
            expression = self.adata_dict[name].X
        else:
            expression = self.load_st(name, self.genes, **self.norm_param).X
        expression = expression.toarray() if sparse.issparse(expression) else np.asarray(expression)

        item = {'img_emb': emb, 'coord': coords, 'label': torch.FloatTensor(expression)}
        if self.phase == 'train':
            item['bio_salience_score'] = local_bio_salience_score(expression, coords.numpy(), self.k_neighbors)
        return item
