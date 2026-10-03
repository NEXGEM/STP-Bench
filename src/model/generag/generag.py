import torch
import torch.nn as nn
import torch.nn.functional as F

from .solvers import fista_elasticnet, op_norm_sq


class GeneRAG(nn.Module):
    """GeneRAG (MICCAI 2026): retrieval-augmented full-transcriptome prediction.

    Backbone = a linear head on frozen patch embeddings (`DATA.feature_type` embeddings, `uni_v2` by
    default) trained on the *anchor* genes only. Two stages:

      1. train / val: the decoder is trained on the anchor genes and validated on them (no retrieval),
         so the checkpoint is selected by the decoder's own anchor-gene quality.
      2. test / predict: the decoder is frozen; each spot's anchor prediction ``y_init`` and its
         embedding ``f_img`` form a hybrid query into a reference bank of training spots; one sparse
         code ``alpha`` (Eq. 1) is solved and applied to the full bank expression (Eq. 2)::

        alpha = argmin  w||f_img - D_img a||^2 + (1-w)||y_anchor - D_anchor a||^2
                        + gamma*lam||a||^2 + (1-gamma)*lam||a||_1
        y_full = D_full a

    The bank is not a parameter: `GeneRAGAdapter` builds it from the eval
    dataset and hands it over via `set_bank()`. It is deliberately kept out of
    the state_dict (a bank is ~N_spots x N_genes floats).

    Deviation from the reference: the reference keeps only the
    ``n_high_variable_genes`` highest-variance bank genes. Here the output
    panel is always the whole `DATA` gene panel (the benchmark scores against
    it), so choose the panel via `DATA.gene_type` / `DATA.num_genes` instead.
    """

    def __init__(self,
                 num_genes=200,
                 img_embedding_dim=1536,
                 num_anchor_genes=200,
                 max_batch_size=1024,
                 non_negative_output: bool = True,
                 alpha=0.01,
                 l1_ratio=0.9,
                 embedding_ratio=0.75,
                 positive: bool = True,
                 max_iter=2000,
                 tol=1e-4):
        super().__init__()
        # num_genes (full output panel) is injected from the data config but
        # not needed: the head only predicts the anchor genes.
        self.max_batch_size = max_batch_size
        self.non_negative_output = non_negative_output
        self.alpha, self.l1_ratio = alpha, l1_ratio
        self.embedding_ratio = float(min(max(embedding_ratio, 0.0), 1.0))
        self.positive, self.max_iter, self.tol = positive, max_iter, tol

        self.head = nn.Linear(img_embedding_dim, num_anchor_genes)

        self._bank = None      # CPU tensors from the adapter
        self._bank_dev = None  # device-resident solver state, built lazily

    # ------------------------------------------------------------------ bank

    def set_bank(self, expression, embeddings, anchor_idx):
        """expression (N, G), embeddings (N, d), anchor_idx (n_anchor,) into G."""
        self._bank = {'expression': expression, 'embeddings': embeddings,
                      'anchor_idx': torch.as_tensor(anchor_idx, dtype=torch.long)}
        self._bank_dev = None

    @property
    def has_bank(self):
        return self._bank is not None

    def _prepare_bank(self, device):
        cached = self._bank_dev
        if cached is not None and cached['device'] == device:
            return cached
        w = self.embedding_ratio
        expr = self._bank['expression'].to(device=device, dtype=torch.float32)   # (N, G)
        blocks = []
        if w < 1.0:
            blocks.append((1.0 - w) ** 0.5 * expr[:, self._bank['anchor_idx'].to(device)].T)
        if w > 0.0:
            blocks.append(w ** 0.5 * self._bank['embeddings'].to(device=device, dtype=torch.float32).T)
        D = torch.cat(blocks, dim=0) if len(blocks) > 1 else blocks[0]           # (M, N)
        self._bank_dev = {'device': device, 'expr': expr, 'D': D, 'op_sq': op_norm_sq(D)}
        return self._bank_dev

    # ------------------------------------------------------------- retrieval

    @torch.no_grad()
    def retrieve(self, anchor, img_emb):
        """Solve Eq. 1 for every query spot and reconstruct Eq. 2 -> (K, G).

        Query columns are independent, so blocks of `max_batch_size` spots are
        solved separately to bound the (N_bank x K) solver state; FISTA's
        stopping check is global per block, so results can differ from a
        single-block solve by the solver tolerance.
        """
        if self._bank is None:
            raise RuntimeError("GeneRAG reference bank is not set; use adapter 'generag'.")
        # This is a numerical solver and must run in float32: on real banks ||D||^2 is ~1e7, above
        # float16's maximum (65504), so under the trainer's default 16-mixed autocast the FISTA step
        # size and every iterate become NaN. Disable autocast for the whole retrieval.
        with torch.autocast(device_type=anchor.device.type, enabled=False):
            bank = self._prepare_bank(anchor.device)
            w = self.embedding_ratio
            outs = []
            for a, e in zip(anchor.split(self.max_batch_size, dim=0),
                            img_emb.split(self.max_batch_size, dim=0)):
                blocks = []
                if w < 1.0:
                    blocks.append((1.0 - w) ** 0.5 * a.float().T)
                if w > 0.0:
                    blocks.append(w ** 0.5 * e.float().T)
                B = torch.cat(blocks, dim=0) if len(blocks) > 1 else blocks[0]        # (M, K)
                W = fista_elasticnet(bank['D'], B, self.alpha, self.l1_ratio,
                                     positive=self.positive, max_iter=self.max_iter,
                                     tol=self.tol, op_sq=bank['op_sq'])              # (N, K)
                outs.append(W.T @ bank['expr'])                                       # (K, G)
            return torch.cat(outs, dim=0)

    # --------------------------------------------------------------- forward

    def _head(self, img_emb):
        out = self.head(img_emb)
        return F.softplus(out) if self.non_negative_output else out

    def forward(self, img_emb=None, label=None, **kwargs):
        if img_emb is None:
            raise ValueError("Image embeddings must be provided for GeneRAG.")
        phase = kwargs.get('phase', 'train')

        if phase == 'train':
            # label is anchor-width here (GeneRAGDataset slices it).
            anchor = self._head(img_emb)
            result = {'logits': anchor}
            if label is not None:
                result['loss'] = F.mse_loss(anchor, label)
            return result

        anchor = torch.cat([self._head(c) for c in img_emb.split(self.max_batch_size, dim=0)], dim=0)
        if phase == 'val':
            # stage 1: validate the decoder on the anchor genes (GeneRAGAdapter scores them); no retrieval
            return {'logits': anchor}
        # stage 2 (test / predict): decoder frozen, retrieval reconstructs the full panel
        return {'logits': self.retrieve(anchor, img_emb)}
