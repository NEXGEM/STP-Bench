"""Batched FISTA ElasticNet solver for GeneRAG's Dual-Constrained Retrieval.

Ported (torch-only, GPU or CPU) from https://github.com/HyeongSubKim/GeneRAG
(`generag/solvers.py`, MIT License). The scikit-learn fallback and the
CPU-only methods of the reference (omp/nnls/lassolars/bayesian_ridge) are not
carried over: the same FISTA iteration runs on CPU tensors when CUDA is
unavailable.
"""
import torch


def op_norm_sq(D: torch.Tensor, n_iter: int = 30) -> float:
    """Squared spectral norm of ``D`` via power iteration (FISTA Lipschitz constant)."""
    v = torch.randn(D.shape[1], device=D.device, dtype=D.dtype)
    v = v / (v.norm() + 1e-12)
    for _ in range(n_iter):
        v = D.T @ (D @ v)
        nv = v.norm()
        if nv < 1e-20:
            return 0.0
        v = v / nv
    Dv = D @ v
    return float((Dv @ Dv).item())


def fista_elasticnet(D, B, alpha, l1_ratio, positive=True, max_iter=2000, tol=1e-4,
                     check_every=25, op_sq=None):
    """Solve ``min_W 1/(2M)||DW-B||^2 + alpha*(1-l1_ratio)/2 ||W||^2 + alpha*l1_ratio ||W||_1``.

    Matches scikit-learn's ElasticNet loss (``M`` = rows of ``D``), jointly over
    every column of ``B``. Returns ``W`` of shape ``(N, K)``.

    ``op_sq`` lets the caller reuse a cached ``op_norm_sq(D)`` across calls.
    """
    M, N = D.shape
    l1, l2 = alpha * l1_ratio, alpha * (1.0 - l1_ratio)
    if op_sq is None:
        op_sq = op_norm_sq(D)
    step = 1.0 / max(op_sq / M + l2, 1e-12)
    thresh = step * l1

    W = torch.zeros(N, B.shape[1], device=D.device, dtype=D.dtype)
    Z, t = W.clone(), 1.0
    for it in range(max_iter):
        W_prev = W
        grad = D.T @ (D @ Z - B) / M
        if l2 != 0.0:
            grad = grad + l2 * Z
        V = Z - step * grad
        if positive:
            W = torch.clamp(V - thresh, min=0.0)
        else:
            W = torch.sign(V) * torch.clamp(V.abs() - thresh, min=0.0)
        t_new = 0.5 * (1.0 + (1.0 + 4.0 * t * t) ** 0.5)
        Z = W + ((t - 1.0) / t_new) * (W - W_prev)
        t = t_new
        if (it + 1) % check_every == 0:
            if float((W - W_prev).abs().max() / W.abs().max().clamp_min(1e-12)) < tol:
                break
    return W
