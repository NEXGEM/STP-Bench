"""Suppress noisy library output during STPBench pipeline runs.

Usage:
    with suppress_library_output():
        run_cross_validation(cfg, dm)

Everything printed by third-party libraries (PyTorch, Lightning, HuggingFace,
scanpy, …) is silenced while inside the block.  STPBench's own BenchmarkLogger
writes directly to sys.__stdout__ so it remains visible.  Stderr is preserved
so that tqdm progress bars and the Lightning model-summary table still appear.

Errors and warnings are caught, formatted into user-friendly hints, and
re-raised so the caller's section() handler can log them cleanly.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from contextlib import contextmanager
from typing import Optional


# ---------------------------------------------------------------------------
# Logger groups
# ---------------------------------------------------------------------------

_QUIET_LOGGERS = [
    "lightning",
    "lightning.pytorch",
    "lightning.fabric",
    "pytorch_lightning",
    "torch",
    "torch.distributed",
    "transformers",
    "diffusers",
    "timm",
    "datasets",
    "filelock",
    "fsspec",
    "PIL",
    "scanpy",
    "anndata",
    "h5py",
    "numba",
    "umap",
]

# These children must stay at INFO so their messages still reach handlers
# (Lightning model-summary is routed through this logger to stderr)
_KEEP_INFO_LOGGERS = [
    "lightning.pytorch.callbacks.model_summary",
]


# ---------------------------------------------------------------------------
# User-friendly error hints
# ---------------------------------------------------------------------------

_HINT_RULES: list[tuple[type, str, str]] = [
    (
        FileNotFoundError,
        "exemplar",
        "Exemplar files are missing — run preprocessing (preprocess() or --mode preprocess) "
        "with the same model and data config before evaluating.",
    ),
    (
        FileNotFoundError,
        ".ckpt",
        "Checkpoint file not found — verify the --ckpt_path or run training first.",
    ),
    (
        FileNotFoundError,
        ".h5",
        "Feature embedding file (.h5) not found — "
        "run feature extraction for this encoder before training or evaluating.",
    ),
    (
        FileNotFoundError,
        ".json",
        "Gene-list JSON file not found — run preprocessing to generate gene sets.",
    ),
    (
        FileNotFoundError,
        "splits",
        "Data split CSV not found — run preprocessing (split_data step) first.",
    ),
    (
        RuntimeError,
        "out of memory",
        "CUDA out of memory — try reducing batch size or using a GPU with more VRAM.",
    ),
    (
        RuntimeError,
        "size mismatch",
        "Model size mismatch when loading checkpoint — "
        "make sure the model config matches the checkpoint.",
    ),
    (
        RuntimeError,
        "mat1 and mat2 shapes",
        "Tensor shape mismatch during forward pass — "
        "check that emb_dim in the model config matches the feature encoder output dimension.",
    ),
    (
        KeyError,
        "",
        "A required key is missing in the data or config — "
        "check that all config fields are set correctly.",
    ),
    (
        ImportError,
        "",
        "A required package is not installed — run 'uv sync' to install all dependencies.",
    ),
]


def parse_error_hint(exc: BaseException) -> Optional[str]:
    """Return a user-friendly hint string for a known exception pattern, or None."""
    msg = str(exc).lower()
    for exc_type, keyword, hint in _HINT_RULES:
        if isinstance(exc, exc_type) and (not keyword or keyword.lower() in msg):
            return hint
    return None


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

@contextmanager
def suppress_library_output():
    """Redirect stdout to /dev/null and quiet library loggers for the duration."""

    # --- stdout: redirect to /dev/null so library print() calls vanish ---
    devnull = open(os.devnull, "w")
    old_stdout = sys.stdout
    sys.stdout = devnull

    # --- logging: push WARNING onto every noisy logger ---
    saved_levels: dict[str, Optional[int]] = {}
    for name in _QUIET_LOGGERS:
        lg = logging.getLogger(name)
        saved_levels[name] = lg.level
        lg.setLevel(logging.WARNING)

    # Keep model-summary logger at INFO so its table still appears on stderr
    for name in _KEEP_INFO_LOGGERS:
        lg = logging.getLogger(name)
        saved_levels[name] = lg.level
        lg.setLevel(logging.INFO)

    # --- warnings: suppress common noisy categories ---
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Quiet HuggingFace via env var (affects subprocess spawns too)
        old_tf_verbosity = os.environ.get("TRANSFORMERS_VERBOSITY")
        old_tok_verbosity = os.environ.get("TOKENIZERS_PARALLELISM")
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        try:
            yield
        finally:
            # Restore everything in reverse order
            sys.stdout = old_stdout
            devnull.close()

            for name, level in saved_levels.items():
                lg = logging.getLogger(name)
                if level is None:
                    lg.setLevel(logging.NOTSET)
                else:
                    lg.setLevel(level)

            if old_tf_verbosity is None:
                os.environ.pop("TRANSFORMERS_VERBOSITY", None)
            else:
                os.environ["TRANSFORMERS_VERBOSITY"] = old_tf_verbosity

            if old_tok_verbosity is None:
                os.environ.pop("TOKENIZERS_PARALLELISM", None)
            else:
                os.environ["TOKENIZERS_PARALLELISM"] = old_tok_verbosity
