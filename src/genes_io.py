"""Gene-set files of a meta_dir.

<meta_dir>/genes.json holds every gene set of the dataset:
    {"sets": {"hmhvg_200": {"genes": [...]}, "total": {...}, "eval_fixed200": {"genes": [...], "samples": [...], ...}}}
Key = "<gene_type>_<num_genes>" (total: "total"). Evaluation-only dirs (cross_external/<condition>/<pair>) hold one
set "eval": {"genes", "train_set", "test_set", "condition", "panel", "samples"}.
A gene set is addressed everywhere as one string, "<meta_dir>/genes.json#<key>" (see gene_ref / read_genes_path).
A *_holdout meta_dir shares the genes.json of its base dir (symlink).
"""
import json
import os
from contextlib import contextmanager

REF_SEP = "#"
EVAL_KEY = "eval"


def genes_json(meta_dir):
    return os.path.join(meta_dir, "genes.json")


def _load(path):
    with open(path) as f:
        return json.load(f)


def _sets(meta_dir):
    p = genes_json(meta_dir)
    return _load(p)["sets"] if os.path.isfile(p) else None


def set_key(gene_type, num_genes=None):
    if gene_type == "total" or num_genes is None:
        return gene_type
    return f"{gene_type}_{int(num_genes)}"


def gene_ref(meta_dir, gene_type, num_genes=None):
    """'<meta_dir>/genes.json#<key>' -- the address of one gene set."""
    return f"{genes_json(meta_dir)}{REF_SEP}{set_key(gene_type, num_genes)}"


def gene_ref_in(dirs, gene_type, num_genes=None):
    """gene_ref of the first dir that has the set (else of the first dir, so the caller can raise a clear error)."""
    dirs = [d for d in dirs if d]
    for d in dirs:
        if has_genes(d, gene_type, num_genes):
            return gene_ref(d, gene_type, num_genes)
    return gene_ref(dirs[0], gene_type, num_genes)


def find_set(meta_dir, gene_type, num_genes=None):
    """The gene-set dict ({"genes": [...], ...}) or None."""
    s = _sets(meta_dir)
    return None if s is None else s.get(set_key(gene_type, num_genes))


def has_genes(meta_dir, gene_type, num_genes=None):
    return find_set(meta_dir, gene_type, num_genes) is not None


def read_genes(meta_dir, gene_type, num_genes=None):
    d = find_set(meta_dir, gene_type, num_genes)
    if d is None:
        raise FileNotFoundError(f"gene set {set_key(gene_type, num_genes)!r} not found in {genes_json(meta_dir)}")
    return d["genes"]


def read_eval(meta_dir, panel):
    """Evaluation gene set dict of an internal dataset (key eval_<panel>, e.g. panel='fixed200')."""
    d = find_set(meta_dir, f"eval_{panel}")
    if d is None:
        raise FileNotFoundError(f"evaluation gene set eval_{panel} not found in {genes_json(meta_dir)}")
    return d


def read_eval_genes(meta_dir):
    """Explicit evaluation gene list of an evaluation-only dir (cross_external pair: set "eval"), else None."""
    d = find_set(meta_dir, EVAL_KEY)
    return None if d is None else d["genes"]


def available_sets(meta_dir):
    """Keys of the sets in meta_dir's genes.json."""
    s = _sets(meta_dir)
    return [] if s is None else list(s)


def _split_ref(ref):
    path, sep, key = ref.rpartition(REF_SEP)
    if not sep or os.path.basename(path) != "genes.json" or not key:
        raise ValueError(f"gene set address must look like '<meta_dir>/genes.json#<key>': {ref!r}")
    return os.path.dirname(path) or ".", key


def read_doc_path(ref):
    """Gene-set dict for a '<meta_dir>/genes.json#<key>' address."""
    meta_dir, key = _split_ref(ref)
    s = _sets(meta_dir)
    if s is None:
        raise FileNotFoundError(genes_json(meta_dir))
    if key not in s:
        raise FileNotFoundError(f"gene set {key!r} not found in {genes_json(meta_dir)}")
    return s[key]


def read_genes_path(ref):
    return read_doc_path(ref)["genes"]


def path_exists(ref):
    """True if a '<meta_dir>/genes.json#<key>' address resolves to a gene set."""
    try:
        read_doc_path(ref)
        return True
    except (FileNotFoundError, ValueError):
        return False


@contextmanager
def _lock(path):
    import fcntl
    fd = os.open(os.path.dirname(path), os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def write_set(meta_dir, gene_type, num_genes, genes, **meta):
    """Add / replace one set in <meta_dir>/genes.json (read-modify-write, locked); returns its key."""
    path = os.path.realpath(genes_json(meta_dir))
    key = set_key(gene_type, num_genes)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with _lock(path):
        d = _load(path) if os.path.isfile(path) else {"sets": {}}
        d["sets"][key] = dict(genes=list(genes), **meta)
        tmp = path + f".tmp{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(d, f, indent=1)
        os.replace(tmp, path)
    return key
