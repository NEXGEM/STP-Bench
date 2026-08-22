"""Pathway gene-set loading, ported from
analysis/src/figure4_geneset.py::_read_gmt / _load_gene_sets in the main
STpredBench repo."""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def read_gmt(path: str) -> Dict[str, List[str]]:
    """Parse a .gmt pathway file: `term\tdescription\tgene1\tgene2\t...` per line."""
    gene_sets: Dict[str, List[str]] = {}
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                gene_sets[parts[0]] = parts[2:]
    return gene_sets


def load_gene_sets(
    library: str,
    measured_genes: Sequence[str],
    min_overlap: int | float = 5,
    cache_dir: str = "output/downstream_cache/genesets",
) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    """Load an Enrichr library (by name, cached as pickle) or a local .gmt
    file, then keep only pathways with enough of their genes measured.

    `min_overlap`: int = fixed minimum gene count. float in (0, 1) = at
    least that fraction of the pathway's raw gene count, AND at least 5
    genes regardless (both conditions must hold).

    Returns (filtered_gene_sets, overlap_counts) — gene names in
    filtered_gene_sets are drawn from `measured_genes` (case-insensitively
    matched against the library's own gene symbols).
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path / f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', library)}.pkl"

    if Path(library).exists():
        raw_gene_sets = read_gmt(library)
    elif cache_file.exists():
        with open(cache_file, "rb") as f:
            raw_gene_sets = pickle.load(f)
    else:
        import gseapy as gp

        raw_gene_sets = gp.get_library(name=library, organism="Human")
        with open(cache_file, "wb") as f:
            pickle.dump(raw_gene_sets, f)

    measured_genes = list(measured_genes)
    upper_to_gene = {g.upper(): g for g in measured_genes}
    filtered: Dict[str, List[str]] = {}
    overlap_counts: Dict[str, int] = {}

    for term, genes in raw_gene_sets.items():
        overlap = []
        seen = set()
        for gene in genes:
            actual = upper_to_gene.get(str(gene).upper())
            if actual is not None and actual not in seen:
                overlap.append(actual)
                seen.add(actual)
        if isinstance(min_overlap, float) and 0 < min_overlap < 1:
            threshold = max(5, int(len(genes) * min_overlap))
        else:
            threshold = min_overlap
        if len(overlap) >= threshold:
            filtered[term] = overlap
            overlap_counts[term] = len(overlap)

    return filtered, overlap_counts
