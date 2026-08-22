"""Downstream analyses (gene enrichment, deconvolution, spatial domain) that
run on top of STPred's ST predictions.

Each mode subpackage (`gene_enrichment`, `deconvolution`, `spatial_domain`)
exposes a `run_<mode>(cfg)` and `evaluate_<mode>(cfg)` entrypoint. Heavy
third-party dependencies (gseapy, cell2location, SpaGCN) are imported lazily
inside those subpackages, not here, so importing `downstream` never requires
all three to be installed.
"""
