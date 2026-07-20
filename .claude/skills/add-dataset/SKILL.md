---
name: add-dataset
description: Register and preprocess a new dataset (raw WSIs + spatial transcriptomics data, or a HEST-formatted cohort) for STP-Bench. Use when the user asks to add a new dataset, onboard raw slides/ST data for benchmarking, or set up an internal/external evaluation cohort ("add <name> dataset").
---

# Adding a New Dataset to STP-Bench

Full narrative reference: `docs/guide.md#adding-a-new-dataset`. This skill is
the actionable checklist.

## Step 0 — Get the raw data from the user

STP-Bench doesn't ship raw WSIs or ST data — the user must provide their own
cohort. Before writing a data config, ask where it already lives, in order of
preference:

1. **A local filesystem path** to the per-sample raw directories (or an
   already-HEST-processed root) — almost always the right answer for WSI/ST
   data, since these datasets are typically already sitting on the same
   machine or shared storage the pipeline runs on (too large to move
   casually). This becomes `preprocess.input_dir`.
2. **A download URL or accession** if the data isn't local yet — a
   HuggingFace dataset repo (see `snapshot_download` in the README's
   Benchmark Data section), a GEO/SRA/Zenodo accession, or a cloud bucket
   link. Confirm the destination path to download into before pulling
   anything, and don't download large raw data without the user's
   confirmation of where it should land.

Once you have a path, verify the actual layout on disk (`ls`/`find`) against
the expected per-sample structure below **before** writing the data config —
don't assume it already matches; raw exports from SpaceRanger/Xenium tooling
vary in nesting.

## Step 1 — Raw input layout

One subdirectory per sample under a common `input_dir`, each holding the WSI
(`.tiff`/`.svs`/`.ndpi`) plus the platform's native ST output (SpaceRanger for
Visium, Xenium output, etc.) — follows
[HEST](https://github.com/mahmoodlab/HEST) conventions.

```
input_dir/
├── sample_A/
│   ├── *.tiff
│   └── ...            # SpaceRanger / Xenium output
├── sample_B/
│   └── ...
```

## Step 2 — Data config

`config/data/<namespace>/<name>.yaml` — generate a template with
`STPred.init_data_config("<namespace>/<name>")`, then fill it in. Minimum
required blocks: `GENERAL`, `TRAINING`, `DATA`, `preprocess` (full annotated
example in `docs/guide.md`). Fields most likely to be gotten wrong:

- `DATA.meta_dir` — where `ids.csv` and the gene-set JSON get written. Keep
  this dataset-specific even when `DATA.data_dir` is a root shared across
  every dataset in the repo (the established convention here) — reusing
  another dataset's `meta_dir` silently cross-contaminates fold splits and
  gene panels.
- `DATA.gene_type` / `num_genes` — every existing config under
  `config/data/` uses `gene_type: hmhvg` with `num_genes: 200`. Match this
  unless there's a specific reason to diverge, since several models'
  configs assume a 200-gene panel.
- `preprocess.mode` — `raw` vs `stpbench`, see below. Getting this wrong
  either wastes hours re-extracting patches that already exist, or silently
  reads nothing.
- `preprocess.platform` — `visium` / `xenium` / `merfish` / ... must match
  what the raw data actually is; this selects the HEST reader.

## `raw` vs `stpbench` mode

- **`raw`** — `preprocess.input_dir` points at your per-sample raw
  directories; the pipeline extracts patches + expression into
  `DATA.data_dir` from scratch. Use this for genuinely new data.
- **`stpbench`** — `preprocess.input_dir` points at an already-HEST-processed
  root (`patches/`, `st/`, `wsis/` already present, e.g. downloaded from the
  `nexgem/STP-Bench` HF dataset). Every config currently under
  `config/data/` uses this mode. Don't use `raw` for data that's already
  been through this pipeline once.

**This is a one-time choice, not a pipeline stage you progress through.**
It only describes what format the *raw* data is in for the initial ingestion
step — there is no "run `raw` once, then switch to `stpbench`" workflow for
the same dataset. Both modes write to the identical output layout
(`data_dir/patches/`, `data_dir/st/`, `meta_dir/ids.csv`); every step after
the first `stp.preprocess()` call — gene set prep, CV splits, feature
extraction, training, evaluation, even a later re-run of `preprocess()`
itself — reads only that resulting layout and never inspects
`preprocess.mode` again (verified: no reference to it outside
`src/api/data_pipeline.py`'s initial ingestion). Once a user's own raw data
has been ingested with `mode: raw`, leave it at `raw` — don't "upgrade" the
config to `stpbench` afterward.

## Step 3 — Run preprocessing

```python
stp = STPred(models=["StNet"])   # any cheap model — preprocess() covers every configured model
stp.preprocess(data="<namespace>/<name>")
```

Runs, in order: raw preprocessing (patches/expression) → gene set selection
(`<meta_dir>/<gene_type>_<num_genes>genes.json`) → CV split assignment
(`<meta_dir>/ids.csv` with `fold_*` columns) → feature extraction
(`<data_dir>/emb/<feature_type>/features_<model_name>/`).

**Before running a long preprocessing job**, dry-run it:

```python
stp.check(data="<namespace>/<name>", strict=False)
```

This catches missing/misconfigured paths before committing GPU time.

## Verify before declaring done

- `<meta_dir>/ids.csv` has a `sample_id` column plus one `fold_N` column per
  requested fold, and every raw sample appears exactly once.
- `<meta_dir>/<gene_type>_<num_genes>genes.json` exists with exactly
  `num_genes` entries.
- `<data_dir>/patches/<sample_id>.h5` and `<data_dir>/st/<sample_id>.h5ad`
  exist for every sample.
- Run an actual train + evaluate on one cheap model end to end
  (`stp.train(...)`, `stp.evaluate_internal()`) — don't stop at "preprocessing
  exited 0", since a job can complete without actually processing every
  sample (per-sample exceptions are sometimes caught and logged rather than
  raised).
- If this dataset will ever serve as an **external** eval target for models
  trained elsewhere, also smoke-test
  `stp.evaluate_external(data="<namespace>/<name>", train_data=<existing_trained_data>)`.
  External evaluation exercises a distinct, historically bug-prone code path
  (gene-panel overlap resolution between the training and external panels,
  `meta_dir` vs `data_dir` scoping, per-model reference-bank/cache
  namespacing) that internal-only preprocessing never touches.
