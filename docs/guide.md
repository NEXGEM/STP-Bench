# STP-Bench Guide

A complete reference for the `STPred` API: creating an instance, every workflow
method and its parameters, result objects, and the two extension guides
(adding a dataset, adding a model). For install steps and the shortest
possible end-to-end example, see the [README](../README.md) first.

## Contents

- [Creating an STPred Instance](#creating-an-stpred-instance)
- [Core Workflow](#core-workflow)
  - [Validating Configuration](#validating-configuration)
  - [Preprocessing](#preprocessing)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [One-shot Benchmark](#one-shot-benchmark)
  - [Prediction](#prediction)
  - [Easy Inference Directly on a WSI](#easy-inference-directly-on-a-wsi)
  - [Visualizing a Prediction](#visualizing-a-prediction)
- [Result Objects](#result-objects)
- [Resuming and Persisting a Run](#resuming-and-persisting-a-run)
- [Discovery and Config Helpers](#discovery-and-config-helpers)
- [Adding a New Dataset](#adding-a-new-dataset)
- [Adding a New Model](#adding-a-new-model)
- [Claude Code Skills](#claude-code-skills)

## Creating an STPred Instance

Everything else in this guide starts from one `STPred` object:

```python
from stpbench import STPred

stp = STPred(
    models=["LinearProb", "EGN", "BLEEP", "TRIPLEX", "DeepSpot", "StFlow"],   # or a single model name as a string
    repo_root="/path/to/repo",     # where config/, logs/, etc. live
    gpu=1,
    gpu_id=0,
)
```

`STPred(...)` never touches disk at all — it just records settings. Config
files are only resolved when a workflow method (`preprocess()`, `train()`,
...) actually runs.

| Parameter | Default | Meaning |
|---|---|---|
| `models` | *required* | A model config name, or a list of them (`"LinearProb"` or `["LinearProb", "EGN", "BLEEP", "TRIPLEX", "DeepSpot", "StFlow"]`). Every workflow method runs all of these unless overridden per-call (`preprocess(..., models=[...])`, `predict(..., models=[...])`). Must resolve to `config/model/<name>.yaml`. |
| `repo_root` | `"."` | Root directory containing `config/`, `logs/`, etc. All relative paths in configs (`meta_dir`, `log_path`, `output_dir`, ...) resolve against this, not the process's current working directory. Pass it explicitly whenever you run from somewhere other than the repo root. |
| `gpu` | `1` | Number of GPUs to use, starting at `gpu_id`. Models run in parallel across `gpu` GPUs when more than one model/fold is queued. |
| `gpu_id` | `0` | Index of the first GPU. `gpu=2, gpu_id=1` uses physical GPUs 1 and 2. |
| `debug` | `False` | Skips loggers/checkpoint callbacks and only runs sanity-check validation steps — for quickly smoke-testing a model wiring without writing real logs or checkpoints. |
| `dry_run` | `False` | `preprocess()`/`benchmark()` return the computed plan (what would run) instead of executing it. |
| `preprocess_overrides` | `None` | Dict merged into every preprocessing run's config (e.g. `{"overwrite": True}` to always force re-processing across every `preprocess()`/`predict()` call this instance makes). Per-call arguments (`preprocess(..., overwrite=True)`) still take precedence over this when explicitly given. |
| `verbose` | `True` | Print `[STPBench]` progress lines to the console. |
| `log_file` | `None` | Path to also persist structured JSONL events (one line per step, with elapsed seconds) — independent of `verbose`. |
| `wandb` | `False` | Opt-in to online Weights & Biases tracking during training. |
| `wandb_project` | `"ST_prediction"` | W&B project name (only used when `wandb=True`). |

`STPred` also tracks lightweight workflow state as you call methods — the
dataset last used for training (`internal_data`), the checkpoint timestamp per
model/fold, results from each step, and (once you call a WSI-path `predict()`)
the `output_dir` that call wrote into. This is what lets later calls omit
arguments they can infer, e.g. `stp.train()` with no `data` reuses whatever
`preprocess()` most recently ran against, and `stp.visualize()` with no
`output_dir` reuses the last `predict()` call's. See
[Resuming and Persisting a Run](#resuming-and-persisting-a-run) to save/restore
this state across processes.

## Core Workflow

Every method below is a normal Python call on the `stp` instance created
above. All of them accept `data`/config names as exact `config/data/<name>.yaml`
/ `config/model/<name>.yaml` matches — see
[Configuration](../README.md#configuration) in the README for the config file
shape, and [Discovery and Config Helpers](#discovery-and-config-helpers) below
for how to list/inspect what's available.

### Validating Configuration

```python
stp.check(data="ncche/xenium", mode="train", strict=False)
```

Verifies config shape and expected on-disk artifacts *before* committing to a
potentially long run — checks `data_dir`, `meta_dir/ids.csv`, gene-set/CV-split
files (unless `mode="inference"`), and a sample of patch/embedding files per
model. Returns a report dict (`ok`, `missing`, per-model `reports`); pass
`strict=True` to raise `FileNotFoundError` instead of returning a report with
missing paths listed. `mode` is one of `"train"` (default), `"eval"`, or
`"inference"` — `"inference"` skips the gene-set/CV-split checks that don't
apply to unlabeled prediction targets. `preflight` is an alias for `check`.

### Preprocessing

```python
stp.preprocess(data="ncche/xenium")
stp.preprocess(data="ncche/xenium", models=["LinearProb"])   # only prep for one model
stp.preprocess(data="ncche/xenium", overwrite=True)      # force re-processing
```

Runs one deduplicated plan for every configured model: raw preprocessing
(patch/ST extraction), gene-set preparation, cross-validation splits, and
feature (patch-embedding) extraction, skipping any step whose output already
exists unless `overwrite=True`. This is a single pass over the *union* of
what every configured model needs, not one pass per model — if two models
both use `DATA.model_name: uni_v2` with `feature_type: global`, that
embedding is computed once and shared, even though both models ask for it.
Only a model's own `extra_preprocess` step (e.g. EGN/EGGN graph building,
OmiCLIP's similarity matrix — see
[Adding a New Model](#adding-a-new-model)) is inherently model-specific and
still runs once per model that declares one. See
[Adding a New Dataset](#adding-a-new-dataset) for what each step produces and
the raw-vs-`stpbench`-mode distinction. Extra keyword arguments are merged
into the run's `preprocess:` config (e.g. `platform=`, `n_splits=`); pass
`dry_run=True` (or construct `STPred(..., dry_run=True)`) to get the computed
plan back without executing it — the returned plan's `feature_tasks` list is
exactly this deduplicated set of `(patch_encoder, feature_type)` work items.

### Training

```python
stp.train(data="ncche/xenium")
stp.train()   # reuses whatever preprocess()/train() most recently used
```

Trains every configured model on `data`'s cross-validation folds. `data`
defaults to the instance's last-used internal dataset (set by a prior
`preprocess()`/`train()` call), so it can be omitted on a second call against
the same dataset. Returns a `BenchmarkResult` with per-model, per-fold metrics
and updates `stp.state["train_results"]`.

### Evaluation

```python
stp.evaluate_internal(data="ncche/xenium")               # test folds from training
stp.evaluate_external(data="hest/LUAD", train_data="ncche/xenium")
```

- `evaluate_internal(data=None, folds=None, timestamps=None)` — evaluates the
  internal test folds. `data` defaults to the last-trained dataset.
- `evaluate_external(data, train_data=None, folds=None, timestamps=None)` —
  evaluates a **labeled** external dataset (has ground-truth ST expression)
  using checkpoints trained on `train_data` (defaults to the instance's
  internal dataset). If the external dataset is missing base artifacts
  (patches/embeddings), `preprocess()` runs automatically first. Any
  model-specific `extra_preprocess` also re-runs in *external* mode (correct
  train-vs-eval namespacing, e.g. EGN/EGGN reference banks, OmiCLIP similarity
  matrices) rather than treating the external set as a standalone dataset.

Both are thin wrappers over `evaluate(mode, data=None, external_data=None,
folds=None, timestamps=None)` (`mode="int"` or `"ext"`) if you need the
combined form directly. `timestamps` lets you pin a specific training run
(`{"LinearProb": "2026-05-18-12-00-00"}`) instead of the latest one; see
[Resuming and Persisting a Run](#resuming-and-persisting-a-run).

### One-shot Benchmark

```python
result = stp.benchmark(internal_data="ncche/xenium", external_data="hest/LUAD")
```

Runs `preprocess` → `train` → `evaluate_internal`, and if `external_data` is
given, also `preprocess` → `predict` → `evaluate_external` against it — the
whole pipeline in one call. Extra keyword arguments are forwarded to the
`preprocess()` calls. Returns a `BenchmarkResult` wrapping every step's own
result under `result["steps"]`.

### Prediction

```python
stp.predict(
    data,                    # named config, WSI path, or asset dir — see below
    ckpt_path=None,          # explicit checkpoint (dict per-model, or one path for all)
    train_data=None,         # resolve the latest checkpoint from this training run
    folds=None,
    timestamps=None,
    models=None,             # per-call model subset, doesn't mutate stp.models
    output_dir=None,         # required for WSI/asset-dir targets — see below
    gene_list=None,          # restrict output to these genes
    wsi_dir=None,            # only for the WSI-batch-directory form — see below
    overwrite=None,          # force re-extraction for WSI/asset-dir targets
    coordinates=None,        # crop patches at these coordinates instead of tissue-seg tiling
    batch_size=32,           # patches per model forward pass — see below
)
```

Predicts on slide-image-only data (no ground-truth expression needed). A
checkpoint must be resolvable via `ckpt_path`, or via `train_data` (or a prior
`train()`/`from_run()` call on this instance) which picks the latest matching
run automatically. `gene_list` restricts prediction to a user-supplied subset
of the training gene panel — names not found in that panel are dropped with a
warning rather than failing the whole run, and the call raises `ValueError` if
*none* of the requested names are found. `models` overrides which models run
for this call only (`stp.models` itself is unchanged).

`coordinates` is a path to an `.h5ad` (patch centers read from
`obsm['spatial']`) or `.csv` (`x`/`y` columns) file of patch-**center**
coordinates — when given, `predict()` crops exactly those patches instead of
running tissue segmentation + automatic tiling. Only valid when `data` is a
single WSI file (raises `ValueError` for any other kind of target). See
[Easy Inference Directly on a WSI](#easy-inference-directly-on-a-wsi) below
for a full example and the `overwrite=True` caveat when re-predicting into an
`output_dir` that already has patches extracted.

`batch_size` overrides how many patches are batched together per model
forward pass — defaults to 32 (the data config's own `DATA.test_dataloader.
batch_size`, usually 1, is used only if `batch_size=None` is passed
explicitly). Raise or lower it depending on GPU memory and slide size.

`data` accepts four different kinds of target:

| Kind | Example | Needs `output_dir` |
|---|---|---|
| Named config | `"cptac/xenium"` | No |
| Single WSI file | `"/path/to/slide.svs"` | Yes |
| Directory of WSI files | `"/path/to/slides_dir/"` | Yes |
| Already-preprocessed asset dir (has `patches/*.h5`) | `"/path/to/existing_assets"` | Yes |

The last three are covered in detail next.

### Easy Inference Directly on a WSI

For the latter three kinds, `output_dir` says where extracted
patches/embeddings/predictions get written, and STP-Bench runs patch
extraction and feature embedding automatically — no data config to write:

```python
# A single slide file
stp.predict(
    data="/path/to/slide.svs",
    output_dir="/path/to/output",
    train_data="ncche/xenium",   # or ckpt_path="/path/to/checkpoint.ckpt"
)

# A directory of slides -- one prediction per slide, same output_dir.
# wsi_dir overrides where the *original* slide files are looked up from
# later (e.g. by visualize()) if it differs from the directory passed as data.
stp.predict(data="/path/to/slides_dir/", output_dir="/path/to/output", train_data="ncche/xenium")

# Already-preprocessed assets (a directory containing patches/*.h5) --
# output_dir is where predictions (and ids.csv/embeddings, if a chosen
# model needs embeddings not already present) are written; the asset
# directory itself is only ever read, never written to, so it's safe to
# point at something shared or read-only.
stp.predict(data="/path/to/existing_assets", output_dir="/path/to/predictions", train_data="ncche/xenium")

# Restrict output to specific genes; names not in the training panel are
# dropped with a warning rather than failing the whole run
stp.predict(
    data="/path/to/slide.svs", output_dir="/path/to/output", train_data="ncche/xenium",
    gene_list=["GENE1", "GENE2"],
)

# Per-call model override -- doesn't mutate stp.models
stp.predict(data="/path/to/slide.svs", output_dir="/path/to/output", train_data="ncche/xenium", models=["LinearProb"])

# Force re-extraction of patches/embeddings even if output_dir already has them
stp.predict(data="/path/to/slide.svs", output_dir="/path/to/output", train_data="ncche/xenium", overwrite=True)

# Predict at your own patch coordinates instead of automatic tissue-seg
# tiling -- crops exactly these locations, no segmentation run at all.
# coordinates is a path to an .h5ad (patch centers from obsm['spatial'])
# or .csv (x, y columns) file; only valid for a single WSI file.
stp.predict(
    data="/path/to/slide.svs", output_dir="/path/to/output", train_data="ncche/xenium",
    coordinates="/path/to/spots.h5ad",   # or "/path/to/spots.csv"
)
```

The output `.h5ad` includes `obsm['spatial']` patch coordinates, which is what
`visualize()` (next section) plots against. This path works for models using
the default `feature_type` mechanism (StNet, TRIPLEX, DeepSpot, HisToGene,
DeepSpotM, ...); models with `extra_preprocess` (EGN, EGGN, Sepal, OmiCLIP,
M2ORT, M2OST) and SGN/Stem still require the named-config path above.

`coordinates` takes patch **centers**, converted internally to the
non-overlapping top-left-corner format the pipeline stores on disk — no need
to do that math yourself. If `output_dir` already has patches extracted by an
earlier call (e.g. a prior tissue-seg run against the same slide), pass
`overwrite=True` too, or the old patches are reused unchanged rather than
re-cropped at the new coordinates.

### Visualizing a Prediction

```python
path = stp.visualize(
    gene="SFTPB",
    sample="TENX118",
    output_dir=None,   # defaults to the most recent predict() call's output_dir
    model=None,         # disambiguate if more than one model predicted this sample
    save_path=None,     # defaults to <output_dir>/viz/<sample>_<gene>.png
)
```

Renders one gene's predicted expression for one sample as a heatmap and saves
it to a PNG (returns the saved path). If the original WSI can still be
located (via the `predict()` run's own config, so this works automatically
right after a WSI-path `predict()` call above), the heatmap is overlaid on the
slide's own thumbnail with each patch drawn at its true non-overlapping
footprint; otherwise it falls back to a plain spatial scatter plot and prints
a note to stderr rather than failing. If more than one model predicted the
same sample into the same `output_dir`, pass `model=` to pick which one's
prediction to visualize — otherwise `visualize()` raises, listing the
candidates.

```python
stp.predict(data="/path/to/slide.svs", output_dir="/path/to/output", train_data="ncche/xenium")
stp.visualize(gene="SFTPB", sample="TENX118")   # output_dir inferred from the call above
```

## Result Objects

Every workflow method above (except `check()` and `visualize()`, which return
a plain dict / path string) returns a `BenchmarkResult`. They're
dict-compatible and provide:

```python
result.summary()          # per-model results, cross-fold aggregate stats (mean, std, per_fold) per metric
result.to_records()        # flat list of per-fold dicts
result.to_dataframe()       # pandas DataFrame of records
result.best_checkpoints()   # best checkpoint path per model/fold
result.prediction_dirs()    # prediction output directories
result.save("results.csv")  # write records to CSV
```

`benchmark()`'s result additionally exposes each step's own result under
`result["steps"]` (e.g. `result["steps"]["train"]`).

## Resuming and Persisting a Run

**Resume a known training run** without keeping the original Python process
alive — reconstructs the checkpoint-resolution state `predict()`/`evaluate()`
need, without re-running `train()`:

```python
stp = STPred.from_run(
    data="ncche/xenium",
    models=["LinearProb"],
    timestamp="2026-05-18-12-00-00",   # omit to auto-pick each model's latest run
)
stp.evaluate_external(data="hest/LUAD")
```

`timestamp` accepts a single string (used for every model), a
`{model: timestamp}` dict, or `None` (auto-picks the latest run directory per
model under `<log_path>/<data>/<model>/`).

**Persist and restore full workflow state** — everything `save_state()`
writes (constructor settings, `internal_data`/`external_data`, results,
checkpoint timestamps, `last_output_dir`, ...), so a later process can pick up
exactly where this one left off:

```python
stp.save_state("logs/my_stpred_state.yaml")

stp2 = STPred(models=["LinearProb"])
stp2.load_state("logs/my_stpred_state.yaml")
stp2.predict(data="cptac/xenium")
```

`STPred.from_run(..., state_path="logs/my_stpred_state.yaml")` combines
construction and `load_state()` in one call.

## Discovery and Config Helpers

```python
# On an instance: list configured models and inspect configs using stp.repo_root.
stp.list_data()               # data config names available under config/data/
stp.list_models()              # echoes back stp.models — what THIS instance was constructed with
stp.list_available_models()    # model config names available under config/model/ (discovery, not stp.models)
stp.describe_data("ncche/xenium")
stp.describe_model("LinearProb")

# As classmethods: pass repo_root explicitly if not running from repo root.
STPred.list_data(repo_root="/path/to/repo")
STPred.list_available_models(repo_root="/path/to/repo")
STPred.init_data_config("my_data")    # write an editable template
STPred.init_model_config("MyModel")
```

`list_models()` and `list_available_models()` are easy to mix up:
`list_models()` only echoes back whatever was passed to `STPred(models=[...])`
at construction; `list_available_models()` is the actual config-discovery view
of everything under `config/model/`.

## Adding a New Dataset

Adding a new dataset requires a **data config** and raw input data in the expected directory layout. The pipeline (`stp.preprocess`) handles patch extraction, gene set preparation, cross-validation splits, and feature embedding automatically.

### Step 1 — Prepare raw input data

Organize your raw data so each sample lives in its own subdirectory under a common `input_dir`:

```
input_dir/
├── sample_A/          # one directory per sample
│   ├── *.tiff         # whole-slide image (or .svs, .ndpi, etc.)
│   └── ...            # SpaceRanger / Xenium output files
├── sample_B/
│   └── ...
└── ...
```

The supported platform formats follow the conventions of the [HEST](https://github.com/mahmoodlab/HEST) library (Visium SpaceRanger output, Xenium output, etc.).

### Step 2 — Write the data config

Create `config/data/<namespace>/<name>.yaml`. Use `STPred.init_data_config("my_namespace/my_data")` to generate a template, then fill in your paths:

```yaml
GENERAL:
  seed: 2021
  log_path: ./logs

TRAINING:
  num_k: 5                    # number of cross-validation folds
  learning_rate: 1.0e-4
  num_epochs: 200
  monitor: PearsonCorrCoef
  mode: max
  early_stopping: {patience: 20}
  lr_scheduler: {patience: 5, factor: 0.1}

DATA:
  data_dir: /path/to/processed_data    # where preprocessed outputs are stored
  meta_dir: input/my_namespace/my_data # ids.csv and gene lists go here
  output_dir: output/pred
  gene_type: hmhvg                     # gene set type (hmhvg, hvg, heg, ...)
  num_genes: 200
  num_outputs: 200
  normalize: true
  model_name: uni_v2                   # patch encoder for feature extraction
  tech: Visium                         # ST technology (for metadata)
  train_dataloader: {batch_size: 128, num_workers: 4, pin_memory: false, shuffle: true}
  test_dataloader:  {batch_size: 1,   num_workers: 4, pin_memory: false, shuffle: false}

preprocess:
  mode: raw                            # use raw for new datasets
  platform: visium                     # visium | xenium | merfish | ...
  input_dir: /path/to/raw_data         # root of per-sample subdirectories
  output_dir: /path/to/processed_data  # same as DATA.data_dir
  meta_dir: input/my_namespace/my_data
```

### Step 3 — Run preprocessing

```python
stp = STPred(models=["LinearProb"])
stp.preprocess(data="my_namespace/my_data")
```

This runs four steps in order:

| Step | What it does | Output |
|---|---|---|
| Raw preprocessing | Extracts patches and ST expression per sample | `<data_dir>/patches/`, `<data_dir>/st/` |
| Gene set preparation | Selects highly variable / expressed genes | `<meta_dir>/<gene_type>_<num_genes>genes.json` |
| CV splits | Assigns samples to train/test folds | `<meta_dir>/ids.csv` (with `fold_*` columns) |
| Feature extraction | Runs patch encoder on all samples | `<data_dir>/emb/<feature_type>/features_<model_name>/` |

Any configured model using `feature_type: neighbor`/`all` (e.g. DeepSpot)
needs neighbor patches (`<data_dir>/patches/neighbor/`) in addition to the
target patches above. On `mode: stpbench` data downloaded from the STP-Bench
HF dataset, neighbor patches are **not** pre-extracted — the first
`preprocess()` call for such a model re-opens every raw WSI to run tissue
segmentation + neighbor tiling from scratch, which can take tens of minutes
*per gigapixel slide* with no fine-grained progress output. Budget for this
before running it against a large new cohort, and prefer a
`feature_type: global`-only model (e.g. LinearProb) for a first smoke test of a
new dataset.

**Budget disk space too, not just time.** Neighbor patches are stored as
uncompressed per-spot image grids (e.g. 1120×1120×3 for `num_n: 25`), which
is far larger than it sounds: on a 19-sample Xenium cohort in-house testing
measured `patches/neighbor/` reaching **~230 GB** (peaking higher mid-run,
before per-sample images are dropped post-feature-extraction) against
**~9 GB** of original WSIs for the same samples — roughly a 25× multiplier.
First-run wall time for that same 19-sample cohort was ~2h40m for raw
neighbor-patch extraction plus ~3h30m for neighbor feature extraction (global
feature extraction alone, for comparison, was ~8 minutes). Scale both numbers
by sample count and slide resolution before committing a `feature_type:
neighbor`/`all` model to a large new cohort — this is on top of, not
instead of, the per-gigapixel time budget above.

After preprocessing, the directory layout looks like:

```
data_dir/
├── patches/
│   └── <sample_id>.h5
├── st/
│   └── <sample_id>.h5ad
└── emb/
    └── global/
        └── features_uni_v2/
            └── <sample_id>.h5

meta_dir/
├── ids.csv                       # sample_id, fold_0, fold_1, ...
└── hmhvg_200genes.json           # selected gene list
```

### Using benchmark data (stpbench mode)

Datasets already included in STPBench (downloaded from HuggingFace) use `mode: stpbench`.
This mode reads pre-extracted patches and ST expression from the STPBench data directory
instead of generating them from raw WSIs. All existing configs under `config/data/` use this mode:

```yaml
preprocess:
  mode: stpbench               # for datasets already in STPBench
  platform: visium
  input_dir: /path/to/stp_bench   # root of the downloaded HF dataset (contains patches/, st/, wsis/)
  output_dir: /path/to/stp_bench
  meta_dir: input/hest/my_cohort
```

**`mode: stpbench` requires `<meta_dir>/ids.csv` to already exist** (a CSV
with at minimum a `sample_id` column listing the samples to process) —
unlike `mode: raw`, which generates `ids.csv` itself by scanning
`input_dir`, `stpbench` mode never scans the download directory for you and
raises `FileNotFoundError` with the exact expected path if it's missing.
Create it by hand (or from whatever manifest your download came with)
*before* calling `preprocess()`; every dataset already under
`config/data/` in this repo has its `ids.csv` committed under
`input/<namespace>/<name>/ids.csv` for exactly this reason.

For **new datasets not yet in STPBench**, use `mode: raw` and point `input_dir` at your raw data (per-sample SpaceRanger / Xenium output directories).

**`raw` vs `stpbench` is a one-time choice, not a pipeline stage.** It only
describes what format your *raw* data is in for the initial ingestion step —
it is not something you switch after preprocessing, and there is no "run raw
first, then switch to stpbench" workflow for the same dataset. Both modes
write to the exact same output layout (`data_dir/patches/`, `data_dir/st/`,
`meta_dir/ids.csv`); every step after that first `stp.preprocess()` call
(gene set prep, CV splits, feature extraction, training, evaluation, and any
later re-run of `preprocess()` itself) reads only that resulting layout and
never looks at `preprocess.mode` again. Once your own raw data has been
ingested with `mode: raw`, leave it set to `raw` — there's nothing to change
afterward.

One more `preprocess.mode` value exists but isn't meant to be hand-written in
a data config: `inference` (no ST/expression data — this is what a WSI-path
`predict()` uses internally). It has two sub-cases controlled by
`preprocess.extract_from_wsi`: `False` (the default) reuses patches that
already exist under an already-preprocessed asset directory, while `True`
extracts patches fresh from a bare WSI with no companion ST data at all —
tissue segmentation and tiling for a single slide file or directory of
slides. See
[Easy Inference Directly on a WSI](#easy-inference-directly-on-a-wsi) for the
`predict()`-level interface to both; you won't normally write either by
hand.

### Dry-run check before preprocessing

Use `stp.check()` to verify config and artifact paths before committing to a long run:

```python
stp.check(data="my_namespace/my_data", strict=False)
```

## Adding a New Model

Adding a new model requires three files: a **model class**, an **`__init__.py`**, and a **model config**.

### Step 1 — Write the model class

Create `src/model/<module_name>/<module_name>.py`:

```python
# src/model/my_model/my_model.py
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, num_genes: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1536, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_genes),
        )

    def forward(self, img_emb, label=None, **kwargs):
        output = F.softplus(self.net(img_emb))
        if label is not None:
            return {"loss": F.mse_loss(output, label), "logits": output}
        return {"logits": output}
```

Two rules to follow:
- `__init__` parameter names are automatically wired from `MODEL.*` config fields. `num_genes` is always injected from the data config.
- `forward` must return `{"loss": ..., "logits": ...}` during training and `{"logits": ...}` during inference.

The inputs to `forward` depend on which features the dataset provides:

| `feature_type` | Key passed to `forward` |
|---|---|
| `global` / `neighbor` / `target` | `img_emb` — pre-extracted patch embeddings |
| `none` + `use_emb: false` | `img` — raw patch images |
| `all` | `img_emb` — all features concatenated |

### Step 2 — Add `__init__.py`

```python
# src/model/my_model/__init__.py
from .my_model import MyModel
```

### Step 3 — Write the model config

Create `config/model/MyModel.yaml`:

```yaml
MODEL:
  model_name: my_model.MyModel   # <module_name>.<ClassName>
  hidden_dim: 512                # any __init__ parameter can be set here
  adapter: default

DATA:
  dataset_name: STDataset
  feature_type: global           # which patch embeddings to load
  model_name: uni_v2             # patch encoder — keep at the benchmark default
```

`DATA.model_name` picks the **patch encoder**, a choice kept separate and
swappable from `MyModel`'s own architecture (`MODEL.*`). Default it to
`uni_v2` — every model that consumes pre-extracted embeddings is benchmarked
against that same encoder, which is what makes a `PearsonCorrCoef`
difference between models a statement about architecture rather than about
which encoder happened to produce better features. Only change it if you
have a specific reason to and are aware you're leaving that shared
comparison basis. Models that instead bring their own internal image
encoder (`feature_type: none` with a custom backbone in the model class, or
a zero-shot foundation model with fixed pretrained weights) aren't on this
axis at all — say so explicitly in a config comment, so results don't get
silently read as "architecture X beats architecture Y" when the real
difference is the encoder.

That's it. `MyModel` will appear in `STPred.list_available_models()` — the
config-discovery view, not `list_models()`, which only echoes back whatever
was passed to `STPred(models=[...])` — and is ready to use:

```python
stp = STPred(models=["MyModel"])
stp.check(data="ncche/xenium")
stp.preprocess(data="ncche/xenium")
stp.train(data="ncche/xenium")
```

### Common config options

| Scenario | Config setting |
|---|---|
| Use pre-extracted patch embeddings | `feature_type: global` (or `neighbor`, `target`, `all`) |
| Use raw patch images directly | `feature_type: none`, `use_emb: false` |
| Slide-level batching | `load_level: slide`, `train_dataloader: {batch_size: 1}` |
| Need extra preprocessing | Add `extra_preprocess:` entries (see below) |

### Adapters

An adapter controls how the training loop interacts with your model — how the batch is prepared, how the loss is computed, and how predictions are collected. Most models can use `adapter: default`.

If your model has a non-standard training loop (e.g. contrastive learning, graph-based batching, or custom prediction aggregation), you can either pick one of the built-in adapters or implement your own.

**Built-in adapters:**

| Name | When to use |
|---|---|
| `default` | Standard patch-level regression (most models) |
| `deepspot` | DeepSpot: chunked spot/sub_spot/neighbor batch contract |
| `egn` | EGN / EGGN: exemplar-guided neighborhood models |
| `graph` | Graph-based models (SGN) |
| `contrastive` | Contrastive learning objectives |
| `triplex` | TRIPLEX multi-branch architecture |
| `sepal` | SEPAL two-stage pipeline |
| `stem` | STEM architecture |

**Custom adapter:** If none of the above fit, subclass `ModelAdapter` and register it:

```python
# src/model/my_model/adapter.py
from core.model_adapters import register_adapter
from core.model_adapters.base import ModelAdapter

class MyAdapter(ModelAdapter):
    def prepare_batch(self, module, batch, stage):
        # customize how the batch is unpacked
        ...

    def forward(self, module, batch, phase):
        # customize the forward pass and loss computation
        ...

register_adapter("my_adapter", MyAdapter)
```

Import the adapter in your model's `__init__.py` so it registers at load time:

```python
# src/model/my_model/__init__.py
from .my_model import MyModel
from . import adapter  # registers MyAdapter
```

Then set it in the model config:

```yaml
MODEL:
  model_name: my_model.MyModel
  adapter: my_adapter
```

### Extra preprocessing

If your model needs preprocessing before training (e.g. graph construction, similarity matrices), **STP-Bench does not write this script for you** — you (or the model's reference implementation) must supply it, the same way you supply the model code itself. Place it in `src/model/<module_name>/preprocess/` and register it in the config:

```yaml
MODEL:
  model_name: my_model.MyModel
  extra_preprocess:
  - script: build_graph.py      # resolves to src/model/my_model/preprocess/build_graph.py
    args:
      num_neighbors: 6
      model_name: ~             # ~ is auto-filled from the data config's model_name
  adapter: default
```

The script must accept at least `--data_dir`. Supporting `--overwrite` lets the pipeline skip already-computed outputs on re-runs.

The following args are auto-filled from the data config's `preprocess` section when set to `~`:

| Arg | Source |
|---|---|
| `model_name` | `DATA.model_name` |
| `gene_type` | `DATA.gene_type` |
| `num_genes` | `DATA.num_genes` |
| `input_dir` | `preprocess.input_dir` |
| `platform` | `preprocess.platform` |
| `mode` | `preprocess.mode` |

**Model-level preprocess flags** — a model config can also define a `preprocess:` section to set pipeline-level flags. Boolean flags are OR'd across all active models (if any model requires the flag, the pipeline enables it):

```yaml
MODEL:
  model_name: my_model.MyModel
  preprocess:
    save_neighbor_imgs: true   # merged into the pipeline preprocess config
  extra_preprocess:
    ...
```

## Claude Code Skills

This repository ships [Claude Code](https://claude.com/claude-code) skills
under `.claude/skills/` that encode the two extension guides above
(**[Adding a New Dataset](#adding-a-new-dataset)**,
**[Adding a New Model](#adding-a-new-model)**) as agent-actionable
checklists, grounded in the repository's actual internals rather than a
generic description — adapter registry lookups, `dataset_name` resolution,
and known failure modes around external evaluation (gene-panel mismatches,
reference-bank corruption, unforwarded `genes_override`/`ref_data_dir`
kwargs) that this repo's history has repeatedly hit.

| Skill | Triggers on |
|---|---|
| `add-dataset` | "Add a new dataset", onboarding raw slides/ST data, setting up an internal/external evaluation cohort |
| `add-model` | "Add a new model", integrating a published ST-prediction method, wiring a model class into `STPred`'s benchmark loop |

Each skill starts by asking for the model source or raw data location — see
[Extending STP-Bench](../README.md#extending-stp-bench) in the README for what
to have ready (a local path or git URL for model code, a local path or
download URL/accession for raw data). Claude Code discovers both skills
automatically; simply ask it to add a new model or dataset and it follows the
corresponding skill's checklist, including the verification steps at the end
(`stp.check()`, `stp.preprocess()`, `stp.train()`, `stp.evaluate_internal()`,
`stp.evaluate_external()`, and — for models using the default `feature_type`
mechanism — a sanity-check `stp.predict()` call directly on a WSI file).

This mechanism is specific to Claude Code and is not read by other coding
agents or by `STPred` itself — the skills are a development-time aid, not
part of the runtime API.
