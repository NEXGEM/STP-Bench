# STP-Bench Guide

Step-by-step guides for the workflows that don't fit in the [README](../README.md)'s
quick start: running the pipeline stage by stage, adding a new dataset, and
adding a new model.

## Contents

- [Usage Patterns](#usage-patterns)
- [Adding a New Dataset](#adding-a-new-dataset)
- [Adding a New Model](#adding-a-new-model)

## Usage Patterns

```python
# List available data configs and the models configured on this STPred instance.
print(stp.list_data())
print(stp.list_models())
```

**Step by step, instead of one-shot `benchmark()`:**

```python
stp.preprocess(data="ncche/xenium")
stp.train(data="ncche/xenium")
int_result = stp.evaluate_internal()
ext_result = stp.evaluate_external(data="hest/LUAD")

# Results are dict-compatible and provide convenience helpers.
print(ext_result.summary())
print(ext_result.best_checkpoints())
ext_result.save("benchmark_results.csv")
```

**Inference on unlabeled external data:**

```python
pred_result = stp.predict(data="cptac/xenium")
```

**Easy inference directly on a WSI (no data config to write):**

`data` also accepts a filesystem path instead of a named config — STP-Bench
extracts patches and features for you. Non-config targets require
`output_dir` (where extracted patches/embeddings/predictions are written)
and a resolvable checkpoint (`ckpt_path`, or `train_data`/a prior
`train()`/`from_run()` call):

```python
# A single slide file
stp.predict(
    data="/path/to/slide.svs",
    output_dir="/path/to/output",
    train_data="ncche/xenium",   # or ckpt_path="/path/to/checkpoint.ckpt"
)

# A directory of slides -- one prediction per slide, same output_dir
stp.predict(data="/path/to/slides_dir/", output_dir="/path/to/output", train_data="ncche/xenium")

# Already-preprocessed assets (a directory containing patches/*.h5) --
# output_dir here is just where predictions are written, not re-extracted into
stp.predict(data="/path/to/existing_assets", output_dir="/path/to/predictions", train_data="ncche/xenium")

# Restrict output to specific genes; names not in the training panel are
# dropped with a warning rather than failing the whole run
stp.predict(
    data="/path/to/slide.svs", output_dir="/path/to/output", train_data="ncche/xenium",
    gene_list=["GENE1", "GENE2"],
)

# Per-call model override -- doesn't mutate stp.list_models()
stp.predict(data="/path/to/slide.svs", output_dir="/path/to/output", train_data="ncche/xenium", models=["TRIPLEX"])
```

The output `.h5ad` includes `obsm['spatial']` patch coordinates. This path
works for models using the default `feature_type` mechanism (StNet, TRIPLEX,
DeepSpot, HisToGene, DeepSpotM, ...); models with `extra_preprocess`
(EGN, EGGN, Sepal, OmiCLIP, M2ORT, M2OST) and SGN/Stem still require the
named-config path above.

**Resume a known training run** without keeping the original Python process alive:

```python
stp = STPred.from_run(
    data="ncche/xenium",
    models=["StNet"],
    timestamp="2026-05-18-12-00-00",
)
stp.evaluate_external(data="hest/LUAD")
```

**Persist and restore workflow state:**

```python
stp.save_state("logs/my_stpred_state.yaml")

stp2 = STPred(models=["StNet"])
stp2.load_state("logs/my_stpred_state.yaml")
stp2.predict(data="cptac/xenium")
```

**Discovery and config helpers:**

```python
# On an instance: list configured models and inspect configs using stp.repo_root.
stp.list_data()
stp.list_models()
stp.list_available_models()
stp.describe_data("ncche/xenium")
stp.describe_model("StNet")

# As classmethods: pass repo_root explicitly if not running from repo root.
STPred.list_data(repo_root="/path/to/repo")
STPred.init_data_config("my_data")    # write an editable template
STPred.init_model_config("MyModel")
```

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
stp = STPred(models=["StNet"])
stp.preprocess(data="my_namespace/my_data")
```

This runs four steps in order:

| Step | What it does | Output |
|---|---|---|
| Raw preprocessing | Extracts patches and ST expression per sample | `<data_dir>/patches/`, `<data_dir>/st/` |
| Gene set preparation | Selects highly variable / expressed genes | `<meta_dir>/<gene_type>_<num_genes>genes.json` |
| CV splits | Assigns samples to train/test folds | `<meta_dir>/ids.csv` (with `fold_*` columns) |
| Feature extraction | Runs patch encoder on all samples | `<data_dir>/emb/<feature_type>/features_<model_name>/` |

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
```

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
