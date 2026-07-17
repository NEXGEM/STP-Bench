# STP-Bench: A Unified Systematic Benchmark for Virtual Spatial Transcriptomics from Histopathology Images

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

STP-Bench is a benchmark suite for **virtual spatial transcriptomics** — predicting
spot-level gene expression directly from H&E histopathology images. It provides a
single, unified `STPred` API to preprocess data, train, and evaluate a growing
collection of published models under matched **internal cross-validation** and
**external-dataset** evaluation protocols, so results stay directly comparable
across models and datasets.

## Updates

- **2026-07-17** — Added **DeepSpotM** as a new zero-shot pretrained model.
- **2026-05-28** — Initial release.

## Installation

```bash
git clone https://github.com/NEXGEM/STP-Bench.git
cd STP-Bench
bash scripts/create_env.sh
source .stpbench/bin/activate
```

<details>
<summary><strong>Installation details</strong> (CUDA extras, manual setup, compatibility)</summary>

The setup script creates a Python 3.11 virtual environment and installs:

- editable `stp_bench`
- `torch==2.3.1+cu118`
- `torchvision==0.18.1+cu118`
- `torchaudio==2.3.1+cu118`
- runtime benchmark dependencies
- preprocessing dependencies
- `flash-attn==2.5.9.post1` *(optional — only needed for models that use Flash Attention)*

To skip `flash-attn`:

```bash
SKIP_FLASH_ATTN=1 bash scripts/create_env.sh .stpbench
```

If you do install `flash-attn`, use the pinned PyTorch/CUDA stack above to
ensure a compatible prebuilt wheel is available.

#### Optional CUDA Extras

CUDA dataframe extras are not required for the core benchmark API. Install them
only on machines where RAPIDS CUDA 12 packages are supported:

```bash
INSTALL_CUDA_EXTRAS=1 bash scripts/create_env.sh .stpbench
```

#### Manual Setup

Use this only if you do not want the setup script:

```bash
uv venv --python 3.11 .stpbench
source .stpbench/bin/activate

python -m pip install --upgrade pip setuptools wheel packaging ninja
uv pip install -r requirements/torch-cu118.txt
uv pip install -e .
uv pip install -r requirements/runtime.txt
uv pip install -r requirements/preprocess.txt

# Optional: only needed for models that use Flash Attention
uv pip install -r requirements/flash-attn.txt --no-build-isolation
```

`requirements/all.txt` contains the pinned torch stack plus runtime and
preprocessing dependencies:

```bash
uv pip install -e .
uv pip install -r requirements/all.txt

# Optional: only needed for models that use Flash Attention
uv pip install -r requirements/flash-attn.txt --no-build-isolation
```

#### Compatibility Contract

The supported default environment is:

- Linux x86_64
- Python 3.11
- Ubuntu 20.04 / `glibc 2.31` or newer
- PyTorch `2.3.1+cu118`
- TorchVision `0.18.1+cu118`
- FlashAttention `2.5.9.post1` *(optional)*

Check for accidental mismatch:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "import flash_attn; print(flash_attn.__version__)"  # optional
```

If installing `flash-attn`, avoid mismatched environments such as:

- `torch==2.10.0+cu128` with a CUDA 11.7 local toolkit.
- `flash-attn==2.8.3` on Ubuntu 20.04 / `glibc 2.31`.

Those combinations have no reliable local `flash-attn` install path in this
project. Use the pinned requirements above instead.

</details>

## Benchmark Data

Preprocessed benchmark data (patches, ST expression, embeddings, metadata) is
hosted on Hugging Face at [`nexgem/STP-Bench`](https://huggingface.co/datasets/nexgem/STP-Bench).

```python
from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="nexgem/STP-Bench",
    repo_type="dataset",
    local_dir="/path/to/stp_bench",
)
```

Set the downloaded directory as `DATA.data_dir` (and `preprocess.output_dir`) in
your data config.

<details>
<summary><strong>Data download and layout details</strong> (CLI download, directory structure)</summary>

#### Download via CLI

```bash
huggingface-cli download nexgem/STP-Bench \
    --repo-type dataset \
    --local-dir /path/to/stp_bench
```

#### Use as `data_dir`

```yaml
DATA:
  data_dir: /path/to/stp_bench   # root of the downloaded HF dataset

preprocess:
  input_dir: /path/to/stp_bench
  output_dir: /path/to/stp_bench
```

The expected directory layout after download:

```
stp_bench/
├── patches/          # patch .h5 files per sample
├── st/               # aligned expression .h5ad files per sample
├── emb/              # pre-extracted patch embeddings (if included)
├── metadata/         # per-sample JSON metadata (HEST format)
└── wsis/             # whole-slide images (if included)
```

</details>

## Quick Start

> When working directly from the repository without installing the package, use
> `from api import STPred` with `src/` on `PYTHONPATH`.

```python
from stpbench import STPred

stp = STPred(
    models=["StNet"],
    gpu=1,
    repo_root="/path/to/repo",  # path where config files exist
)

stp.check(data="ncche/xenium", strict=True)          # validate before a heavy run
result = stp.benchmark(internal_data="ncche/xenium", external_data="hest/LUAD")

print(result.summary())
result.save("benchmark_results.csv")
```

Config names map exactly to YAML files (`config/data/ncche/xenium.yaml`,
`config/model/StNet.yaml`, ...). If a file is missing, `STPred` raises an error
with the exact path to create.

For step-by-step workflows (running each stage separately, resuming a run,
persisting/restoring state, discovery helpers), see
**[docs/guide.md — Usage Patterns](docs/guide.md#usage-patterns)**.

## Python API

Primary workflow methods:

- `stp.benchmark(internal_data, external_data=None)`: preprocess, train, internal evaluate, and optionally external predict/evaluate.
- `stp.preprocess(data, dry_run=False)`: prepare one dataset for all selected models.
- `stp.train(data=None)`: train all selected models.
- `stp.evaluate_internal(data=None)`: evaluate internal test folds.
- `stp.evaluate_external(data, train_data=None)`: evaluate external labeled data using an internal training run. If the external dataset is missing base artifacts (patches/embeddings), `preprocess()` is triggered automatically before evaluation.
- `stp.predict(data, train_data=None)`: predict on slide-image-only external data.
- `stp.check(data, strict=False)`: validate configs and expected artifacts before a heavy run.

Benchmark logging is enabled by default and uses a consistent `[STPBench]` line
format with elapsed seconds for each major step. Pass `verbose=False` to silence
console logs, or `log_file="logs/run.jsonl"` to persist structured JSONL events.
Weights & Biases is opt-in: pass `wandb=True` to `STPred(...)` when online W&B
tracking is desired. Use `wandb_project="..."` to choose the W&B project name.

Result objects are `BenchmarkResult` instances. They are dict-compatible and
provide:

- `result.summary()` — per-model results including cross-fold aggregate stats (`mean`, `std`, `per_fold`) for each metric
- `result.to_records()` — flat list of per-fold dicts
- `result.to_dataframe()` — pandas DataFrame of records
- `result.best_checkpoints()` — best checkpoint path per model/fold
- `result.prediction_dirs()` — prediction output directories
- `result.save("results.csv")` — write records to CSV

## Configuration

`STPred` uses exact-name YAML files under `config/data/` and `config/model/`.
A data config must define at minimum:

```yaml
GENERAL:
  seed: 2021
  log_path: ./logs

TRAINING:
  num_k: 5
  learning_rate: 1.0e-4
  num_epochs: 200
  monitor: PearsonCorrCoef
  mode: max
  early_stopping: {patience: 20}
  lr_scheduler: {patience: 5, factor: 0.1}

DATA:
  data_dir: /path/to/processed_data
  dataset_name: STDataset
  gene_type: hmhvg
  num_genes: 200
  num_outputs: 200
  model_name: uni_v2          # patch encoder
  train_dataloader: {batch_size: 128, num_workers: 4, pin_memory: false, shuffle: true}
  test_dataloader:  {batch_size: 1,   num_workers: 4, pin_memory: false, shuffle: false}

preprocess:
  mode: raw                   # raw | stpbench
  input_dir: /path/to/raw_data
  output_dir: /path/to/processed_data
```

Use `STPred.init_data_config("my_data")` to generate an editable template.

**`repo_root` parameter** — all relative paths in configs (`meta_dir`, `log_path`, `output_dir`) are resolved relative to `repo_root` (defaults to `.`). Pass it explicitly when running from a directory other than the repo root:

```python
stp = STPred(models=["StNet"], gpu=1, repo_root="/path/to/stp_bench")
```

## Outputs

Default locations (can be changed in the data config YAML):

- Logs: `<GENERAL.log_path>/<data>/<model>/<timestamp>/`
- Checkpoints: `<GENERAL.log_path>/<data>/<model>/<timestamp>/fold<k>/`
- Predictions (eval): `<DATA.output_dir>/<data>/<model>/fold<k>/`
- Predictions (inference): `<DATA.output_dir>/<data>/<model>/<train_data>/fold<k>/`

## Extending STP-Bench

STP-Bench doesn't ship model implementations or raw datasets — you bring your
own. The easiest way to provide either is to point at wherever it already
lives:

- **Model code**: a local path to an existing implementation, or a git/
  GitHub URL to clone. A reference implementation (or its pretrained
  weights) lets the integration match the real input/output shapes instead
  of guessing from a paper description.
- **Raw data**: a local path to the per-sample WSI/ST directories (most
  datasets are already sitting on the same machine or shared storage —
  too large to move casually), or a download URL/accession (HuggingFace
  dataset repo, GEO/SRA/Zenodo, cloud bucket) if it isn't local yet.
- **Extra preprocessing** (only if the model needs it — graph construction,
  similarity matrices, custom patch sampling, etc.): STP-Bench does not
  write this for you either. Bring the preprocessing logic along with the
  model code (it's usually already part of the reference implementation)
  so it can be adapted into `src/model/<module_name>/preprocess/`.

With that in hand:

- **Adding a new dataset** — see **[docs/guide.md — Adding a New Dataset](docs/guide.md#adding-a-new-dataset)**.
- **Adding a new model** — see **[docs/guide.md — Adding a New Model](docs/guide.md#adding-a-new-model)**.

#### For Claude Code

This repository ships two [Claude Code](https://claude.com/claude-code) skills
under `.claude/skills/` — `add-model` and `add-dataset` — that encode the
procedures above as agent-actionable checklists grounded in the repository's
actual internals (adapter registry, `dataset_name` resolution, and known
failure modes around external evaluation). Each skill starts by asking for
the model source or raw data location described above. Claude Code discovers
them automatically; simply ask it to add a new model or dataset and it will
follow the corresponding skill. This mechanism is specific to Claude Code and
is not read by other coding agents.

## License

Released under [CC BY-NC-SA 4.0](LICENSE.md) — non-commercial use with attribution, and derivatives must be shared under the same license.
