---
name: add-model
description: Integrate a new prediction model into STP-Bench (model class, config, optional custom adapter/dataset/preprocessing). Use when the user asks to add a new model, integrate a published ST-prediction method, or wire up a model class for STPred's benchmark loop ("새 모델 추가", "add <ModelName>").
---

# Adding a New Model to STP-Bench

Full narrative reference: `docs/guide.md#adding-a-new-model`. This skill is the
actionable checklist — follow it in order, and verify each claim against the
actual repo state rather than assuming it hasn't drifted.

## Step 0 — Get the model source from the user

STP-Bench doesn't ship model implementations for you — the user must provide
their own. **Don't invent the architecture from a paper description alone.**
Before writing any code, ask the user for one of, in order of preference:

1. **A local filesystem path** to an existing implementation (their own code,
   or a cloned reference repo) — the simplest option when working on the same
   machine/cluster the data and this repo live on. Read it directly with the
   available file tools.
2. **A git/GitHub URL** to clone (`git clone <url> /tmp/<name>-ref` or similar)
   when the reference implementation is public and not already local.
3. **Pretrained weights location** (local path, HuggingFace repo id, or
   download URL) if the model is zero-shot/pretrained rather than trained
   from scratch in this repo — see `src/model/deepspotm/` or
   `src/model/omiclip/` for the vendored/HF-download pattern to follow.

Only fall back to writing the model from a plain description if the user
explicitly confirms they have no existing code and wants a minimal
from-scratch model — flag that this is a simplified re-implementation, not a
faithful reproduction of the published method.

**If the model needs preprocessing before training** (graph construction,
similarity matrices, custom patch sampling — anything beyond the standard
patch/embedding pipeline), that preprocessing code is also something the
user must supply, not something to write from scratch. It's usually already
part of the reference implementation gathered above — ask for it explicitly
if it isn't obviously included, before starting on the "Extra preprocessing"
step below.

## Minimum required files

1. `src/model/<module_name>/<module_name>.py` — the model class (plain `nn.Module`)
2. `src/model/<module_name>/__init__.py` — `from .<module_name> import <ClassName>`
3. `config/model/<ClassName>.yaml` — model config

## Step 1 — Write the model class

- `__init__` parameter names are wired automatically from `MODEL.*` config
  fields via `instancialize()` (`src/core/base_module.py`) — every constructor
  kwarg you want configurable must have a matching `MODEL.<key>` entry in the
  YAML, or a default value.
- `num_genes` is always injected from the data config — accept it even if the
  model doesn't use it directly.
- `forward(self, ..., label=None, **kwargs)` must return
  `{"loss": ..., "logits": ...}` when `label` is not `None` (train/val), and
  `{"logits": ...}` otherwise (test/predict). Always accept `**kwargs` — the
  adapter passes extra batch keys (`phase`, `device`, `dataset`, ...) that a
  plain model can just ignore.
- The input feature kwarg depends on `DATA.feature_type` in the model config:

  | `feature_type` | kwarg passed to `forward` |
  |---|---|
  | `global` / `neighbor` / `target` | `img_emb` — pre-extracted patch embeddings |
  | `none` + `use_emb: false` | `img` — raw patch images |
  | `all` | `img_emb` — all features concatenated |

## Step 2 — `__init__.py`

```python
from .<module_name> import <ClassName>
```

`MODEL.model_name: <module_name>.<ClassName>` resolves via
`importlib.import_module(f"model.{module_name}")` then
`getattr(module, class_name)` (`src/core/base_module.py::load_model`) — the
class must be importable at the package's top level, exactly matching this
dotted string. No separate model registry file to touch.

## Step 3 — Model config

`config/model/<ClassName>.yaml`:

```yaml
MODEL:
  model_name: <module_name>.<ClassName>
  <any other __init__ kwarg>: <value>
  adapter: default   # see Adapters below
DATA:
  dataset_name: STDataset   # or a custom dataset class, see below
  feature_type: global
```

Verify: `stp.list_available_models()` should show `<ClassName>` immediately —
adding the config is what makes it discoverable. (`stp.list_models()` is a
different method — it only echoes back whatever was passed to
`STPred(models=[...])`, not what's discoverable on disk; don't confuse the
two when verifying this step.)

## Adapters

An adapter controls batch prep / loss computation / prediction aggregation.
Built-ins (`src/core/model_adapters/registry.py`): `default`, `deepspot`,
`egn`, `graph`, `contrastive`, `triplex`, `sepal`, `stem`. Use `default`
unless the training loop is genuinely non-standard (contrastive objectives,
graph batching, multi-stage pipelines, or DeepSpot's chunked slide-level
batch contract).

Custom adapter — subclass `ModelAdapter`, register with
`register_adapter(name, cls)`, and **import the module from the model's
`__init__.py`** so registration actually runs at load time:

```python
# src/model/<module_name>/__init__.py
from .<module_name> import <ClassName>
from . import adapter   # registers the adapter as a side effect of import
```

Skipping this import is a silent failure mode — `MODEL.adapter: my_adapter`
will raise `Unknown model adapter` (or, if a class fallback happens to match,
silently resolve to the wrong adapter) rather than something obviously wrong
at the model class itself.

## Optional: custom dataset class

If `STDataset` isn't enough (extra preprocessing, non-standard `__getitem__`),
subclass it in `src/dataset/<name>.py`. **You must also add
`from .<name> import <DatasetClass>` to `src/dataset/__init__.py`** —
`dataset_name` resolves via
`getattr(importlib.import_module('dataset'), camel_name)`
(`src/core/base_datamodule.py::load_data_module`), so a class that only lives
in its own file and isn't re-exported from the package `__init__.py` fails to
resolve. This step is easy to forget and won't surface until someone actually
runs preprocess/train against the new model.

When subclassing `STDataset`, forward **every** constructor kwarg to
`super().__init__(...)` — in particular `ref_data_dir` and `genes_override`.
This repo has repeatedly shipped external-evaluation bugs from subclasses that
accept these two kwargs but silently drop them instead of forwarding, which
breaks external-dataset evaluation (wrong gene panel, reference-bank
corruption) without raising anything — it just produces wrong numbers.

## Optional: extra preprocessing

Reminder: this script is user-supplied (see Step 0), not something to
generate from scratch — adapt the reference implementation's own
preprocessing code into this location and interface, rather than
reinventing it.

```yaml
MODEL:
  extra_preprocess:
  - script: build_graph.py   # resolves to src/model/<module_name>/preprocess/build_graph.py
    args:
      num_neighbors: 6
      model_name: ~          # ~ = auto-fill from DATA.model_name; same pattern for
                              # gene_type, num_genes, input_dir, platform, mode
  adapter: default
```

The script must accept `--data_dir` at minimum. Support `--overwrite` so
re-runs skip already-computed outputs. If the script needs to behave
differently for external evaluation (a different dataset than the one the
model was trained on), it must accept `--external_dir` /
`--external_meta_dir` / `--external_asset_dir` — see `src/model/EGN/preprocess/`
or `src/model/EGGN/preprocess/` for the established convention, and namespace
any external-run output by the *training* run's `meta_dir` (not `data_dir`,
which is often a root shared across every dataset in this repo).

Model-level `preprocess:` blocks (distinct from `extra_preprocess:`) set
pipeline-wide boolean flags and are OR'd across every active model.

## Verify before declaring done

```python
stp = STPred(models=["<ClassName>"])
stp.check(data="ncche/xenium")        # config/artifact sanity check
stp.preprocess(data="ncche/xenium")
stp.train(data="ncche/xenium")
stp.evaluate_internal()
```

If an external dataset is in scope, also run
`stp.evaluate_external(data=..., train_data="ncche/xenium")` — this repo's
external-eval path has repeatedly caught bugs (gene-panel mismatches,
reference-bank corruption, unforwarded kwargs, stale/incompletely-written
caches) that internal-only testing never exercises. Don't consider a new
model done until it has been run through both.

If the model uses the default `feature_type` mechanism (no `extra_preprocess`),
also sanity-check `stp.predict()` directly on a WSI file — see
[docs/guide.md — Usage Patterns](../../../docs/guide.md#usage-patterns) for
the call shape. Models with `extra_preprocess` (graph/reference-bank builders)
aren't wired to WSI-only assets yet and only support the named-config predict
path.
