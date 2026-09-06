---
name: add-model
description: Integrate a new prediction model into STP-Bench (model class, config, optional custom adapter/dataset/preprocessing). Use when the user asks to add a new model, integrate a published ST-prediction method, or wire up a model class for STPred's benchmark loop ("add <ModelName>").
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

## Inference-time batch chunking

Train batches are small and dataloader-controlled (`DATA.train_dataloader.
batch_size`). Val/test/predict batches are not: `STDataset`'s test-phase
`__getitem__` returns one *whole slide's* spots as a single item, so the
val/test/predict `DataLoader` (batch_size from `DATA.test_dataloader`, often
left at its default) hands your model every spot of a held-out slide — easily
1000+ — in one forward call. A model that trains fine at `batch_size: 16`
can OOM the instant it hits its first real validation pass, because nothing
exercises that path until an actual fold trains far enough to validate. This
bit `AsymST` in practice: DenseNet121+UNI2-h forward on ~1270 spots at once
during validation OOM'd in `batch_norm`, even though training itself (batch
16) had been running cleanly for a full epoch.

The fix, already used by ~18 models in this repo (`grep -rl max_batch_size
src/model/` before inventing your own variant) — accept a `max_batch_size`
constructor kwarg (default e.g. `1024`) and split any inference-phase batch
bigger than it before the expensive part of `forward`, concatenating results
after. Minimal version (`src/model/st_net/st_net.py`):

```python
def forward(self, img, label=None, **kwargs):
    phase = kwargs.get('phase', 'train')
    if phase == 'train':
        output = self.model(img)
    elif img.shape[0] > self.max_batch_size:
        output = torch.cat([self.model(c) for c in img.split(self.max_batch_size, dim=0)], dim=0)
    else:
        output = self.model(img)
    ...
```

Gate on `phase != 'train'` (the `phase` kwarg the adapter always passes — see
`src/model/EGN/EGN.py`, `src/model/TRIPLEX/TRIPLEX.py`) or equivalently
`not self.training`; train batches are already small, so there's nothing to
chunk there, and if `forward` computes a multi-term training loss you cannot
split it mid-computation without risking a different backward graph. If the
model has multiple auxiliary heads/losses like `AsymST`, chunk only the
shared per-chunk predictions and compute the loss once over the concatenated
result, so a chunked pass produces bit-for-bit the same output as an
unchunked one — verify this with a unit test (small fake batch, compare
`max_batch_size=len(batch)` vs. a small value) before trusting a real run.

**Exception — don't chunk if the model is genuinely slide-level.** Some
architectures need every spot of a slide together in one forward pass because
spots attend to or message-pass with each other — chunking would silently
change predictions, not just save memory. `st_flow.StFlow`'s denoiser is the
concrete example already in this repo: `sample()` asserts
`img_features.shape[0] == 1` and treats the whole slide's spot sequence as
one jointly-attended unit (`max_batch_size` is declared in its config but
never actually used in `forward` — it's vestigial, not a working safeguard).
GNN-based models (EGGN, SGN) are the same in spirit: message passing needs
the whole graph. If you're integrating a model like this, say so explicitly
in a comment near `forward` (so a future editor doesn't "fix" the missing
chunking) and pair it with the matching datamodule policy below.

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
  model_name: uni_v2        # patch encoder — see Patch encoder vs. architecture below
```

**Patch encoder vs. architecture**: for any model using
`feature_type: global/neighbor/target/all`, `DATA.model_name` picks the
**patch encoder** — a choice deliberately kept separate and swappable from
the model's own architecture (`MODEL.*`). Default it to `uni_v2` (the
benchmark's standard encoder) unless you have a specific reason not to:
every such model is benchmarked against that same encoder, which is what
makes a `PearsonCorrCoef` difference between models a statement about
*architecture*, not about which encoder happened to extract better
features. Changing it for only some models silently breaks that shared
comparison basis. Models that instead bring their own internal image
encoder (`feature_type: none` with a custom CNN/ViT backbone in the model
class itself — see `src/model/AsymST/` — or a zero-shot foundation model
with fixed pretrained weights, e.g. DeepSpotM) aren't on this comparison
axis at all — say so explicitly in the config's comments, so results don't
get silently read as "architecture X beats architecture Y" when the real
difference is the encoder.

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

## Datamodule policies

A different knob from the adapter: `src/core/datamodule_policies.py` decides
*how batches are constructed and iterated* (the adapter decides what happens
to a batch once built). Set via `DATA.datamodule_policy: <name>` in the model
config; unset falls back to `default`.

- **`default`** — a plain `torch.utils.data.DataLoader` over your dataset,
  sized by `DATA.train_dataloader`/`DATA.test_dataloader`. What you get
  unless you ask for something else.
- **`graph`** — the same shape but backed by PyTorch Geometric's `DataLoader`,
  for datasets that yield PyG `Data` objects (see `config/model/EGGN.yaml`,
  `config/model/SGN.yaml`).
- **`direct`** — val/test/predict skip the `DataLoader` entirely and iterate
  the dataset object itself, so the dataset's own `__getitem__`/iteration
  defines what "one step" means (see `config/model/Sepal.yaml`).

`graph`/`direct` models are exactly the "genuinely slide-level" models from
the chunking section above — the two choices tend to travel together, since
both stem from the same fact: the model needs a whole slide's structure
(graph edges, or dataset-defined iteration) as one unit, not an arbitrary
per-spot batch a generic `DataLoader` would produce.

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

Calling `stp.train()` and having it start is not verification — it only
proves the train-phase path works, which is the smaller, dataloader-batched
half of the model contract. Let it actually run until the first validation
pass completes (watch the fold's `metrics.csv` for a populated
`val_<metric>`/`val_target` row, or tail the run's log for the validation
progress bar reaching 100%), not just a handful of training steps. This is
the only way to exercise the val/test-phase code path — whole-slide batches,
`label=None` inference branches, `max_batch_size` chunking if you added it —
none of which a short train-only smoke test touches. The `AsymST` OOM bug
this section's chunking guidance is based on only surfaced this way: training
(batch 16) ran cleanly for a full epoch, then the first validation pass
(~1270 spots in one forward call) crashed immediately. A smoke test that
declared success after a few training steps would have missed it entirely.

If an external dataset is in scope, also run
`stp.evaluate_external(data=..., train_data="ncche/xenium")` — this repo's
external-eval path has repeatedly caught bugs (gene-panel mismatches,
reference-bank corruption, unforwarded kwargs, stale/incompletely-written
caches) that internal-only testing never exercises. Don't consider a new
model done until it has been run through both.

If the model uses the default `feature_type` mechanism (no `extra_preprocess`),
also sanity-check `stp.predict()` directly on a WSI file — see
[docs/guide.md — Easy Inference Directly on a WSI](../../../docs/guide.md#easy-inference-directly-on-a-wsi)
for the call shape. Models with `extra_preprocess` (graph/reference-bank builders)
aren't wired to WSI-only assets yet and only support the named-config predict
path.
