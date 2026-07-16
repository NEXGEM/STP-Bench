# DeepSpotM

## Checkpoint Setup

DeepSpotM is a **gated** HuggingFace repo — request access at
[huggingface.co/ratschlab/DeepSpotM](https://huggingface.co/ratschlab/DeepSpotM),
then authenticate before running:

```bash
huggingface-cli login
```

Once access is granted, `config/model/DeepSpotM.yaml`'s default
`MODEL.repo_id_or_path: ratschlab/DeepSpotM` downloads and caches the model
automatically on first use — no manual download step needed.

Alternatively, download it locally and point `repo_id_or_path` at that
directory instead:

```bash
huggingface-cli download ratschlab/DeepSpotM --local-dir <some_dir>
```

```yaml
MODEL:
  repo_id_or_path: <some_dir>   # must contain config.json, model.safetensors, tokens.csv
```

## Gene-embedding source

DeepSpotM is a multi-source model — `MODEL.source` selects which frozen
gene-embedding pathway to use: one of `evo2`, `orthrus`, `prott5`, `scgpt`,
`apertus`. Defaults to `scgpt`.

## Vendored code

`DeepSpotM/` is vendored from [ratschlab/DeepSpotM](https://github.com/ratschlab/DeepSpotM)
(code: PolyForm-Noncommercial-1.0.0; weights: CC-BY-NC-SA-4.0 — non-commercial
use only, see `DeepSpotM/LICENSE` and `DeepSpotM/WEIGHTS_LICENSE.md`).
