# STPath

## Checkpoint Setup

Download the pretrained STPath weight from [Hugging Face (tlhuang/STPath)](https://huggingface.co/tlhuang/STPath) and place it in the `weight/` directory:

```bash
# Option A — using huggingface_hub
python - << 'EOF'
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="tlhuang/STPath",
    filename="stfm.pth",
    local_dir="src/model/stpath/weight",
)
EOF

# Option B — using the CLI
huggingface-cli download tlhuang/STPath stfm.pth \
    --local-dir src/model/stpath/weight
```

The expected file location after download:

```
src/model/stpath/weight/stfm.pth
```

> **Note:** STPath also requires the [Prov-GigaPath](https://github.com/prov-gigapath/prov-gigapath) pretrained encoder.
> Follow the instructions in the official repository to obtain access and download the weights.
