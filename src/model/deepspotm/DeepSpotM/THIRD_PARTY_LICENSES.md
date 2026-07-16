# Third-party components

DeepSpotM bundles, or derives weights from, the following third-party models.
Each is redistributed under its own permissive license; the notices below are
retained as required.

| Component | Used as | License | Source |
|-----------|---------|---------|--------|
| Midnight (`kaiko-ai/midnight`) | Vision backbone (weights baked into `model.safetensors`) | MIT | https://huggingface.co/kaiko-ai/midnight |
| Evo 2 | Gene embedding source (`bio_evo2`) | Apache-2.0 | https://github.com/ArcInstitute/evo2 |
| Orthrus | Gene embedding source (`bio_orthrus`) | MIT | https://github.com/bowang-lab/Orthrus |
| ProtT5 (`Rostlab/prot_t5_xl_uniref50`) | Gene embedding source (`bio_prott5`) | Apache-2.0 | https://huggingface.co/Rostlab/prot_t5_xl_uniref50 |
| scGPT | Gene embedding source (`bio_scgpt`) | MIT | https://github.com/bowang-lab/scGPT |
| Apertus (`swiss-ai`) | Gene embedding source (`bio_apertus`) | Apache-2.0 | https://huggingface.co/swiss-ai |
