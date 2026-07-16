import torch
from abc import abstractmethod
from .utils import get_normalize_params, get_eval_transforms


def _resolve_weights(weights_path, hf_repo_id=None, hf_filename=None):
    """Resolve model weights: explicit path > HuggingFace Hub cache.

    Raises FileNotFoundError if no weights can be found.
    """
    wp = weights_path or ""
    if not wp and hf_repo_id and hf_filename:
        from huggingface_hub import hf_hub_download
        wp = hf_hub_download(hf_repo_id, hf_filename, local_files_only=True)
    if not wp:
        raise FileNotFoundError(
            f"Could not find pretrained weights for {hf_repo_id}. "
            f"Expected file: {hf_filename}. "
            f"Download the model or pass weights_path explicitly."
        )
    return wp


def _load_state_dict(model, path):
    """Load state dict, auto-detecting safetensors vs PyTorch format."""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(path)
    else:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)


class ImageEncoder(torch.nn.Module):

    # LoRA target module names — subclasses override for their architecture
    LORA_TARGET_MODULES = ["qkv"]  # default for timm ViTs

    # Image normalization — every subclass must set these
    NORM_MEAN = None
    NORM_STD = None

    def __init__(self, weights_path=None, pretrained=True, **build_kwargs):
        super().__init__()
        self.weights_path = weights_path
        # When False, encoders that support it build the architecture without
        # downloading pretrained weights (weights come from a DeepSpotM
        # checkpoint instead). Stored as an attribute so it never leaks into
        # ``build_kwargs``, which are forwarded verbatim to the image transforms.
        self.pretrained = pretrained
        self.build_kwargs = build_kwargs
        self.model, self.transforms, self.hidden_dim = self._build(self.weights_path, **self.build_kwargs)

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError

    @abstractmethod
    def _build(self, weights_path, **build_kwargs):
        raise NotImplementedError

    def set_transforms(self, **build_kwargs):
        """Rebuild image transforms without reloading the model."""
        build_kwargs = {**self.build_kwargs, **build_kwargs}
        self.transforms = get_eval_transforms(
            self.NORM_MEAN, self.NORM_STD,
            target_img_size=224, **build_kwargs,
        )

    def forward_patch_tokens(self, pixel_values):
        """Return spatial patch tokens (B, N_patches, D), excluding CLS/register tokens.

        Handles three model families generically:
          1. timm: ``forward_features`` returns the token sequence;
             ``num_prefix_tokens`` reflects CLS + register tokens.
          2. HF AutoModel with a ``config``: ``last_hidden_state`` returns the
             token sequence; we strip 1 CLS (if ``config.use_cls_token`` is
             True / unset) + ``config.num_register_tokens`` (DINOv2-style).
          3. Anything else: raise — subclasses must override.
        """
        if hasattr(self.model, "forward_features"):
            tokens = self.model.forward_features(pixel_values)
            n_prefix = getattr(self.model, "num_prefix_tokens", 1)
            return tokens[:, n_prefix:, :]

        if hasattr(self.model, "config"):
            cfg = self.model.config
            n_cls = 1 if getattr(cfg, "use_cls_token", True) else 0
            n_reg = getattr(cfg, "num_register_tokens", 0)
            output = self.model(pixel_values).last_hidden_state
            return output[:, n_cls + n_reg:, :]

        raise NotImplementedError(
            f"forward_patch_tokens not implemented for {type(self).__name__}. "
            "Override in your encoder subclass."
        )

    @property
    def patch_dim(self):
        """Per-patch embedding dimension."""
        if hasattr(self.model, "embed_dim"):
            return self.model.embed_dim
        if hasattr(self.model, "config") and hasattr(self.model.config, "hidden_size"):
            return self.model.config.hidden_size
        return self.hidden_dim


class _TimmEncoder(ImageEncoder):
    """Base class for timm-based encoders.

    Subclasses declare model + weights metadata via class attributes; the build
    pipeline (create_model → resolve weights → load → transforms) is shared.
    """

    TIMM_NAME = None
    TIMM_KWARGS = {}
    HF_REPO_ID = None
    HF_FILENAME = None
    HIDDEN_DIM = None
    # "cls_only": forward returns the model's default (CLS) output.
    # "cls_plus_mean_patch": forward returns cat(CLS, patch_tokens.mean(1)).
    POOLING = "cls_only"

    def _build(self, weights_path, **transform_kwargs):
        import timm

        model = timm.create_model(
            self.TIMM_NAME,
            pretrained=False,
            num_classes=0,
            **self.TIMM_KWARGS,
        )
        wp = _resolve_weights(weights_path, self.HF_REPO_ID, self.HF_FILENAME)
        _load_state_dict(model, wp)

        transform = get_eval_transforms(
            self.NORM_MEAN, self.NORM_STD,
            target_img_size=224, **transform_kwargs,
        )
        return model, transform, self.HIDDEN_DIM

    def forward(self, pixel_values, **kwargs):
        if self.POOLING == "cls_only":
            return self.model(pixel_values)
        if self.POOLING == "cls_plus_mean_patch":
            output = self.model.forward_features(pixel_values)
            n_prefix = getattr(self.model, "num_prefix_tokens", 1)
            cls = output[:, 0]
            patches = output[:, n_prefix:]
            return torch.cat([cls, patches.mean(1)], dim=-1)
        raise ValueError(f"Unknown POOLING: {self.POOLING!r}")


# ── Pathology foundation models (timm) ─────────────────────────────────────

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_BIOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
_BIOPTIMUS_STD = (0.211883, 0.230117, 0.177517)


class UNI1Encoder(_TimmEncoder):
    TIMM_NAME = "vit_large_patch16_224"
    TIMM_KWARGS = {"img_size": 224, "patch_size": 16, "init_values": 1e-5, "dynamic_img_size": True}
    HF_REPO_ID = "MahmoodLab/UNI"
    HF_FILENAME = "pytorch_model.bin"
    HIDDEN_DIM = 1024
    NORM_MEAN = _IMAGENET_MEAN
    NORM_STD = _IMAGENET_STD


class UNI2Encoder(_TimmEncoder):
    TIMM_NAME = "vit_giant_patch14_224"
    # SwiGLU + register tokens; mirrors the official UNI2-h config.
    @classmethod
    def _build_kwargs(cls):
        import timm
        return {
            "img_size": 224, "patch_size": 14,
            "depth": 24, "num_heads": 24,
            "init_values": 1e-5, "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "no_embed_class": True,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
            "reg_tokens": 8, "dynamic_img_size": True,
        }

    HF_REPO_ID = "MahmoodLab/UNI2-h"
    HF_FILENAME = "pytorch_model.bin"
    HIDDEN_DIM = 1536
    NORM_MEAN = _IMAGENET_MEAN
    NORM_STD = _IMAGENET_STD

    def _build(self, weights_path, **transform_kwargs):
        # UNI2 needs runtime kwargs (timm.layers.SwiGLUPacked import happens at
        # build time, not class-body time, to avoid importing timm eagerly).
        self.TIMM_KWARGS = type(self)._build_kwargs()
        return super()._build(weights_path, **transform_kwargs)


class H0miniEncoder(_TimmEncoder):
    TIMM_NAME = "vit_base_patch14_reg4_dinov2"
    HF_REPO_ID = "bioptimus/H0-mini"
    HF_FILENAME = "model.safetensors"
    HIDDEN_DIM = 768 * 2  # CLS + mean-pooled patch features
    NORM_MEAN = _BIOPTIMUS_MEAN
    NORM_STD = _BIOPTIMUS_STD
    POOLING = "cls_plus_mean_patch"

    @classmethod
    def _build_kwargs(cls):
        import timm
        return {
            "mlp_ratio": 5.33334, "reg_tokens": 4,
            "init_values": 1e-05, "img_size": 224,
            "global_pool": "", "dynamic_img_size": True,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
        }

    def _build(self, weights_path, **transform_kwargs):
        self.TIMM_KWARGS = type(self)._build_kwargs()
        return super()._build(weights_path, **transform_kwargs)


class H0Encoder(_TimmEncoder):
    TIMM_NAME = "vit_giant_patch14_reg4_dinov2"
    TIMM_KWARGS = {"img_size": 224, "init_values": 1e-5, "dynamic_img_size": True}
    HF_REPO_ID = "bioptimus/H-optimus-0"
    HF_FILENAME = "model.safetensors"
    HIDDEN_DIM = 1536
    NORM_MEAN = _BIOPTIMUS_MEAN
    NORM_STD = _BIOPTIMUS_STD


class H1Encoder(_TimmEncoder):
    TIMM_NAME = "vit_giant_patch14_reg4_dinov2"
    TIMM_KWARGS = {"img_size": 224, "init_values": 1e-5, "dynamic_img_size": True}
    HF_REPO_ID = "bioptimus/H-optimus-1"
    HF_FILENAME = "model.safetensors"
    HIDDEN_DIM = 1536
    NORM_MEAN = _BIOPTIMUS_MEAN
    NORM_STD = _BIOPTIMUS_STD


# ── HuggingFace AutoModel encoders ─────────────────────────────────────────


class ImageCLIPEncoder(ImageEncoder):
    LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj"]
    # Standard OpenAI CLIP normalization.
    NORM_MEAN = (0.48145466, 0.4578275, 0.40821073)
    NORM_STD = (0.26862954, 0.26130258, 0.27577711)

    def _build(self, weights_path, **transform_kwargs):
        from transformers import AutoImageProcessor, CLIPModel

        clip_model = CLIPModel.from_pretrained(weights_path, local_files_only=True)

        class VisionOnlyModel(torch.nn.Module):
            def __init__(self, vision_model, projection):
                super().__init__()
                self.vision_encoder = vision_model
                self.projection = projection

            def forward(self, pixel_values):
                features = self.vision_encoder(pixel_values).pooler_output
                return self.projection(features)

        model = VisionOnlyModel(
            vision_model=clip_model.vision_model,
            projection=clip_model.visual_projection,
        )

        # If the checkpoint ships an image processor, prefer its mean/std over
        # the class defaults — covers fine-tuned CLIP variants with custom norms.
        try:
            processor = AutoImageProcessor.from_pretrained(
                weights_path, local_files_only=True, use_fast=True,
            )
            mean, std = get_normalize_params(processor)
            if mean is not None and std is not None:
                type(self).NORM_MEAN, type(self).NORM_STD = tuple(mean), tuple(std)
        except Exception:
            pass

        transform = get_eval_transforms(
            self.NORM_MEAN, self.NORM_STD, target_img_size=224, **transform_kwargs,
        )
        return model, transform, 512

    def forward(self, pixel_values):
        return self.model(pixel_values)

    def forward_patch_tokens(self, pixel_values):
        output = self.model.vision_encoder(pixel_values)
        return output.last_hidden_state[:, 1:, :]  # skip CLS


class MidnightEncoder(ImageEncoder):
    LORA_TARGET_MODULES = ["query", "key", "value"]
    NORM_MEAN = (0.5, 0.5, 0.5)
    NORM_STD = (0.5, 0.5, 0.5)

    def _build(self, weights_path, **transform_kwargs):
        """Build the Midnight backbone.

        ``self.pretrained=True`` (default) downloads the kaiko-ai/midnight
        weights. ``self.pretrained=False`` constructs the architecture from a
        bundled config with random init — used when the trained backbone
        weights are loaded from a DeepSpotM checkpoint that has them baked in,
        so no network access or dependency on the kaiko-ai/midnight repo is
        required at load time.
        """
        from transformers import AutoModel, AutoConfig

        if weights_path:
            model = AutoModel.from_pretrained(weights_path, local_files_only=True)
        elif self.pretrained:
            model = AutoModel.from_pretrained("kaiko-ai/midnight")
        else:
            import importlib.resources
            with importlib.resources.path(
                "deepspotm.assets", "midnight_config.json",
            ) as cfg_path:
                config = AutoConfig.from_pretrained(str(cfg_path))
            model = AutoModel.from_config(config)

        transform = get_eval_transforms(
            self.NORM_MEAN, self.NORM_STD, target_img_size=224, **transform_kwargs,
        )
        return model, transform, 1536 * 2  # cls + patch emb

    def forward(self, pixel_values):
        output = self.model(pixel_values).last_hidden_state
        cls_embedding = output[:, 0, :]
        patch_embeddings = output[:, 1:, :]
        return torch.cat([cls_embedding, patch_embeddings.mean(1)], dim=-1)

    def forward_patch_tokens(self, pixel_values):
        output = self.model(pixel_values).last_hidden_state
        return output[:, 1:, :]  # skip CLS


# ── Registry ──────────────────────────────────────────────────────────────

ENCODER_REGISTRY = {
    "H0Encoder": H0Encoder,
    "H0miniEncoder": H0miniEncoder,
    "H1Encoder": H1Encoder,
    "UNI1Encoder": UNI1Encoder,
    "UNI2Encoder": UNI2Encoder,
    "MidnightEncoder": MidnightEncoder,
    "ImageCLIPEncoder": ImageCLIPEncoder,
}


def build_encoder(name: str, **kwargs) -> ImageEncoder:
    """Factory function to build an encoder by name."""
    if name not in ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown encoder: {name!r}. "
            f"Available: {list(ENCODER_REGISTRY.keys())}"
        )
    return ENCODER_REGISTRY[name](**kwargs)


def build_encoder_transforms(name: str, **kwargs):
    """Build image transforms for an encoder without loading model weights."""
    if name not in ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown encoder: {name!r}. "
            f"Available: {list(ENCODER_REGISTRY.keys())}"
        )
    cls = ENCODER_REGISTRY[name]
    return get_eval_transforms(
        cls.NORM_MEAN, cls.NORM_STD, target_img_size=224, **kwargs,
    )
