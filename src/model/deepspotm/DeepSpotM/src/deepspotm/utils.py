import torch
import numpy as np
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision import transforms

def get_normalize_params(obj):
    """
    Extract mean and std from either:
      - torchvision Compose object (with Normalize)
      - CLIPImageProcessor object or dict (with image_mean and image_std)
    Returns (mean, std) or (None, None) if not found.
    """
    if isinstance(obj, transforms.Compose):
        for t in obj.transforms:
            if isinstance(t, transforms.Normalize):
                return t.mean, t.std
        return None, None

    if isinstance(obj, dict):
        return obj.get("image_mean"), obj.get("image_std")

    return getattr(obj, "image_mean", None), getattr(obj, "image_std", None)


def get_eval_transforms(mean, std, target_img_size=-1, center_crop=False):
    """
    Returns a transform pipeline that automatically handles NumPy, PIL, or tensor inputs.
    """
    trsforms = []

    # Step 1: Ensure input is PIL if it's NumPy
    def ensure_pil(x):
        if isinstance(x, np.ndarray):
            if x.dtype != np.uint8:
                x = (x * 255).clip(0, 255).astype(np.uint8)
            return Image.fromarray(x)

        elif torch.is_tensor(x):
            # CHW → HWC, float [0,1] → uint8 [0,255]
            x = x.detach().cpu()
            if x.ndim == 3 and x.shape[0] in (1, 3):
                x = x.permute(1, 2, 0)  # CHW → HWC
            x = (x.numpy() * 255).clip(0, 255).astype(np.uint8)
            return Image.fromarray(x)

        elif isinstance(x, Image.Image):
            return x

        else:
            raise TypeError(f"Unsupported input type: {type(x)}")
    trsforms.append(transforms.Lambda(ensure_pil))

    # Step 2: Apply resizing/cropping if needed (works only on PIL)
    if target_img_size > 0:
        trsforms.append(transforms.Resize(target_img_size, interpolation=InterpolationMode.BICUBIC))

    if center_crop:
        assert target_img_size > 0, "target_img_size must be set if center_crop is True"
        trsforms.append(transforms.CenterCrop(target_img_size))

    # Step 3: Convert to tensor if needed
    def ensure_tensor(x):
        if torch.is_tensor(x):
            return x
        else:
            return transforms.ToTensor()(x)
    trsforms.append(transforms.Lambda(ensure_tensor))

    # Step 4: Normalize if mean/std provided
    if mean is not None and std is not None:
        trsforms.append(transforms.Normalize(mean, std))

    return transforms.Compose(trsforms)
