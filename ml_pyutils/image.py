from pathlib import Path

import numpy as np
import torch
import torchvision.utils
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


def save_image(file_path: Path, img: np.ndarray | torch.Tensor, **kwargs):
    """
    Saves numpy array (HWC) or torch tensor (CHW) to the given file path, by lirfu.
    """
    if file_path.parent:
        file_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(img, np.ndarray):
        img_pil = Image.fromarray(img)
    elif isinstance(img, Image.Image):
        img_pil = img
    elif isinstance(img, torch.Tensor):
        torchvision.utils.save_image(img, fp=file_path, **kwargs)
        return
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")
    img_pil.save(
        file_path,
        optimize=kwargs.get("optimize", False),
        compress_level=kwargs.get("compress_level", 0),
    )


def validate_image(
    image: np.ndarray | torch.Tensor,
    allowed_num_channels: tuple[int, ...] = (0, 1, 3, 4),
) -> None:
    """Validates that the input image has an appropriate shape and number of channels.

    Args:
        image: The input image, either as a NumPy array (..., H, W, C) or a Torch tensor (..., C, H, W).
        allowed_num_channels: A tuple of allowed channel numbers (default: (0, 1, 3, 4)).

    Raises:
        ValueError: If the image format or number of channels is invalid.
    """
    HELP_TEXT = (
        f"Input image must be a NumPy array (..., H, W, C) or a torch tensor (..., C, H, W). "
        f"Number of channels (C) must be one of {allowed_num_channels}."
    )

    if isinstance(image, np.ndarray):
        if image.ndim < 2:
            raise ValueError(HELP_TEXT)
        if image.ndim == 2 and 0 in allowed_num_channels:
            num_channels = 0  # Grayscale image without explicit channel dimension
        elif image.ndim == 3:
            _, _, num_channels = image.shape  # (H, W, C)
        else:
            raise ValueError(HELP_TEXT)
    elif isinstance(image, torch.Tensor):
        if image.ndim not in {3, 4}:  # Must be (C, H, W) or (B, C, H, W)
            raise ValueError(HELP_TEXT)
        num_channels = image.shape[-3]  # Extract channel dimension (C)
    else:
        raise TypeError("Image must be either a NumPy array or a Torch tensor.")

    if num_channels not in allowed_num_channels:
        raise ValueError(HELP_TEXT)
