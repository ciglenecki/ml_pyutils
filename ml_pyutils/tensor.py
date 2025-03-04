from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

PaddingMode = Literal["constant", "reflect", "replicate", "circular"]


@dataclass(frozen=True)
class Size:
    _: dataclasses.KW_ONLY
    height: int
    width: int

    def as_tuple(self) -> tuple[int, int]:
        return self.height, self.width

    def __iter__(self):
        return iter((self.height, self.width))

    def __eq__(self, value: object) -> bool:
        if isinstance(value, tuple):
            return value == (self.height, self.width)
        elif isinstance(value, Size):
            return value.height == self.height and value.width == self.width
        else:
            return False

    def __add__(self, other: Size) -> Size:
        return Size(height=self.height + other.height, width=self.width + other.width)

    def scale(self, factor: float) -> Size:
        return Size(
            height=round(self.height * factor), width=round(self.width * factor)
        )

    def __getitem__(self, idx: int) -> int:
        if idx == 0:
            return self.height
        elif idx == 1:
            return self.width
        else:
            raise IndexError("Index out of bounds")


@dataclass
class Padding:
    left: int
    top: int
    right: int
    bottom: int

    def __iter__(self):
        return iter((self.left, self.top, self.right, self.bottom))

    def get_torch_nn_functional_pad_args(self):
        return self.left, self.right, self.top, self.bottom

    def get_torchvision_transforms_pad_args(self):
        return self.left, self.top, self.right, self.bottom


def undo_padding(
    padded_tensor: torch.Tensor | np.ndarray, padding: Padding
) -> torch.Tensor | np.ndarray:
    """
    Undos the height and width padding of a tensor.

    Args:
        padded_tensor (torch.Tensor | np.ndarray):
            The input tensor with padding.
            Expected shape for torch is (..., C, H, W).
            Expected shape for numpy is (..., H, W, C).
        padding: left, top, right and bottom

    Returns:
        (torch.Tensor | np.ndarray): The tensor with padding removed.
    """
    left, top, right, bottom = padding.left, padding.top, padding.right, padding.bottom

    is_numpy = isinstance(padded_tensor, np.ndarray)
    is_torch = isinstance(padded_tensor, torch.Tensor)

    if is_numpy:
        height, width = padded_tensor.shape[-3:-1]
    elif is_torch:
        height, width = padded_tensor.shape[-2:]
    else:
        raise ValueError("Unsupported tensor type")

    if is_numpy:
        return padded_tensor[..., top : height - bottom, left : width - right, :]
    else:
        return padded_tensor[..., top : height - bottom, left : width - right]


def calc_padding_from_size(original_size: Size, target_size: Size) -> Padding:
    """
    Calculates the padding required to resize an original size to fit within a target size by padding on the right and bottom sides.

    This function computes the padding required to make an `original_size` fit into a `target_size`, such that the original content remains aligned at the top-left corner, and any additional space is padded on the right and bottom. It returns a `Padding` object with zero padding for the left and top, and computed padding for the right and bottom.

    Args:
        original_size: the original size.
        target_size: the target size.

    Returns:
        Padding: An object containing the padding values needed to fit the original size within the target size

    Example:
        >>> calc_padding_from_size((100, 200), (150, 250))
        Padding(left=0, top=0, right=50, bottom=50)
    """

    pad_width = max(target_size.width - original_size.width, 0)
    pad_height = max(target_size.height - original_size.height, 0)

    return Padding(left=0, top=0, right=pad_width, bottom=pad_height)


def pad_infinite(
    tensor: torch.Tensor,
    padding: Padding,
    padding_mode: PaddingMode,
) -> torch.Tensor:
    """
    Reflectively pad a tensor with the given padding, potentially applying padding multiple times.
    This function exists because PyTorch's F.pad() function does not support infinite padding.

    Args:
        tensor: The input tensor to pad.

        pad: A tuple of four integers (left, top, right, bottom).

        padding_mode: padding mode used for F.pad

    Returns: The padded tensor.
    """

    def pad_once(
        tensor: torch.Tensor, padding: Padding
    ) -> tuple[torch.Tensor, Padding]:
        max_h, max_w = tensor.size(-2), tensor.size(-1)

        padding_top_clamped = min(padding.top, max_h - 1)
        padding_bottom_clamped = min(padding.bottom, max_h - 1)
        padding_left_clamped = min(padding.left, max_w - 1)
        padding_right_clamped = min(padding.right, max_w - 1)

        clamped_padding = Padding(
            left=padding_left_clamped,
            top=padding_top_clamped,
            right=padding_right_clamped,
            bottom=padding_bottom_clamped,
        )

        padded_tensor = F.pad(
            tensor,
            clamped_padding.get_torch_nn_functional_pad_args(),
            mode=padding_mode,
        )

        remaining_pad = Padding(
            left=padding.left - padding_left_clamped,
            top=padding.top - padding_top_clamped,
            right=padding.right - padding_right_clamped,
            bottom=padding.bottom - padding_bottom_clamped,
        )

        return padded_tensor, remaining_pad

    while any(dim > 0 for dim in padding):
        tensor, padding = pad_once(tensor, padding)

    return tensor


def min_max_scale(t: torch.Tensor | np.ndarray):
    return (t - t.min()) / (t.max() - t.min())


def tensor_sum_of_elements_to_one(tensor: torch.Tensor, dim):
    """Scales elements of the tensor so that the sum is 1."""
    return tensor / torch.sum(tensor, dim=dim, keepdim=True)
