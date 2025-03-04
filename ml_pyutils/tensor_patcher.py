from __future__ import annotations

import dataclasses
import functools
import operator
from dataclasses import dataclass
from typing import TypedDict

import torch
import torch.nn.functional as F
from typing_extensions import Unpack


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


class TorchPatcherContext(TypedDict):
    original_size: Size
    overlap: int
    scale: int


class TorchPatcher:
    """
    Splits a tensor into overlapping patches in height and width (last two) dimensions.
    """

    def __init__(
        self,
        upscale_factor: int,
        patch_size: Size,
        overlap: int,
    ):
        self.patch_size = patch_size
        self.overlap = overlap
        self.upscale_factor = upscale_factor
        self.stride = self.caculate_stride()

    def caculate_tensor_padding_for_patches(
        self,
        input_size: Size,
        stride: Size,
    ) -> Size:
        """
        Calculate the required padding for an input tensor to ensure that last patches fits to the border of the tensor. The padding ensures that the last patches fit correctly into the tensor without truncation.

        https://asciiflow.com/#/

        tensor before padding
        - arrow denotes the needed padding
        - x is overlap between patches
        ┌───────────────────┐
        │patch_1   patch_2  │
        ├─────────┬────┬────│────┐
        │         │xxxx│    │    │
        │         │xxxx│    │    │
        │         │xxxx│    │    │
        │         │xxxx│    ├───►│
        │         │xxxx│    │    │
        │         │xxxx│    │    │
        │         │xxxx│    │    │
        ├─────────┴────┴────│────┘
        │                   │
        │                   │
        └───────────────────┘
        tensor after padding
        ┌────────────────────────┐
        │patch_1   patch_2       │
        ├─────────┬────┬─────────┤
        │         │xxxx│         │
        │         │xxxx│         │
        │         │xxxx│         │
        │         │xxxx│         │
        │         │xxxx│         │
        │         │xxxx│         │
        │         │xxxx│         │
        ├─────────┴────┴─────────┤
        │                        │
        │                        │
        └────────────────────────┘

        Args:
            input_size: A tuple representing the height and width of the input tensor.
            patch_size: A tuple representing the height and width of the patches.
            stride: A tuple representing the stride along the height and width dimensions.

        Returns:
            A tuple containing the amount of padding needed along the height and width dimensions.
            The padding should be applied in the bottom-right corner of the tensor.

        Example:
            input_size = (50, 50)
            patch_size = (40, 40)
            stride = (30, 30)
            # Returns: (20, 20)

        Notes:
            - The padding is calculated such that if the stride is equal to the padding needed, no padding is added.
            - The function ensures that the output tensor size will accommodate the given patches and strides without truncation.
        """
        patch_size = self.patch_size

        diff_h_patch_size = input_size.height - patch_size.height
        diff_w_patch_size = input_size.width - patch_size.width

        last_pad_needed_amount_h = diff_h_patch_size % stride.height
        last_pad_needed_amount_w = diff_w_patch_size % stride.width

        free_additional_pad_h = stride.height - last_pad_needed_amount_h
        free_additional_padding_width = stride.width - last_pad_needed_amount_w

        # final operation ensures that if the previous step resulted in stride_height, we get 0 instead.
        # this is because adding stride_height padding is equivalent to adding no padding in terms of the output size
        padding_height = (
            free_additional_pad_h if free_additional_pad_h != stride.height else 0
        )
        padding_width = (
            free_additional_padding_width
            if free_additional_padding_width != stride.width
            else 0
        )

        return Size(height=padding_height, width=padding_width)

    def caculate_stride(self) -> Size:
        """Calculate the stride based on the patch size and overlap."""
        patch_size = self.patch_size
        overlap = self.overlap
        stride_height = patch_size.height - overlap
        stride_width = patch_size.width - overlap
        return Size(height=stride_height, width=stride_width)

    def caculate_num_patches(
        self,
        input_size: Size,
        padding: Size,
        patch_size: Size,
        stride: Size,
    ) -> Size:
        """Calculate the number of patches that can be extracted from an input tensor."""
        num_patches_h = (
            input_size.height + padding.height - patch_size.height
        ) // stride.height + 1
        num_patches_w = (
            input_size.width + padding.width - patch_size.width
        ) // stride.width + 1
        return Size(height=num_patches_h, width=num_patches_w)

    def calculate_stride_padding_and_num_patches(
        self,
        input_size: Size,
    ) -> tuple[Size, Size]:
        patch_size = self.patch_size
        padding = self.caculate_tensor_padding_for_patches(
            input_size=input_size,
            stride=self.stride,
        )
        num_patches = self.caculate_num_patches(
            input_size=input_size,
            padding=padding,
            patch_size=patch_size,
            stride=self.stride,
        )
        return padding, num_patches

    def split_tensor_with_overlap(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Split a tensor into overlapping patches.

        Note:
            - The overlap should be less than the minimum dimension of the patch_size.

        This function takes an input tensor and splits it into overlapping patches
        of specified size. It handles cases where the input tensor dimensions are not
        perfect multiples of the patch size minus overlap, by adding appropriate padding.

        The function performs the following steps:
        1. Calculates the stride based on the patch size and overlap.
        2. Determines the necessary padding for an original image to ensure that all patches are of equal size.
        3. Applies padding to the input tensor.
        4. Uses torch.nn.functional.unfold to create sliding patches.
        5. Reshapes and permutes the result to get the desired output format.

        Args:
            input_tensor: The input tensor to be split with shape (B, C, H, W).
            patch_size: The size of each patch (Ph, Pw).
            overlap: The number of overlapping pixels between adjacent patches.

        Returns:
            A tensor containing the overlapping patches (B, Nh, Nw, C, Ph, Pw).

            Nh: Number of patches along the height dimension
            Nw: Number of patches along the width dimension
            Ph: Patch height
            Pw: Patch width
        """
        patch_size = self.patch_size
        overlap = self.overlap
        assert input_tensor.dim() >= 4, (
            "Input tensor must have at least 4 dimensions (batch, channels, height, width)."
        )
        assert overlap < min(patch_size), (
            "Overlap must be less than the minimum dimension of the patch size."
        )

        b, *middle_dims, c, input_height, input_width = input_tensor.shape
        input_size = Size(height=input_height, width=input_width)

        temp_batch_dim = functools.reduce(operator.mul, [b, *middle_dims])

        # | B * ...M | C | H | W |, merge middle dimensions with batch dimension
        input_tensor = input_tensor.reshape(temp_batch_dim, *input_tensor.size()[-3:])

        padding, num_patches = self.calculate_stride_padding_and_num_patches(input_size)

        padding_height, padding_width = padding
        num_patches_h, num_patches_w = num_patches

        # Pad input_tensor with zeros on the right and bottom side.
        padded_input_tensor = F.pad(input_tensor, (0, padding_width, 0, padding_height))

        # | B * ...M | C * Ph * Pw | Nh * Nw |, use unfold to create sliding patches
        patches = F.unfold(
            padded_input_tensor,
            kernel_size=patch_size.as_tuple(),
            stride=self.stride.as_tuple(),
        )

        # | B | ...M | C | Ph | Pw | Nh | Nw |, reshape to a meaningful format
        patches = patches.reshape(
            b,  # 0
            *middle_dims,  # md_start - md_end
            c,  # -5
            patch_size.height,  # -4
            patch_size.width,  # -3
            num_patches_h,  # -2
            num_patches_w,  # -1
        )

        dims = list(range(patches.dim()))

        # [B] + [Nh, Nw] + [...M, C, Ph, Pw]
        dims = [dims[0]] + dims[-2:] + dims[1:-2]

        # | B | Nh | Nw | ...M | C | Ph | Pw |, bring Nh and Nw dimensions next to B dimension
        patches = patches.permute(*dims)
        return patches

    def reconstruct_tensor_from_upscaled_patches(
        self,
        patches: torch.Tensor,
        original_size: Size,
        overlap: int,
        upscale_factor: int,
    ) -> torch.Tensor:
        """
        Reconstruct a tensor from upscaled overlapping patches.

        This function takes upscaled overlapping patches and reconstructs the original tensor,
        accounting for the scaling factor. It performs the inverse operation of
        'split_tensor_with_overlap' while also handling the upscaling.

        The function performs the following steps:
        1. Calculates the stride based on the input size and overlap.
        2. Computes the scaled dimensions and adjusts for potential rounding errors.
        3. Rescales the patches to the original size.
        4. Uses a fold operation to reconstruct the tensor.
        5. Removes any extra padding to match the original size.

        Args:
            patches : The input tensor containing upscaled overlapping patches (B, Nh, Nw, ...M, C, Ph, Pw).
            original_size: The original size of the tensor before splitting (H, W).
            patch_size: The size of each patch before upscaling (Ph, Pw).
            overlap: The number of overlapping pixels between adjacent patches before upscaling.
            scale: The factor by which the patches were upscaled (e.g., 2 for 2x, 4 for 4x).

        Returns:
            torch.Tensor: The reconstructed tensor (B, ...M, C, H, W).
        Note:
            - The function assumes that all patches have been upscaled by the same factor.
            - The original_size should be the size before any padding was applied in the split operation.
        """
        if patches.dim() < 6:
            raise ValueError(
                "Input should have at least 6 dimensions (B, Nh, Nw, ...M, C, Ph, Pw)."
            )

        # note, patch_height and patch_width are already scaled!
        b, num_patches_h, num_patches_w, *middle_dims, c, patch_height, patch_width = (
            patches.shape
        )

        num_patches = num_patches_h * num_patches_w
        num_patches = num_patches_h * num_patches_w

        original_h, original_w = original_size

        # Calculate stride
        stride_height = patch_height - (overlap * upscale_factor)
        stride_width = patch_width - (overlap * upscale_factor)
        stride = (stride_height, stride_width)

        # Calculate scaled dimensions
        scaled_height = original_h * upscale_factor
        scaled_width = original_w * upscale_factor

        # Adjust for potential rounding errors in the scaling process
        padding_height = (
            stride_height - (scaled_height - patch_height) % stride_height
        ) % stride_height
        padding_width = (
            stride_width - (scaled_width - patch_width) % stride_width
        ) % stride_width

        dims = list(range(patches.dim()))

        # [b] + [...M, channels, patch_height, patch_width] + [num_patches_h, num_patches_w]
        dims = [dims[0], *dims[3:], *dims[1:3]]

        # | B | ...M | C | Ph | Pw | Nh | Nw |, move Nh and Nw dimensions at the end
        patches = patches.permute(*dims)

        temp_batch_dim = functools.reduce(operator.mul, [b, *middle_dims])

        # | B * ...M | C * Ph * Pw | Nh * Nw |, merge middle dimensions with batch dimension
        patches = patches.reshape(temp_batch_dim, -1, num_patches)

        # | B * ...M | C | H * s | W * s |, fold the patches to reconstruct the tensor
        output_folded = F.fold(
            patches,
            output_size=(scaled_height + padding_height, scaled_width + padding_width),
            kernel_size=(patch_height, patch_width),
            stride=stride,
        )

        # | B * ...M | C | H * s | W * s |, create normalization mask (number of 'active' patches at each pixel)
        patch_count_mask = F.fold(
            torch.ones_like(patches),
            output_size=(scaled_height + padding_height, scaled_width + padding_width),
            kernel_size=(patch_height, patch_width),
            stride=stride,
        )
        output_padded = output_folded / patch_count_mask

        # Remove extra padding
        output = output_padded[:, :, :scaled_height, :scaled_width]

        # | B | ...M | C | H * s | W * s |
        output = output.reshape(b, *middle_dims, c, scaled_height, scaled_width)
        return output

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, TorchPatcherContext]:
        patches = self.split_tensor_with_overlap(x)
        h, w = x.shape[-2:]
        context = TorchPatcherContext(
            original_size=Size(height=h, width=w),
            overlap=self.overlap,
            scale=self.upscale_factor,
        )
        return patches, context

    def undo(
        self,
        x: torch.Tensor,
        **kwargs: Unpack[TorchPatcherContext],
    ) -> torch.Tensor:
        original_size = kwargs["original_size"]
        overlap = kwargs["overlap"]
        scale = kwargs["scale"]
        return self.reconstruct_tensor_from_upscaled_patches(
            patches=x,
            original_size=original_size,
            overlap=overlap,
            upscale_factor=scale,
        )
