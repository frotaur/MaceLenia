import torch
import torch.nn.functional as F
from typing import Tuple

def unfold3d(tensor_padded: torch.Tensor, kernel_size: Tuple[int, int, int] = (3, 3, 3),stride: int = 1) -> torch.Tensor:

    if tensor_padded.ndim != 4:
        raise ValueError(f"Input tensor must be 4D (B, C_pad, H_pad, W_pad), but got shape {tensor_padded.shape}")
    if len(kernel_size) != 3:
         raise ValueError(f"kernel_size must be a tuple of 3 integers (KC, KH, KW), but got {kernel_size}")


    kernel_C, kernel_H, kernel_W = kernel_size


    unfold_w = tensor_padded.unfold(dimension=3, size=kernel_W, step=stride)


    unfold_hw = unfold_w.unfold(dimension=2, size=kernel_H, step=stride)

    unfold_hw = unfold_hw.permute(0, 1, 2, 3, 5, 4)


    unfold_chw = unfold_hw.unfold(dimension=1, size=kernel_C, step=stride)

    unfold_chw = unfold_chw.permute(0, 1, 6, 4, 5, 2, 3)

    return unfold_chw