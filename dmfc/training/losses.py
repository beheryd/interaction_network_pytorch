"""Loss-mask system for the four Rajalingham IN training variants.

The 7-d Rajalingham output is laid out as:
    [0, 1]  ball position (x, y) — supervised on every step (vis-sim variants)
    [2, 3]  ball position (x, y) — supervised on the visible epoch only
    [4, 5]  ball position (x, y) — supervised on the occluded epoch only
    [6]     final intercept y    — supervised on every valid step

Each loss variant masks which output indices contribute on each step:

    Variant         Rajalingham label  Supervised indices (per step)
    --------------  -----------------  -------------------------------------
    mov             Intercept          [6] every valid step
    vis-mov         Vis                [2,3] visible-and-valid; [6] every valid
    vis-sim-mov     Vis+Occ            [0,1] every valid; [6] every valid
    sim-mov         Vis&Occ            [4,5] occluded-and-valid; [6] every valid

The loss is mean-squared-error averaged over supervised positions only.
"""

from __future__ import annotations

import torch

LOSS_VARIANTS: tuple[str, ...] = ("mov", "vis-mov", "vis-sim-mov", "sim-mov")
OUTPUT_DIM: int = 7


def supervision_mask(
    variant: str, visible_mask: torch.Tensor, valid_mask: torch.Tensor
) -> torch.Tensor:
    """Build the per-step per-output supervision indicator.

    Args:
        variant: one of LOSS_VARIANTS.
        visible_mask: [B, T] — 1 where ball is visible, 0 where occluded.
        valid_mask:   [B, T] — 1 for in-trial steps, 0 for padding past t_f.

    Returns:
        mask: [B, T, OUTPUT_DIM] with 1.0 on supervised positions, else 0.0.
    """
    if variant not in LOSS_VARIANTS:
        raise ValueError(f"unknown loss variant {variant!r}; expected one of {LOSS_VARIANTS}")
    b, t = valid_mask.shape
    mask = torch.zeros(b, t, OUTPUT_DIM, dtype=valid_mask.dtype, device=valid_mask.device)
    visible_and_valid = visible_mask * valid_mask
    occluded_and_valid = (1.0 - visible_mask) * valid_mask
    if variant == "mov":
        mask[..., 6] = valid_mask
    elif variant == "vis-mov":
        mask[..., 2] = visible_and_valid
        mask[..., 3] = visible_and_valid
        mask[..., 6] = valid_mask
    elif variant == "vis-sim-mov":
        mask[..., 0] = valid_mask
        mask[..., 1] = valid_mask
        mask[..., 6] = valid_mask
    elif variant == "sim-mov":
        mask[..., 4] = occluded_and_valid
        mask[..., 5] = occluded_and_valid
        mask[..., 6] = valid_mask
    return mask


def compute_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    variant: str,
    visible_mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Masked MSE loss for one of the four variants.

    Args:
        outputs: [B, T, OUTPUT_DIM] — model predictions.
        targets: [B, T, OUTPUT_DIM] — ground truth (only supervised positions
            matter; unsupervised positions can hold anything).
        variant: one of LOSS_VARIANTS.
        visible_mask: [B, T] — 1 visible, 0 occluded.
        valid_mask:   [B, T] — 1 in-trial, 0 padded.

    Returns:
        scalar tensor — mean squared error over supervised positions.
    """
    mask = supervision_mask(variant, visible_mask, valid_mask)
    err = (outputs - targets) ** 2
    masked_sum = (err * mask).sum()
    denom = mask.sum().clamp(min=1.0)
    return masked_sum / denom


__all__ = ["LOSS_VARIANTS", "OUTPUT_DIM", "compute_loss", "supervision_mask"]
