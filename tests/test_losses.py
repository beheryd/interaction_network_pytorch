"""Tests for the four-variant loss-mask system.

The hardest bug class in this whole milestone is "the wrong output index got
supervised". These tests pin down which indices each variant actually
backpropagates through.
"""

from __future__ import annotations

import pytest
import torch

from dmfc.training.losses import (
    LOSS_VARIANTS,
    OUTPUT_DIM,
    compute_loss,
    supervision_mask,
)


def _make_masks() -> tuple[torch.Tensor, torch.Tensor]:
    """Two trials, 5 steps each. Different visibility/validity layouts."""
    visible = torch.tensor([[1, 1, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=torch.float32)
    valid = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]], dtype=torch.float32)
    return visible, valid


def test_supervision_mask_per_variant() -> None:
    visible, valid = _make_masks()
    expected = {
        "mov": [0, 0, 0, 0, 0, 0, 7],  # only [6] every valid step
        "vis-mov": [0, 0, 3, 3, 0, 0, 7],  # [2,3] visible-and-valid; [6] valid
        "vis-sim-mov": [7, 7, 0, 0, 0, 0, 7],  # [0,1] every valid; [6] valid
        "sim-mov": [0, 0, 0, 0, 4, 4, 7],  # [4,5] occluded-and-valid; [6] valid
    }
    for variant, expected_per_index in expected.items():
        m = supervision_mask(variant, visible, valid)
        assert m.shape == (2, 5, OUTPUT_DIM)
        per_index = m.sum(dim=(0, 1)).tolist()
        assert per_index == expected_per_index, f"{variant}: {per_index}"


def test_supervision_mask_rejects_unknown_variant() -> None:
    visible, valid = _make_masks()
    with pytest.raises(ValueError):
        supervision_mask("bogus", visible, valid)


def test_compute_loss_zero_when_outputs_match_targets_at_supervised_indices() -> None:
    visible, valid = _make_masks()
    outputs = torch.zeros(2, 5, OUTPUT_DIM)
    targets = torch.zeros(2, 5, OUTPUT_DIM)
    # Push a non-zero error into an unsupervised position. Loss should still be 0.
    outputs[0, 0, 5] = 999.0  # index 5 is not supervised under "mov" or "vis-sim-mov"
    for variant in ("mov", "vis-sim-mov"):
        loss = compute_loss(outputs, targets, variant, visible, valid)
        assert loss.item() == pytest.approx(0.0)


def test_compute_loss_picks_up_error_only_at_supervised_indices() -> None:
    visible, valid = _make_masks()
    outputs = torch.zeros(2, 5, OUTPUT_DIM)
    targets = torch.zeros(2, 5, OUTPUT_DIM)
    # Set an error of 1.0 at every (b, t, i). compute_loss should average that to 1.0
    # over the supervised positions for any variant.
    outputs[...] = 1.0
    for variant in LOSS_VARIANTS:
        loss = compute_loss(outputs, targets, variant, visible, valid)
        assert loss.item() == pytest.approx(1.0), f"variant={variant}"


def test_compute_loss_isolates_indices_per_variant() -> None:
    """An error injected at index k contributes iff (variant, k) is supervised."""
    visible, valid = _make_masks()
    targets = torch.zeros(2, 5, OUTPUT_DIM)
    expected_supervised_indices = {
        "mov": {6},
        "vis-mov": {2, 3, 6},
        "vis-sim-mov": {0, 1, 6},
        "sim-mov": {4, 5, 6},
    }
    for variant, supervised in expected_supervised_indices.items():
        for k in range(OUTPUT_DIM):
            outputs = torch.zeros(2, 5, OUTPUT_DIM)
            outputs[..., k] = 1.0
            loss = compute_loss(outputs, targets, variant, visible, valid)
            if k in supervised:
                assert loss.item() > 0.0, f"variant={variant} idx={k} should be supervised"
            else:
                assert loss.item() == pytest.approx(
                    0.0
                ), f"variant={variant} idx={k} should not be supervised"
