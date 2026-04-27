"""Tests for ``dmfc.analysis.two_stage_endpoint``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dmfc.analysis.two_stage_endpoint import (
    KIN_DIM,
    TwoStageResult,
    kinematics_for_canonical_79,
    two_stage_decode,
)

PILOT_RUN_DIR = Path("runs/in_sim-mov_h10_s0_20260426-125840_fd614df-dirty")


def test_kinematics_for_canonical_79_shape() -> None:
    kin, valid = kinematics_for_canonical_79(T_max=80)
    assert kin.shape == (79, 80, KIN_DIM)
    assert valid.shape == (79, 80)
    # Every condition should have at least 1 valid bin.
    assert valid.sum(axis=1).min() > 0
    # Initial bin is the spec's (x0, y0, dx, dy).
    from dmfc.envs.conditions import load_conditions

    specs = load_conditions()
    for i, s in enumerate(specs[:5]):
        np.testing.assert_allclose(kin[i, 0, 0], s.x0)
        np.testing.assert_allclose(kin[i, 0, 1], s.y0)
        np.testing.assert_allclose(kin[i, 0, 2], s.dx)
        np.testing.assert_allclose(kin[i, 0, 3], s.dy)


def _synthetic_kinematics_endpoint(
    n_cond: int, T: int, K: int, rng: np.random.Generator, w: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Per-condition kinematic factor that persists across time + small bin noise.

    Mirrors Mental Pong's structural property: kinematics at any time carry
    the same condition identity that determines the endpoint.
    """
    factor = rng.normal(size=(n_cond, K))
    bin_noise = rng.normal(scale=0.05, size=(n_cond, T, K))
    kinematics = factor[:, None, :] + bin_noise
    if w is None:
        w = rng.normal(size=(K,))
    endpoint = factor @ w
    return kinematics, endpoint


def test_states_equal_kinematics_perfect_recovery() -> None:
    """If states ARE the kinematics, both direct and kinematics-mediated → 1."""
    rng = np.random.default_rng(0)
    n_cond, T, K = 30, 6, 4
    kinematics, endpoint = _synthetic_kinematics_endpoint(n_cond, T, K, rng)
    states = kinematics.copy()

    res = two_stage_decode(states, kinematics, endpoint, n_splits=3)
    assert isinstance(res, TwoStageResult)
    # At every timestep, all three routes should fully recover the endpoint.
    assert np.nanmean(res.direct_r) > 0.95
    assert np.nanmean(res.kinematics_mediated_r) > 0.95
    assert np.nanmean(res.kinematics_only_r) > 0.95


def test_states_uncorrelated_with_kinematics_collapses_kinmed() -> None:
    """If states have zero info about kinematics, kinematics-mediated → 0
    even when kinematics-only (true kinematics) is high."""
    rng = np.random.default_rng(1)
    n_cond, T, K = 60, 6, 4
    kinematics, endpoint = _synthetic_kinematics_endpoint(
        n_cond, T, K, rng, w=np.array([0.7, -0.3, 0.5, 0.2])
    )
    # States are pure noise — uncorrelated with kinematics
    states = rng.normal(size=(n_cond, T, 8))

    res = two_stage_decode(states, kinematics, endpoint, n_splits=3)
    # Kinematics-only baseline should be high (true kinematics → endpoint linearly)
    assert np.nanmean(res.kinematics_only_r) > 0.7
    # State→kinematics should be near zero
    assert np.nanmean(np.abs(res.state_to_kinematics_r)) < 0.4
    # Kinematics-mediated curve should drop because Stage 1 fails
    assert np.nanmean(res.kinematics_mediated_r) < 0.4


def test_partial_kinematic_signal_gap() -> None:
    """When states encode kinematics noisily, kinmed sits between direct and 0."""
    rng = np.random.default_rng(2)
    n_cond, T, K = 80, 5, 4
    kinematics, endpoint = _synthetic_kinematics_endpoint(n_cond, T, K, rng)

    # States: kinematics with extra noise (Stage 1 will be imperfect).
    states = kinematics + rng.normal(scale=0.5, size=kinematics.shape)

    res = two_stage_decode(states, kinematics, endpoint, n_splits=3)
    finite = np.isfinite(res.direct_r) & np.isfinite(res.kinematics_mediated_r)
    assert finite.any()
    # Kinematics-only ≥ kinmed (kinmed pays for Stage-1 noise)
    assert np.nanmean(res.kinematics_only_r) >= np.nanmean(res.kinematics_mediated_r) - 1e-3


def test_valid_mask_excludes_invalid_cells() -> None:
    rng = np.random.default_rng(3)
    n_cond, T, K = 30, 6, 4
    kinematics, endpoint = _synthetic_kinematics_endpoint(n_cond, T, K, rng)
    states = kinematics.copy()
    mask = np.ones((n_cond, T), dtype=bool)
    mask[:, -1] = False  # mask out last timestep

    res = two_stage_decode(states, kinematics, endpoint, valid_mask=mask, n_splits=3)
    assert np.isnan(res.direct_r[-1])
    assert np.isnan(res.kinematics_mediated_r[-1])


def test_shape_validation() -> None:
    rng = np.random.default_rng(4)
    with pytest.raises(ValueError, match="3-D states"):
        two_stage_decode(rng.normal(size=(4, 5)), rng.normal(size=(4, 5, 4)), np.zeros(4))
    with pytest.raises(ValueError, match="3-D kinematics"):
        two_stage_decode(rng.normal(size=(4, 5, 6)), rng.normal(size=(4, 5)), np.zeros(4))
    with pytest.raises(ValueError, match="endpoint_y shape"):
        two_stage_decode(
            rng.normal(size=(4, 5, 6)),
            rng.normal(size=(4, 5, 4)),
            np.zeros(3),
        )


@pytest.mark.skipif(not PILOT_RUN_DIR.exists(), reason="pilot run dir missing")
def test_pilot_smoke() -> None:
    """End-to-end on the M3 pilot.

    The IN's input includes ``(x, y, dx, dy)`` at every step, so direct and
    kinematics-mediated should both be very high (the IN's effect_receivers
    contain the kinematics linearly). The expected finding is that the gap
    is small — the IN's apparent rapid rise is largely explained by
    instantaneous kinematics. Test asserts only that the pipeline runs.
    """
    from dmfc.analysis.endpoint_decoding import flatten_receivers

    with np.load(PILOT_RUN_DIR / "hidden_states.npz") as data:
        receivers = data["effect_receivers"].astype(np.float64)
        targets = data["targets"].astype(np.float64)
        valid_mask = data["valid_mask"].astype(bool)

    states = flatten_receivers(receivers)
    endpoint_y = targets[:, 0, 6]
    T = states.shape[1]
    kinematics, kin_valid = kinematics_for_canonical_79(T_max=T)

    # Combine the two validity sources
    mask = valid_mask & kin_valid

    res = two_stage_decode(
        states=states,
        kinematics=kinematics,
        endpoint_y=endpoint_y,
        valid_mask=mask,
        n_splits=5,
    )
    # Average over finite bins
    direct_mean = np.nanmean(res.direct_r)
    kinmed_mean = np.nanmean(res.kinematics_mediated_r)
    kinonly_mean = np.nanmean(res.kinematics_only_r)
    assert np.isfinite(direct_mean) and direct_mean > 0.7
    assert np.isfinite(kinmed_mean) and kinmed_mean > 0.7
    assert np.isfinite(kinonly_mean) and kinonly_mean > 0.7
    # Per-axis state→kinematics: ball position columns should be very high
    s2k_pos = np.nanmean(res.state_to_kinematics_r[:, 0:2])
    assert s2k_pos > 0.7
