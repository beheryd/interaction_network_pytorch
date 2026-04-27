"""Numerical tests for ``dmfc.analysis.endpoint_decoding``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dmfc.analysis.endpoint_decoding import (
    DecodingResult,
    decode_endpoint,
    flatten_receivers,
    load_pilot_states,
)

PILOT_RUN = Path("runs/in_sim-mov_h10_s0_20260426-125840_fd614df-dirty")


def test_flatten_receivers_shape_and_order() -> None:
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((5, 7, 2, 3))
    flat = flatten_receivers(arr)
    assert flat.shape == (5, 7, 6)
    np.testing.assert_array_equal(flat[2, 3, :3], arr[2, 3, 0, :])
    np.testing.assert_array_equal(flat[2, 3, 3:], arr[2, 3, 1, :])


def test_flatten_receivers_rejects_wrong_ndim() -> None:
    with pytest.raises(ValueError):
        flatten_receivers(np.zeros((4, 5, 6)))


def test_decode_perfect_signal_gives_r_one() -> None:
    """If states are exactly the endpoint replicated across time, r → 1, RMSE → 0."""
    rng = np.random.default_rng(0)
    n_cond, T, n_feat = 30, 8, 4
    endpoint_y = rng.standard_normal(n_cond)
    # Each feature is endpoint_y plus a tiny per-feature offset → linearly invertible.
    states = np.broadcast_to(endpoint_y[:, None, None], (n_cond, T, n_feat)).copy()
    states = states + rng.normal(scale=1e-6, size=states.shape)

    res = decode_endpoint(states, endpoint_y, n_splits=5)
    assert isinstance(res, DecodingResult)
    assert res.r.shape == (T,)
    assert res.rmse.shape == (T,)
    np.testing.assert_allclose(res.r, np.ones(T), atol=1e-3)
    assert np.all(res.rmse < 1e-3)


def test_decode_pure_noise_gives_low_r() -> None:
    """Pure-noise features → out-of-fold r near zero, RMSE near std(endpoint_y)."""
    rng = np.random.default_rng(1)
    n_cond, T, n_feat = 79, 10, 8
    endpoint_y = rng.standard_normal(n_cond)
    states = rng.standard_normal((n_cond, T, n_feat))

    res = decode_endpoint(states, endpoint_y, n_splits=5)
    # 79 samples / 5 folds + 8 noisy features → modest |r|, but well below ~0.4.
    assert np.all(np.abs(res.r) < 0.4)
    target_rmse = float(np.std(endpoint_y))
    assert np.all(res.rmse > 0.5 * target_rmse)


def test_decode_shapes_match_inputs() -> None:
    rng = np.random.default_rng(2)
    states = rng.standard_normal((79, 72, 20))
    endpoint_y = rng.standard_normal(79)
    res = decode_endpoint(states, endpoint_y, n_splits=5)
    assert res.n_conditions == 79
    assert res.n_timesteps == 72
    assert res.n_features == 20
    assert res.r_per_fold.shape == (5, 72)
    assert res.rmse_per_fold.shape == (5, 72)


def test_decode_is_deterministic() -> None:
    rng = np.random.default_rng(3)
    states = rng.standard_normal((40, 6, 5))
    endpoint_y = rng.standard_normal(40)
    a = decode_endpoint(states, endpoint_y, n_splits=4)
    b = decode_endpoint(states, endpoint_y, n_splits=4)
    np.testing.assert_array_equal(a.r, b.r)
    np.testing.assert_array_equal(a.rmse, b.rmse)


def test_valid_mask_skips_invalid_timesteps() -> None:
    """If a (cond, t) is masked invalid, that condition shouldn't contribute at t."""
    rng = np.random.default_rng(4)
    n_cond, T, n_feat = 40, 6, 4
    endpoint_y = rng.standard_normal(n_cond)
    states = np.broadcast_to(endpoint_y[:, None, None], (n_cond, T, n_feat)).copy()

    # Corrupt the last 2 timesteps for everyone but mask them out.
    states[:, -2:, :] = rng.standard_normal((n_cond, 2, n_feat)) * 100.0
    valid_mask = np.ones((n_cond, T), dtype=np.float64)
    valid_mask[:, -2:] = 0.0

    res = decode_endpoint(states, endpoint_y, valid_mask=valid_mask, n_splits=5)
    np.testing.assert_allclose(res.r[:-2], np.ones(T - 2), atol=1e-3)
    # With <2 valid points at the masked timesteps, NaN by design.
    assert np.all(np.isnan(res.r[-2:]))


def test_decode_input_validation() -> None:
    with pytest.raises(ValueError):
        decode_endpoint(np.zeros((5, 6)), np.zeros(5))
    with pytest.raises(ValueError):
        decode_endpoint(np.zeros((5, 6, 3)), np.zeros((5, 1)))
    with pytest.raises(ValueError):
        decode_endpoint(np.zeros((5, 6, 3)), np.zeros(4))
    with pytest.raises(ValueError):
        decode_endpoint(
            np.zeros((5, 6, 3)),
            np.zeros(5),
            valid_mask=np.zeros((5, 7)),
        )


@pytest.mark.skipif(not (PILOT_RUN / "hidden_states.npz").exists(), reason="pilot run not present")
def test_pilot_smoke_endpoint_decodable_throughout() -> None:
    """The pilot was supervised on the endpoint; effect_receivers encode it from t=0.

    SCRATCHPAD M3 closeout flagged that supervised-output Pearson r ≈ 0.999
    from t=0 onward. Decoding from effect_receivers should look similar — the
    state already reflects the deterministic kinematics. This test verifies
    the decoder returns finite, high r values across all valid timesteps; it
    does NOT assert a learning curve, because there isn't one to detect with
    this loss variant.
    """
    states, endpoint_y, valid_mask = load_pilot_states(PILOT_RUN)
    assert states.shape == (79, 72, 20)
    assert endpoint_y.shape == (79,)
    assert valid_mask.shape == (79, 72)

    res = decode_endpoint(states, endpoint_y, valid_mask=valid_mask, n_splits=5)
    # Take the band of timesteps that's valid for every condition (max t_occ ≈ 50).
    early = float(np.nanmean(res.r[:5]))
    mid = float(np.nanmean(res.r[20:40]))
    assert early > 0.9, f"pilot endpoint should be decodable from t=0; got early={early:.3f}"
    assert mid > 0.9, f"pilot endpoint decodability should hold mid-trial; got mid={mid:.3f}"
    assert np.isfinite(res.r[0]) and np.isfinite(res.rmse[0])
