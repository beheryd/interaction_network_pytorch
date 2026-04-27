"""Tests for ``dmfc.analysis.simulation_index``."""

from __future__ import annotations

import numpy as np
import pytest

from dmfc.analysis.simulation_index import SimulationIndexResult, simulation_index


def _full_mask(n_cond: int, T: int) -> np.ndarray:
    return np.ones((n_cond, T), dtype=bool)


def test_perfect_linear_signal_zero_mae() -> None:
    """If ball_xy is a linear function of states, decoder recovers it exactly."""
    rng = np.random.default_rng(0)
    n_cond, T, n_feat = 20, 10, 8
    states = rng.normal(size=(n_cond, T, n_feat))
    W = rng.normal(size=(n_feat, 2))
    b = rng.normal(size=(2,))
    ball_xy = states @ W + b

    res = simulation_index(
        states=states,
        ball_xy=ball_xy,
        train_mask=_full_mask(n_cond, T),
        test_mask=_full_mask(n_cond, T),
        k=2,
    )
    assert isinstance(res, SimulationIndexResult)
    np.testing.assert_allclose(res.mae, 0.0, atol=1e-9)
    np.testing.assert_allclose(res.rho, 1.0, atol=1e-9)
    assert res.si == pytest.approx(0.0, abs=1e-9)


def test_pure_noise_chance_level() -> None:
    """States independent of ball_xy ⇒ rho ≈ 0, MAE ≈ scale of ball_xy."""
    rng = np.random.default_rng(1)
    n_cond, T, n_feat = 40, 6, 5
    states = rng.normal(size=(n_cond, T, n_feat))
    ball_xy = rng.normal(size=(n_cond, T, 2)) * 3.0  # std ≈ 3 per coord

    res = simulation_index(
        states=states,
        ball_xy=ball_xy,
        train_mask=_full_mask(n_cond, T),
        test_mask=_full_mask(n_cond, T),
        k=2,
    )
    # Decoder learns to predict the mean; on held-out conditions rho ~ 0.
    assert (np.abs(res.rho) < 0.4).all()
    # MAE roughly std of target (mean prediction error).
    assert (res.mae > 1.0).all()


def test_seed_determinism() -> None:
    rng = np.random.default_rng(2)
    n_cond, T, n_feat = 20, 5, 6
    states = rng.normal(size=(n_cond, T, n_feat))
    ball_xy = rng.normal(size=(n_cond, T, 2))

    a = simulation_index(states, ball_xy, _full_mask(n_cond, T), _full_mask(n_cond, T), k=2, seed=7)
    b = simulation_index(states, ball_xy, _full_mask(n_cond, T), _full_mask(n_cond, T), k=2, seed=7)
    np.testing.assert_array_equal(a.mae, b.mae)
    np.testing.assert_array_equal(a.rho, b.rho)


def test_different_seeds_different_folds() -> None:
    rng = np.random.default_rng(3)
    n_cond, T, n_feat = 20, 5, 6
    states = rng.normal(size=(n_cond, T, n_feat))
    ball_xy = rng.normal(size=(n_cond, T, 2))

    a = simulation_index(states, ball_xy, _full_mask(n_cond, T), _full_mask(n_cond, T), k=2, seed=7)
    b = simulation_index(states, ball_xy, _full_mask(n_cond, T), _full_mask(n_cond, T), k=2, seed=8)
    # Different folds → different MAEs (overwhelming probability with rng).
    assert not np.allclose(a.mae, b.mae)


def test_test_mask_subset_of_train_mask() -> None:
    """Realistic Rajalingham setup: train on vis+occ epoch, test on occ only."""
    rng = np.random.default_rng(4)
    n_cond, T, n_feat = 20, 8, 6
    states = rng.normal(size=(n_cond, T, n_feat))
    W = rng.normal(size=(n_feat, 2))
    ball_xy = states @ W

    train = _full_mask(n_cond, T)
    test = np.zeros((n_cond, T), dtype=bool)
    test[:, T // 2 :] = True  # occluded second half

    res = simulation_index(states, ball_xy, train, test, k=2)
    np.testing.assert_allclose(res.mae, 0.0, atol=1e-9)
    assert res.n_test_cells == n_cond * (T - T // 2)


def test_partial_test_mask_per_condition() -> None:
    """Test mask varies per condition (variable-length occluded epochs)."""
    rng = np.random.default_rng(5)
    n_cond, T, n_feat = 12, 10, 4
    states = rng.normal(size=(n_cond, T, n_feat))
    W = rng.normal(size=(n_feat, 2))
    ball_xy = states @ W

    train = _full_mask(n_cond, T)
    test = np.zeros((n_cond, T), dtype=bool)
    for c in range(n_cond):
        start = 4 + (c % 3)  # variable occlusion onset
        test[c, start:] = True

    res = simulation_index(states, ball_xy, train, test, k=3)
    np.testing.assert_allclose(res.mae, 0.0, atol=1e-9)
    assert res.n_test_cells == int(test.sum())


def test_k_out_of_range_raises() -> None:
    rng = np.random.default_rng(6)
    states = rng.normal(size=(3, 5, 4))
    ball_xy = rng.normal(size=(3, 5, 2))
    with pytest.raises(ValueError, match="out of range"):
        simulation_index(states, ball_xy, _full_mask(3, 5), _full_mask(3, 5), k=10)
    with pytest.raises(ValueError, match="out of range"):
        simulation_index(states, ball_xy, _full_mask(3, 5), _full_mask(3, 5), k=1)


def test_states_shape_validation() -> None:
    rng = np.random.default_rng(7)
    with pytest.raises(ValueError, match="3-D states"):
        simulation_index(
            rng.normal(size=(4, 5)),
            rng.normal(size=(4, 5, 2)),
            _full_mask(4, 5),
            _full_mask(4, 5),
        )


def test_ball_xy_shape_validation() -> None:
    rng = np.random.default_rng(8)
    with pytest.raises(ValueError, match="3-D ball_xy"):
        simulation_index(
            rng.normal(size=(4, 5, 6)),
            rng.normal(size=(4, 5)),
            _full_mask(4, 5),
            _full_mask(4, 5),
        )


def test_dim_mismatch_raises() -> None:
    rng = np.random.default_rng(9)
    with pytest.raises(ValueError, match="train_mask shape"):
        simulation_index(
            rng.normal(size=(4, 5, 6)),
            rng.normal(size=(4, 5, 2)),
            _full_mask(4, 6),
            _full_mask(4, 5),
        )


def test_pilot_smoke() -> None:
    """End-to-end on the M3 pilot artifact, if present."""
    pilot = "runs/in_sim-mov_h10_s0_20260426-125840_fd614df-dirty/hidden_states.npz"
    import os

    if not os.path.exists(pilot):
        pytest.skip(f"pilot artifact missing: {pilot}")
    from dmfc.analysis.endpoint_decoding import flatten_receivers

    with np.load(pilot) as data:
        receivers = data["effect_receivers"]
        targets = data["targets"]
        valid_mask = data["valid_mask"].astype(bool)
        visible_mask = data["visible_mask"].astype(bool)

    states = flatten_receivers(receivers.astype(np.float64))
    ball_xy = targets[:, :, 0:2].astype(np.float64)  # output[0:2] hold true ball position
    train = valid_mask
    test = valid_mask & ~visible_mask  # occluded epoch only

    res = simulation_index(states, ball_xy, train, test, k=2, seed=0)
    # The sim-mov pilot was trained explicitly to predict occluded ball position
    # (R²=0.985 on output[4], 0.989 on output[5] per the M3 closeout). MAE in
    # degrees should be in the low single digits — generous bound here just
    # asserts the pipeline ran end-to-end.
    assert res.si < 5.0
    assert (res.rho > 0.5).all()
