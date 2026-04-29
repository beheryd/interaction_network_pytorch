"""Tests for ``dmfc.analysis.stats`` — synthetic-input known-answer cases.

Each test pins a function's behavior on inputs whose answer is known a
priori, plus the obvious edge cases (NaN handling, degenerate sizes).
Comparisons against ``scipy.stats.ranksums`` and ``sklearn.LinearRegression``
are explicitly there in addition to handcrafted cases — together they
catch both math bugs and silent wrapping mistakes.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as scipy_stats
from sklearn.linear_model import LinearRegression

from dmfc.analysis.stats import (
    WilcoxonResult,
    partial_r2,
    rmse_auc,
    time_to_threshold,
    wilcoxon_rank_sum,
)

# ---------------------------------------------------------------------------
# time_to_threshold


def test_time_to_threshold_basic() -> None:
    time_axis = np.arange(0, 1000, 100, dtype=np.float64)  # 10 bins, 0..900 ms
    # Curve crosses 0.5 at bin 4 (i.e. t=400 ms).
    curve = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    out = time_to_threshold(curve, threshold=0.5, time_axis=time_axis)
    assert out.shape == (1,)
    assert out[0] == 400.0


def test_time_to_threshold_already_above() -> None:
    """Curve at/above threshold at t=0 returns t=0."""
    time_axis = np.arange(0, 500, 50, dtype=np.float64)
    curve = np.full(10, 0.9)  # always above 0.5
    out = time_to_threshold(curve, threshold=0.5, time_axis=time_axis)
    assert out[0] == 0.0


def test_time_to_threshold_never_crosses_returns_nan() -> None:
    time_axis = np.arange(0, 500, 50, dtype=np.float64)
    curve = np.full(10, 0.1)  # never above 0.5
    out = time_to_threshold(curve, threshold=0.5, time_axis=time_axis)
    assert np.isnan(out[0])


def test_time_to_threshold_multi_curve() -> None:
    time_axis = np.arange(0, 1000, 100, dtype=np.float64)
    # Three curves: one crosses at 200 ms, one at 600 ms, one never.
    curves = np.array(
        [
            [0.0, 0.0, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
            np.full(10, 0.0),
        ]
    )
    out = time_to_threshold(curves, threshold=0.5, time_axis=time_axis)
    assert out[0] == 200.0
    assert out[1] == 600.0
    assert np.isnan(out[2])


def test_time_to_threshold_handles_nan_in_curve() -> None:
    """NaN cells should not be treated as crossings (NaN >= threshold is False)."""
    time_axis = np.arange(0, 500, 100, dtype=np.float64)
    curve = np.array([np.nan, np.nan, 0.6, 0.7, 0.8])
    out = time_to_threshold(curve, threshold=0.5, time_axis=time_axis)
    assert out[0] == 200.0  # crosses at index 2


def test_time_to_threshold_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="time_axis"):
        time_to_threshold(np.zeros((2, 5)), 0.5, np.arange(3))


# ---------------------------------------------------------------------------
# rmse_auc


def test_rmse_auc_constant_curve_full_range() -> None:
    """Constant c over [t_min, t_max] → AUC = c × (t_max - t_min)."""
    time_axis = np.linspace(0, 1000, 11)  # 0..1000 ms in 100 ms steps
    curves = np.full((1, 11), 2.0)
    out = rmse_auc(curves, time_axis)
    assert out[0] == pytest.approx(2.0 * 1000.0, rel=1e-12)


def test_rmse_auc_linear_curve_half_rect() -> None:
    """Linear ramp 0..c over T → AUC = 0.5 × c × T."""
    time_axis = np.linspace(0, 1000, 101)
    curves = np.linspace(0, 4, 101)[None, :]
    out = rmse_auc(curves, time_axis)
    assert out[0] == pytest.approx(0.5 * 4 * 1000, rel=1e-3)


def test_rmse_auc_window_restriction() -> None:
    """AUC over a sub-window is the AUC of that sub-window only."""
    time_axis = np.linspace(0, 1000, 1001)  # 1 ms resolution
    # Curve = 1 always; AUC over [200, 500] should be ~300.
    curves = np.ones((1, 1001))
    out = rmse_auc(curves, time_axis, window_ms=(200.0, 500.0))
    assert out[0] == pytest.approx(300.0, rel=1e-3)


def test_rmse_auc_window_outside_returns_nan() -> None:
    time_axis = np.linspace(0, 1000, 11)
    curves = np.ones((2, 11))
    out = rmse_auc(curves, time_axis, window_ms=(2000.0, 3000.0))
    assert np.isnan(out).all()


def test_rmse_auc_all_nan_row_returns_nan() -> None:
    time_axis = np.linspace(0, 1000, 11)
    curves = np.full((2, 11), np.nan)
    curves[0] = 1.0  # one row finite, one NaN
    out = rmse_auc(curves, time_axis)
    assert np.isfinite(out[0])
    assert np.isnan(out[1])


# ---------------------------------------------------------------------------
# wilcoxon_rank_sum


def test_wilcoxon_identical_samples_high_p() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, size=50)
    res = wilcoxon_rank_sum(a, a.copy())
    assert isinstance(res, WilcoxonResult)
    assert res.p_value > 0.5
    assert abs(res.statistic) < 0.5  # ~0 by symmetry


def test_wilcoxon_strongly_separated_low_p() -> None:
    """Disjoint distributions should give p << 0.001."""
    a = np.arange(1, 21, dtype=float)
    b = np.arange(101, 121, dtype=float)
    res = wilcoxon_rank_sum(a, b)
    assert res.p_value < 1e-3
    assert res.effect_size_r > 0.5  # large effect


def test_wilcoxon_matches_scipy_directly() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, size=30)
    b = rng.normal(0.5, 1, size=30)
    res = wilcoxon_rank_sum(a, b)
    ref = scipy_stats.ranksums(a, b)
    assert res.statistic == pytest.approx(float(ref.statistic), rel=1e-12)
    assert res.p_value == pytest.approx(float(ref.pvalue), rel=1e-12)
    assert res.n_a == 30
    assert res.n_b == 30


def test_wilcoxon_drops_nans_independently() -> None:
    a = np.array([1.0, 2.0, np.nan, 3.0])
    b = np.array([10.0, np.nan, 20.0])
    res = wilcoxon_rank_sum(a, b)
    assert res.n_a == 3
    assert res.n_b == 2


def test_wilcoxon_empty_returns_nans() -> None:
    res = wilcoxon_rank_sum(np.array([]), np.array([1.0, 2.0]))
    assert np.isnan(res.statistic)
    assert np.isnan(res.p_value)
    assert res.n_a == 0
    assert res.n_b == 2


# ---------------------------------------------------------------------------
# partial_r2


def test_partial_r2_useless_extra_predictor_zero() -> None:
    """If extra is uncorrelated noise, partial R² should be near 0."""
    rng = np.random.default_rng(0)
    x_base = rng.normal(0, 1, size=200)
    y = 3.0 * x_base + rng.normal(0, 0.1, size=200)
    x_extra = rng.normal(0, 1, size=200)
    p = partial_r2(y, x_base, x_extra)
    assert -0.05 < p < 0.05


def test_partial_r2_perfect_extra_predictor_large() -> None:
    """If y depends on extra, adding it must explain a lot more variance."""
    rng = np.random.default_rng(1)
    x_base = rng.normal(0, 1, size=200)
    x_extra = rng.normal(0, 1, size=200)
    # y depends entirely on x_extra; x_base is unrelated.
    y = 3.0 * x_extra + rng.normal(0, 0.1, size=200)
    p = partial_r2(y, x_base, x_extra)
    assert p > 0.9


def test_partial_r2_matches_manual_sklearn() -> None:
    rng = np.random.default_rng(2)
    n = 100
    x_base = rng.normal(0, 1, size=(n, 2))
    x_extra = rng.normal(0, 1, size=(n, 1))
    y = x_base @ np.array([1.0, -2.0]) + 0.3 * x_extra[:, 0] + rng.normal(0, 0.5, size=n)

    base_model = LinearRegression().fit(x_base, y)
    full_model = LinearRegression().fit(np.concatenate([x_base, x_extra], axis=1), y)
    expected = full_model.score(np.concatenate([x_base, x_extra], axis=1), y) - base_model.score(
        x_base, y
    )
    assert partial_r2(y, x_base, x_extra) == pytest.approx(expected, rel=1e-9)


def test_partial_r2_too_few_samples_returns_nan() -> None:
    """Need at least k_total+2 finite rows; we'll give too few."""
    y = np.array([1.0, 2.0])
    x_base = np.array([[1.0, 0.0], [0.0, 1.0]])
    x_extra = np.array([[1.0], [1.0]])  # 2 samples, 3 predictors → underdetermined
    out = partial_r2(y, x_base, x_extra)
    assert np.isnan(out)


def test_partial_r2_handles_nan_rows() -> None:
    """Rows with any NaN should be dropped before fitting."""
    rng = np.random.default_rng(3)
    n = 200
    x_base = rng.normal(0, 1, size=n)
    x_extra = rng.normal(0, 1, size=n)
    y = x_base + 2.0 * x_extra + rng.normal(0, 0.1, size=n)
    # Inject some NaNs.
    y[5] = np.nan
    x_base[10] = np.nan
    x_extra[15] = np.nan
    out = partial_r2(y, x_base, x_extra)
    assert np.isfinite(out)
    assert out > 0.5  # most variance from x_extra
