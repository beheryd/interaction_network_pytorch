"""Statistical tests for the M5 milestone (Fig. 5B + Fig. 4 hypotheses).

Per ``TASKS.md`` Milestone 5 the project requires four families of test:

* **Fig. 5B time-to-threshold-r** — per-model scalar (the time at which a
  decoding curve first crosses some Pearson r threshold), then a Wilcoxon
  rank-sum to compare the IN seed distribution to each RNN-class member
  distribution.
* **Fig. 5B RMSE-AUC** — per-model integrated RMSE over the 0–1200 ms
  window, again compared via Wilcoxon rank-sum.
* **Fig. 4 NC distributions** — IN swarm vs. each RNN class, Wilcoxon
  rank-sum.
* **Fig. 4 partial R²** — how much variance in Neural Consistency is
  explained by Simulation Index alone, vs. SI plus an indicator for whether
  the row is an IN. Measures whether including IN points changes the
  underlying NC ~ SI relationship.

This module is **pure numerics**. Functions take raw arrays and return
scalars or named tuples; the figure scripts and a future stats-driver
``run_m5_stats.py`` are responsible for shovelling data in and out.

Backed by ``scipy.stats.ranksums`` (Wilcoxon) and
``sklearn.linear_model.LinearRegression`` (partial R²). No statsmodels
dependency — sklearn covers the OLS we need.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats as scipy_stats
from sklearn.linear_model import LinearRegression


@dataclass(frozen=True)
class WilcoxonResult:
    """Output of :func:`wilcoxon_rank_sum`."""

    statistic: float  # Mann-Whitney U-style z-statistic from scipy.ranksums
    p_value: float
    effect_size_r: float  # |Z| / sqrt(n_a + n_b); standard rank-biserial proxy
    n_a: int
    n_b: int


def time_to_threshold(
    curves: np.ndarray,
    threshold: float,
    time_axis: np.ndarray,
) -> np.ndarray:
    """First time each curve crosses ``threshold``; NaN if it never does.

    Args:
        curves: shape ``(n_models, T)`` — per-model decoding-quality curve.
        threshold: scalar to compare each curve element against.
        time_axis: 1-D array of length ``T`` — the time grid in any unit
            (typically ms; the function is unit-agnostic).

    Returns:
        1-D array of length ``n_models``. Each entry is the time at which
        that model's curve first reaches or exceeds ``threshold``. NaN for
        curves that never cross.

    Notes:
        * If a curve is already at/above threshold at ``time_axis[0]``,
          returns ``time_axis[0]`` for that model.
        * NaN values inside ``curves`` are treated as never-crossing for
          the bins they occupy; a model with all-NaN curve gets NaN.
        * Equivalent to ``time_axis[argmax(curve >= threshold)]`` when a
          crossing exists; we explicitly guard the no-crossing case.
    """
    arr = np.asarray(curves, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    n_models, n_t = arr.shape
    if time_axis.shape != (n_t,):
        raise ValueError(f"time_axis shape {time_axis.shape} incompatible with curves {arr.shape}")

    out = np.full(n_models, np.nan)
    above = arr >= threshold  # NaN cells are False per IEEE 754
    has_crossing = above.any(axis=1)
    if has_crossing.any():
        first_idx = above.argmax(axis=1)
        out[has_crossing] = time_axis[first_idx[has_crossing]]
    return out


def rmse_auc(
    rmse_curves: np.ndarray,
    time_axis: np.ndarray,
    window_ms: tuple[float, float] | None = None,
) -> np.ndarray:
    """Per-model area under an RMSE curve over a time window.

    Args:
        rmse_curves: shape ``(n_models, T)`` — per-model RMSE-over-time.
        time_axis: 1-D array of length ``T`` (any unit).
        window_ms: optional ``(t_min, t_max)`` to restrict integration.
            If None, integrates over the full ``time_axis``.

    Returns:
        1-D array of length ``n_models``: trapezoidal AUC of the curve in
        the window. NaN cells inside ``rmse_curves`` are interpolated
        across only when finite values bracket them; an all-NaN row
        returns NaN.

    Note:
        Lower AUC = better RMSE accumulation = better model. The unit
        of the result is ``rmse_unit × time_unit``.
    """
    arr = np.asarray(rmse_curves, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    n_models, n_t = arr.shape
    t = np.asarray(time_axis, dtype=np.float64)
    if t.shape != (n_t,):
        raise ValueError(f"time_axis shape {t.shape} incompatible with curves {arr.shape}")

    if window_ms is not None:
        t_min, t_max = window_ms
        in_win = (t >= t_min) & (t <= t_max)
        if not in_win.any():
            return np.full(n_models, np.nan)
        t = t[in_win]
        arr = arr[:, in_win]

    out = np.full(n_models, np.nan)
    for i in range(n_models):
        finite = np.isfinite(arr[i])
        if finite.sum() < 2:
            continue
        # numpy 1.26 ships ``trapz``; ``trapezoid`` arrived in numpy 2.x. We
        # pin numpy 1.26.4 so the older spelling is correct here.
        out[i] = float(np.trapz(arr[i, finite], t[finite]))  # type: ignore[attr-defined]
    return out


def wilcoxon_rank_sum(
    group_a: np.ndarray,
    group_b: np.ndarray,
) -> WilcoxonResult:
    """Wilcoxon rank-sum (Mann-Whitney) test with rank-biserial effect size.

    Args:
        group_a, group_b: 1-D arrays. NaN entries are dropped from each
            group independently. Empty groups (after NaN drop) → NaN
            statistic and p_value.

    Returns:
        :class:`WilcoxonResult`. ``effect_size_r`` is the standard
        ``|Z| / sqrt(n_a + n_b)`` rank-biserial proxy — small/medium/large
        thresholds are conventionally 0.1 / 0.3 / 0.5.

    Note:
        Wraps ``scipy.stats.ranksums`` (the asymptotic z-version). For
        small n consider ``scipy.stats.mannwhitneyu`` with exact mode
        instead, but this function is what M5 calls for given the
        sweep's 5+ seeds per cell.
    """
    a = np.asarray(group_a, dtype=np.float64)
    b = np.asarray(group_b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    n_a = int(a.size)
    n_b = int(b.size)
    if n_a < 1 or n_b < 1:
        return WilcoxonResult(np.nan, np.nan, np.nan, n_a, n_b)
    res = scipy_stats.ranksums(a, b)
    z = float(res.statistic)
    p = float(res.pvalue)
    effect = abs(z) / np.sqrt(n_a + n_b) if (n_a + n_b) > 0 else np.nan
    return WilcoxonResult(z, p, float(effect), n_a, n_b)


def partial_r2(
    target: np.ndarray,
    base_predictors: np.ndarray,
    extra_predictors: np.ndarray,
) -> float:
    """Increase in R² from adding ``extra_predictors`` to a base OLS model.

    Computes ``R²_full - R²_base`` where:

    * ``R²_base`` fits ``target ~ base_predictors`` via OLS.
    * ``R²_full`` fits ``target ~ [base_predictors, extra_predictors]``.

    The result is non-negative (extra predictors never reduce R² in OLS)
    and bounded above by ``1 - R²_base`` — if the base model already
    explains everything, no extra predictor can help.

    Args:
        target: shape ``(n,)``.
        base_predictors: shape ``(n, k_base)``. May be a 1-D ``(n,)``
            single-column predictor; we reshape to ``(n, 1)``.
        extra_predictors: shape ``(n, k_extra)``. Same 1-D allowance.

    Returns:
        Scalar partial R².

    Notes:
        Equivalent to the *change* in R² when adding ``extra_predictors``
        to ``base_predictors``. The classical "partial R²" defined as
        ``(R²_full - R²_base) / (1 - R²_base)`` is *not* what we return —
        that's the proportion of *remaining* variance newly explained,
        which is a different question. M5 calls for the simpler version.
    """
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    x_base = np.asarray(base_predictors, dtype=np.float64)
    x_extra = np.asarray(extra_predictors, dtype=np.float64)
    if x_base.ndim == 1:
        x_base = x_base[:, None]
    if x_extra.ndim == 1:
        x_extra = x_extra[:, None]
    if x_base.shape[0] != y.shape[0] or x_extra.shape[0] != y.shape[0]:
        raise ValueError(
            "predictor row counts must match target length; got "
            f"y={y.shape}, base={x_base.shape}, extra={x_extra.shape}"
        )

    finite = np.isfinite(y) & np.isfinite(x_base).all(axis=1) & np.isfinite(x_extra).all(axis=1)
    if finite.sum() < x_base.shape[1] + x_extra.shape[1] + 2:
        return float("nan")

    y_f = y[finite]
    xb = x_base[finite]
    xfull = np.concatenate([xb, x_extra[finite]], axis=1)

    r2_base = LinearRegression().fit(xb, y_f).score(xb, y_f)
    r2_full = LinearRegression().fit(xfull, y_f).score(xfull, y_f)
    return float(r2_full - r2_base)


__all__: list[Any] = [
    "WilcoxonResult",
    "time_to_threshold",
    "rmse_auc",
    "wilcoxon_rank_sum",
    "partial_r2",
]
