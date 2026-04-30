"""Tests for ``dmfc.analysis.run_m5_stats`` — the permutation + report bits.

We don't exercise the DMFC- or IN-loading paths here (those need the
Zenodo data + run dirs to be present). We do test the pure-numerics
helpers: per-timestep two-sided p, first-t-below-alpha, and the text
formatter.
"""

from __future__ import annotations

import json

import numpy as np

from dmfc.analysis.run_m5_stats import (
    Fig4R2Result,
    Fig5BPermResult,
    M5StatsBundle,
    _first_t_persistently_below,
    _two_sided_p,
    format_text_report,
    write_outputs,
)


def test_two_sided_p_strong_signal() -> None:
    rng = np.random.default_rng(0)
    real = np.array([0.95, 0.95, 0.95])
    perm = rng.uniform(-0.2, 0.2, size=(2000, 3))
    p = _two_sided_p(real, perm, p_floor=1.0 / 2001)
    # No permutation should beat |0.95| in [-0.2, 0.2]; p hits the floor.
    assert np.allclose(p, 1.0 / 2001)


def test_two_sided_p_no_signal() -> None:
    rng = np.random.default_rng(1)
    perm = rng.standard_normal(size=(1000, 4))
    real = np.zeros(4)
    p = _two_sided_p(real, perm, p_floor=1.0 / 1001)
    # |0| is the smallest possible |stat|, so c = 1000 → p ≈ 1.
    assert np.all(p > 0.99)


def test_two_sided_p_handles_nan() -> None:
    real = np.array([0.5, np.nan])
    perm = np.array([[0.0, np.nan], [-0.1, np.nan]])
    p = _two_sided_p(real, perm, p_floor=1e-3)
    assert np.isfinite(p[0])
    assert np.isnan(p[1])


def test_first_t_persistently_below_basic() -> None:
    p = np.array([0.5, 0.2, 1e-5, 1e-6, 1e-7])
    t_ms = np.array([0.0, 100.0, 200.0, 300.0, 400.0])
    # First index from which p stays < 1e-4 for the rest of the array is t=200.
    assert _first_t_persistently_below(p, t_ms, alpha=1e-4) == 200.0


def test_first_t_persistently_below_breaks() -> None:
    p = np.array([1e-5, 0.5, 1e-6])  # not persistent at t=0
    t_ms = np.array([0.0, 100.0, 200.0])
    # Only index 2 has p<alpha that stays for the rest.
    assert _first_t_persistently_below(p, t_ms, alpha=1e-4) == 200.0


def test_first_t_persistently_below_never() -> None:
    p = np.array([0.5, 0.6, 0.7])
    t_ms = np.array([0.0, 100.0, 200.0])
    assert not np.isfinite(_first_t_persistently_below(p, t_ms, alpha=1e-4))


def test_format_and_write_outputs(tmp_path) -> None:
    f5 = Fig5BPermResult(
        t_ms=np.array([0.0, 50.0, 100.0]),
        r_real=np.array([0.1, 0.5, 0.9]),
        rmse_real=np.array([5.0, 3.0, 1.0]),
        p_two_sided=np.array([0.5, 0.05, 1e-5]),
        first_t_below_alpha=100.0,
        alpha=1e-4,
        n_permutations=100,
        p_floor=1.0 / 101,
    )
    f4 = Fig4R2Result(
        n_in_runs=40,
        raw_r2_nc_from_si=0.7,
        partial_r2_nc_from_si_after_mae=0.5,
        raw_r2_nc_from_mae=0.2,
        partial_r2_nc_from_mae_after_si=0.0,
        raw_r2_nc_from_both=0.7,
    )
    bundle = M5StatsBundle(fig5b=f5, fig4=f4, in_run_dirs=["/path/a", "/path/b"])

    text = format_text_report(bundle)
    assert "Fig. 5B" in text
    assert "Fig. 4" in text
    assert "100 ms" in text  # the first-t summary line
    assert "0.7000" in text  # raw R² printed at 4dp

    txt_path, json_path = write_outputs(bundle, out_dir=tmp_path)
    assert txt_path.read_text() == text

    payload = json.loads(json_path.read_text())
    assert payload["fig5b"]["n_permutations"] == 100
    assert payload["fig5b"]["first_t_below_alpha_ms"] == 100.0
    assert payload["fig4"]["n_in_runs"] == 40
    assert payload["in_run_dirs"] == ["/path/a", "/path/b"]
