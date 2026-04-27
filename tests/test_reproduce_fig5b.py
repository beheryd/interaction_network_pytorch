"""Tests for ``dmfc.analysis.reproduce_fig5b``.

The full pipeline depends on the Zenodo release; tests that need it skip
gracefully when ``data/dmfc`` is missing. Pure-numpy helpers are exercised
without the data files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dmfc.analysis.reproduce_fig5b import (
    RNN_CLASS_LABELS,
    RNN_CLASS_ORDER,
    CurveOnGrid,
    _expand_run_dirs,
    _interp_curve,
    common_time_axis,
)

PILOT_RUN_DIR = Path("runs/in_sim-mov_h10_s0_20260426-125840_fd614df-dirty")
DATA_DIR = Path("data/dmfc")


def test_common_time_axis_default() -> None:
    t = common_time_axis()
    assert t.shape == (100,)
    assert t[0] == 0.0
    assert t[1] == 50.0
    assert t[-1] == 4950.0


def test_common_time_axis_custom() -> None:
    t = common_time_axis(t_max_ms=2000, bin_ms=100)
    assert t.shape == (20,)
    assert t[-1] == 1900.0


def test_interp_extrapolates_to_nan() -> None:
    src = np.arange(0, 200, 50, dtype=np.float64)  # [0, 50, 100, 150]
    curve = np.array([0.0, 1.0, 2.0, 3.0])
    tgt = np.array([-50.0, 0.0, 75.0, 150.0, 250.0])
    out = _interp_curve(curve, src, tgt)
    assert np.isnan(out[0]) and np.isnan(out[-1])
    assert out[1] == pytest.approx(0.0)
    assert out[2] == pytest.approx(1.5)
    assert out[3] == pytest.approx(3.0)


def test_class_label_completeness() -> None:
    for k in RNN_CLASS_ORDER:
        assert k in RNN_CLASS_LABELS


def test_expand_run_dirs_filters_to_valid() -> None:
    if not PILOT_RUN_DIR.exists():
        pytest.skip("pilot run dir missing")
    found = _expand_run_dirs(["runs/in_*"])
    assert PILOT_RUN_DIR in found


def test_expand_run_dirs_skips_dirs_without_artifacts(tmp_path: Path) -> None:
    bad = tmp_path / "in_fake"
    bad.mkdir()
    found = _expand_run_dirs([str(bad)])
    assert found == []


@pytest.mark.skipif(not PILOT_RUN_DIR.exists(), reason="pilot run dir missing")
@pytest.mark.skipif(not DATA_DIR.exists(), reason="Zenodo data missing")
def test_full_pipeline_smoke(tmp_path: Path) -> None:
    """End-to-end: produce a PNG with all curves on the M3 pilot."""
    from dmfc.analysis.reproduce_fig5b import (
        dmfc_curve,
        in_curves,
        plot_fig5b,
        rnn_class_curves,
    )
    from dmfc.rajalingham.load import load_dmfc_neural, load_rnn_metrics

    t_axis = common_time_axis()
    dmfc_data = load_dmfc_neural(data_dir=DATA_DIR)
    rnn_data = load_rnn_metrics(data_dir=DATA_DIR)

    d = dmfc_curve(dmfc_data, t_axis, n_splits=5)
    rs = rnn_class_curves(rnn_data, t_axis)
    ins = in_curves([PILOT_RUN_DIR], t_axis)

    assert isinstance(d, CurveOnGrid)
    assert d.r_mean.shape == t_axis.shape
    assert len(rs) == 4
    assert len(ins) == 1

    out = tmp_path / "fig5b.png"
    plot_fig5b(d, rs, ins, out_path=out)
    assert out.exists()
    assert out.stat().st_size > 1000  # non-trivial PNG


@pytest.mark.skipif(not PILOT_RUN_DIR.exists(), reason="pilot run dir missing")
def test_in_curve_shape() -> None:
    from dmfc.analysis.reproduce_fig5b import in_curves

    t_axis = common_time_axis()
    curves = in_curves([PILOT_RUN_DIR], t_axis)
    assert len(curves) == 1
    assert curves[0].r_mean.shape == t_axis.shape
    # Pilot's sim-mov variant: r is near-perfect across the valid window
    # (M3 reported r ≈ 0.999 throughout). Generous lower bound here just
    # asserts the pipeline produced a high-r curve, not a chance-level one.
    finite = curves[0].r_mean[np.isfinite(curves[0].r_mean)]
    assert finite.size > 0
    assert finite.mean() > 0.7
