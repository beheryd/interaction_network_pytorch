"""Smoke + structural tests for ``dmfc.analysis.reproduce_fig4``.

The heavy parts of the pipeline (NC RDM construction over ~1k cells,
SI KFold OLS) live in already-tested modules; this file checks the
glue: loader column wiring, IN-point shapes, RNN swarm filtering, plot
output, and a CLI smoke run on the four pilot runs when their data
is available.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dmfc.analysis.reproduce_fig4 import (
    NC_COLUMN,
    INPoint,
    in_point_for_run,
    plot_fig4,
    rnn_swarm,
)
from dmfc.rajalingham.load import (
    DEFAULT_DATA_DIR,
    DMFC_NEURAL_PKL,
    RNN_COMPARE_PKL,
    RNN_METRICS_PKL,
    load_dmfc_neural,
    load_rnn_compare,
    load_rnn_metrics,
)


def _have(filename: str) -> bool:
    return (DEFAULT_DATA_DIR / filename).exists()


def _have_pilot_runs() -> list[Path]:
    """Return convergent IN runs (excludes the known-diverged ``mov`` seed-0 pilot).

    SCRATCHPAD M4 progress 5 documents the divergence: the Intercept loss
    variant has a recurrent-stability failure mode under default lr/clip that
    produces all-NaN states by step ~800 on seed=0. Those run dirs still exist
    on disk but their ``hidden_states.npz`` cannot drive the analysis pipeline
    until M5 lands the Intercept-specific HP fix.
    """
    runs_root = Path("runs")
    if not runs_root.exists():
        return []
    out: list[Path] = []
    for p in sorted(runs_root.glob("in_*")):
        npz = p / "hidden_states.npz"
        if not npz.exists():
            continue
        with np.load(npz) as data:
            if np.isnan(np.asarray(data["effect_receivers"])).any():
                continue
        out.append(p)
    return out


@pytest.mark.skipif(not _have(RNN_COMPARE_PKL), reason="rnn_compare pickle not available")
def test_load_rnn_compare_has_published_nc() -> None:
    df = load_rnn_compare()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 192
    assert NC_COLUMN in df.columns
    # Smoke values: NC for an arbitrary published RNN should be a finite scalar
    # within a sensible Pearson-r-like range (allowing SB-corrected overshoot).
    val = float(df.loc[0, NC_COLUMN])
    assert np.isfinite(val)
    assert -1.5 <= val <= 1.5


@pytest.mark.skipif(
    not (_have(RNN_METRICS_PKL) and _have(RNN_COMPARE_PKL)),
    reason="RNN pickles not available",
)
def test_rnn_swarm_filters_to_gabor_pca() -> None:
    metrics = load_rnn_metrics()
    compare = load_rnn_compare()
    df = rnn_swarm(metrics.df, compare, input_representation="gabor_pca")
    assert {"loss_class", "nc", "si"}.issubset(set(df.columns))
    # 96 of 192 RNN models use gabor_pca features.
    assert len(df) == 96
    # Loss classes present should be a subset of the four expected.
    assert set(df["loss_class"].unique()) <= {"mov", "vis-mov", "vis-sim-mov", "sim-mov"}
    assert df["nc"].notna().all()
    assert df["si"].notna().all()


def test_rnn_swarm_raises_on_missing_columns() -> None:
    df_off = pd.DataFrame(
        {"filename": ["a"], "loss_weight_type": ["mov"], "input_representation": ["gabor_pca"]}
    )
    df_cmp = pd.DataFrame({"filename": ["a"]})
    with pytest.raises(KeyError, match="SI"):
        rnn_swarm(df_off, df_cmp)


@pytest.mark.skipif(
    not (_have(DMFC_NEURAL_PKL) and _have_pilot_runs()),
    reason="DMFC data or pilot runs not available",
)
def test_in_point_for_run_returns_finite_metrics() -> None:
    runs = _have_pilot_runs()
    dmfc = load_dmfc_neural()
    p = in_point_for_run(runs[0], dmfc)
    assert isinstance(p, INPoint)
    assert p.loss_class in {"mov", "vis-mov", "vis-sim-mov", "sim-mov"}
    assert np.isfinite(p.nc)
    assert np.isfinite(p.si)
    # SI is MAE in degrees; should be plausibly small for a converged pilot.
    assert 0.0 <= p.si < 10.0


def test_plot_fig4_writes_png(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    rnn_df = pd.DataFrame(
        {
            "loss_class": ["mov", "mov", "vis-mov", "sim-mov"] * 3,
            "nc": rng.uniform(0, 0.5, 12),
            "si": rng.uniform(1, 4, 12),
        }
    )
    ins = [
        INPoint(
            loss_class="vis-mov",
            nc=0.30,
            si=1.5,
            label="IN: Vis (h10 s0)",
            n_hidden=10,
            seed=0,
            run_dir=tmp_path,
        )
    ]
    out = tmp_path / "fig4_smoke.png"
    plot_fig4(rnn_df, ins, out_path=out)
    assert out.exists()
    assert out.stat().st_size > 1024  # non-trivially sized PNG
