"""Loader smoke tests for ``dmfc.rajalingham.load``.

Each test skips if the Zenodo file isn't reachable through ``data/dmfc/`` —
the loaders are a thin layer over disk and there's nothing useful to assert
without the real bytes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dmfc.rajalingham.load import (
    DECODE_PKL,
    DEFAULT_DATA_DIR,
    DMFC_BIN_MS,
    DMFC_N_TIMESTEPS,
    DMFC_NEURAL_PKL,
    N_CONDITIONS,
    RNN_METRICS_PKL,
    load_decode_dmfc,
    load_dmfc_neural,
    load_rnn_metrics,
)


def _have(filename: str) -> bool:
    return (DEFAULT_DATA_DIR / filename).exists()


@pytest.mark.skipif(not _have(DMFC_NEURAL_PKL), reason="DMFC dataset not available")
def test_load_dmfc_neural_shapes() -> None:
    data = load_dmfc_neural()
    assert data.responses.ndim == 3
    assert data.responses.shape[1:] == (N_CONDITIONS, DMFC_N_TIMESTEPS)
    assert data.responses_sh1.shape == data.responses.shape
    assert data.responses_sh2.shape == data.responses.shape
    assert data.bin_ms == DMFC_BIN_MS
    assert data.n_timesteps == DMFC_N_TIMESTEPS
    assert data.epoch == "occ"
    assert data.meta_index.shape == (N_CONDITIONS,)
    assert data.meta_index.dtype == np.int64

    # Canonical masks should round-trip with the right shape.
    for name in ("start_end_pad0", "start_occ_pad0", "occ_end_pad0", "f_pad0"):
        assert name in data.masks, f"missing mask {name}"
        assert data.masks[name].shape == (N_CONDITIONS, DMFC_N_TIMESTEPS)

    # Behavioral targets we'll need downstream.
    for name in ("ball_pos_x", "ball_pos_y", "ball_final_y", "paddle_pos_y", "t_from_occ"):
        assert name in data.behavioral
        assert data.behavioral[name].shape == (N_CONDITIONS, DMFC_N_TIMESTEPS)


@pytest.mark.skipif(not _have(DMFC_NEURAL_PKL), reason="DMFC dataset not available")
def test_load_dmfc_neural_extra_masks() -> None:
    data = load_dmfc_neural(extra_masks=("occ_end_pad0_roll0",))
    assert "occ_end_pad0_roll0" in data.masks
    assert data.masks["occ_end_pad0_roll0"].shape == (N_CONDITIONS, DMFC_N_TIMESTEPS)


@pytest.mark.skipif(not _have(RNN_METRICS_PKL), reason="RNN metrics file not available")
def test_load_rnn_metrics_shapes() -> None:
    metrics = load_rnn_metrics()
    assert isinstance(metrics.df, pd.DataFrame)
    assert len(metrics.df) == 192
    assert len(metrics.per_model) == 192
    assert metrics.n_iterations == 100
    assert metrics.n_timesteps == 90

    sample = next(iter(metrics.per_model.values()))
    assert sample["r_start1_all"].shape == (100, 90)
    assert sample["mae_start1_all"].shape == (100, 90)
    assert sample["yp"].shape == (100, N_CONDITIONS, 90)
    assert sample["yt"].shape == (100, N_CONDITIONS, 90)


@pytest.mark.skipif(not _have(DECODE_PKL), reason="Decode results file not available")
def test_load_decode_dmfc_shapes() -> None:
    res = load_decode_dmfc()
    assert len(res.beh_targets) == 23
    assert "ball_pos_y_TRUE" in res.beh_targets
    assert res.n_conditions == N_CONDITIONS
    assert isinstance(res.entries, list) and len(res.entries) == 11

    first = res.entries[0]
    assert first["r_mu"].shape == (23,)
    assert first["mae_mu"].shape == (23,)
    assert first["r_dist"].shape == (100, 23)
    assert first["mae_dist"].shape == (100, 23)

    assert "groupby" in res.decoder_specs
    assert res.decoder_specs["groupby"] == "condition"
    assert res.neural_data_key == "neural_responses_reliable_FactorAnalysis_50"


def test_file_not_found_raises_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=DMFC_NEURAL_PKL):
        load_dmfc_neural(data_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match=RNN_METRICS_PKL):
        load_rnn_metrics(data_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="decode_"):
        load_decode_dmfc(data_dir=tmp_path)
