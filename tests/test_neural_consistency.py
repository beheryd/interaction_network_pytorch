"""Tests for ``dmfc.analysis.neural_consistency``."""

from __future__ import annotations

import numpy as np
import pytest

from dmfc.analysis.neural_consistency import (
    ConsistencyResult,
    _sb,
    neural_consistency,
    neural_consistency_from_states,
)


def test_sb_correction_unit_case() -> None:
    """SB(1) = 1; SB(0) = 0."""
    assert _sb(1.0) == pytest.approx(1.0)
    assert _sb(0.0) == pytest.approx(0.0)
    assert np.isnan(_sb(-1.0))
    assert np.isnan(_sb(np.nan))


def test_sb_correction_known_value() -> None:
    """SB(0.5) = 2*0.5/(1+0.5) = 0.6667."""
    assert _sb(0.5) == pytest.approx(2.0 / 3.0)


def test_identical_rdms_perfect_consistency() -> None:
    """If model RDM equals neural RDM and neural splits are noise-free, r_xy_n_sb == 1."""
    rng = np.random.default_rng(0)
    rdm = rng.normal(size=200)
    res = neural_consistency(
        model_rdm=rdm,
        neural_rdm=rdm,
        neural_rdm_sh1=rdm,
        neural_rdm_sh2=rdm,
    )
    assert res.r_xy == pytest.approx(1.0)
    assert res.r_xx == pytest.approx(1.0)
    assert res.r_yy == pytest.approx(1.0)
    assert res.r_xy_n_sb == pytest.approx(1.0)


def test_pure_noise_model_zero_consistency() -> None:
    """Random model RDM uncorrelated with neural ⇒ r_xy_n_sb ≈ 0."""
    rng = np.random.default_rng(1)
    neural = rng.normal(size=2000)
    model = rng.normal(size=2000)  # independent
    res = neural_consistency(
        model_rdm=model,
        neural_rdm=neural,
        neural_rdm_sh1=neural,
        neural_rdm_sh2=neural,
    )
    # Both numerator and denominator should be small/finite; consistency near zero.
    assert abs(res.r_xy) < 0.1
    assert abs(res.r_xy_n_sb) < 0.1


def test_deterministic_model_r_yy_is_one() -> None:
    """When model split-halves default to the same vector, r_yy = 1."""
    rng = np.random.default_rng(2)
    rdm = rng.normal(size=300)
    res = neural_consistency(
        model_rdm=rdm,
        neural_rdm=rdm,
        neural_rdm_sh1=rdm,
        neural_rdm_sh2=rdm,
    )
    assert res.r_yy == pytest.approx(1.0)


def test_split_half_reliability_lifts_consistency() -> None:
    """Lower r_xx (noisier neural splits) should *raise* the noise-corrected score
    above the raw r_xy (since we're dividing by sqrt of something < 1).

    Build: neural = signal + small noise; sh1, sh2 = signal + larger independent
    noise. Model = signal exactly.
    """
    rng = np.random.default_rng(3)
    n = 500
    signal = rng.normal(size=n)
    noise_sh = 0.7  # large enough that sh1 vs sh2 reliability ~0.5–0.7
    sh1 = signal + rng.normal(size=n) * noise_sh
    sh2 = signal + rng.normal(size=n) * noise_sh
    neural = (sh1 + sh2) / 2.0  # combined estimate
    model = signal.copy()

    res = neural_consistency(
        model_rdm=model,
        neural_rdm=neural,
        neural_rdm_sh1=sh1,
        neural_rdm_sh2=sh2,
    )
    # raw correlation is high but < 1 because neural is averaged-noisy
    assert 0.5 < res.r_xy < 1.0
    # split-half is materially below 1
    assert 0.2 < res.r_xx < 0.95
    # noise-correction lifts the score toward 1
    assert res.r_xy_n_sb > res.r_xy


def test_shape_mismatch_raises() -> None:
    rng = np.random.default_rng(4)
    rdm = rng.normal(size=100)
    short = rng.normal(size=50)
    with pytest.raises(ValueError, match="RDM shape mismatch"):
        neural_consistency(
            model_rdm=rdm,
            neural_rdm=short,
            neural_rdm_sh1=rdm,
            neural_rdm_sh2=rdm,
        )


def test_returns_consistency_result_with_n_pairs() -> None:
    rng = np.random.default_rng(5)
    rdm = rng.normal(size=42)
    res = neural_consistency(
        model_rdm=rdm,
        neural_rdm=rdm,
        neural_rdm_sh1=rdm,
        neural_rdm_sh2=rdm,
    )
    assert isinstance(res, ConsistencyResult)
    assert res.n_pairs == 42


def test_from_states_smoke() -> None:
    """End-to-end on synthetic states and matching neural responses."""
    rng = np.random.default_rng(6)
    n_cond, T, n_feat, n_units = 8, 6, 5, 12

    # Model states
    model_states = rng.normal(size=(n_cond, T, n_feat))
    # Neural responses are an orthogonal embedding of model states so the
    # Euclidean RDM is preserved exactly: pad with zero columns then rotate.
    embed = np.zeros((n_feat, n_units))
    embed[:n_feat, :n_feat] = np.eye(n_feat)
    Q, _ = np.linalg.qr(rng.normal(size=(n_units, n_units)))
    proj = embed @ Q  # (n_feat, n_units), preserves pairwise euclidean
    neural_cells = (model_states.reshape(n_cond * T, n_feat) @ proj).reshape(n_cond, T, n_units)
    neural = np.transpose(neural_cells, (2, 0, 1))  # → (n_units, n_cond, T)
    neural_sh1 = neural + rng.normal(scale=1e-3, size=neural.shape)
    neural_sh2 = neural + rng.normal(scale=1e-3, size=neural.shape)

    mask = np.ones((n_cond, T), dtype=bool)
    res = neural_consistency_from_states(
        model_states=model_states,
        neural_responses=neural,
        neural_responses_sh1=neural_sh1,
        neural_responses_sh2=neural_sh2,
        mask=mask,
        mask_name="full",
    )
    # Orthogonal embedding preserves Euclidean RDM exactly; r_xy ≈ 1.
    assert res.r_xy > 0.99
    assert res.r_xy_n_sb > 0.99


def test_from_states_rejects_wrong_neural_shape() -> None:
    rng = np.random.default_rng(7)
    states = rng.normal(size=(4, 3, 5))
    bad_neural = rng.normal(size=(4, 3))  # 2-D, not 3-D
    mask = np.ones((4, 3), dtype=bool)
    with pytest.raises(ValueError, match="neural_responses must be 3-D"):
        neural_consistency_from_states(
            model_states=states,
            neural_responses=bad_neural,
            neural_responses_sh1=bad_neural,
            neural_responses_sh2=bad_neural,
            mask=mask,
        )
