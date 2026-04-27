"""Tests for ``dmfc.analysis.rdm``."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import pdist

from dmfc.analysis.rdm import RDMResult, compute_rdm


def _full_mask(n_cond: int, T: int) -> np.ndarray:
    return np.ones((n_cond, T), dtype=bool)


def test_zeros_states_zero_rdm() -> None:
    states = np.zeros((4, 3, 5))
    res = compute_rdm(states, _full_mask(4, 3), metric="euclidean")
    assert isinstance(res, RDMResult)
    assert res.rdm.shape == (4 * 3 * (4 * 3 - 1) // 2,)
    np.testing.assert_allclose(res.rdm, 0.0)


def test_matches_scipy_pdist_euclidean() -> None:
    rng = np.random.default_rng(0)
    states = rng.normal(size=(5, 4, 7))
    res = compute_rdm(states, _full_mask(5, 4), metric="euclidean")

    cells = states.reshape(5 * 4, 7)
    expected = pdist(cells, metric="euclidean")  # condensed = strict lower tri
    # pdist orders pairs as (0,1), (0,2), ..., (0,n-1), (1,2), ... — that's the
    # row-major upper triangle. tril_indices gives the lower triangle in
    # column-major order. The two are equal in *value* because the distance
    # matrix is symmetric, but the orderings differ. Sort before comparing.
    np.testing.assert_allclose(np.sort(res.rdm), np.sort(expected))


def test_mask_bool_and_nan_equivalent() -> None:
    rng = np.random.default_rng(1)
    states = rng.normal(size=(3, 4, 6))
    bool_mask = np.array([[1, 1, 0, 1], [0, 1, 1, 1], [1, 1, 1, 0]], dtype=bool)
    nan_mask = np.where(bool_mask, 1.0, np.nan)

    a = compute_rdm(states, bool_mask)
    b = compute_rdm(states, nan_mask)
    np.testing.assert_array_equal(a.rdm, b.rdm)
    assert a.n_cells == b.n_cells == int(bool_mask.sum())


def test_mask_drops_invalid_cells() -> None:
    rng = np.random.default_rng(2)
    states = rng.normal(size=(4, 3, 5))
    full_mask = _full_mask(4, 3)
    full_res = compute_rdm(states, full_mask)

    # Mask out the last timestep entirely.
    partial_mask = full_mask.copy()
    partial_mask[:, -1] = False
    partial_res = compute_rdm(states, partial_mask)

    assert partial_res.n_cells == 4 * 2
    assert full_res.n_cells == 4 * 3
    assert partial_res.rdm.shape[0] < full_res.rdm.shape[0]


def test_metric_corr_in_range() -> None:
    rng = np.random.default_rng(3)
    states = rng.normal(size=(5, 4, 12))
    res = compute_rdm(states, _full_mask(5, 4), metric="corr")
    # correlation distance is 1 - r, so it lives in [0, 2]
    assert (res.rdm >= 0.0).all()
    assert (res.rdm <= 2.0).all()


def test_unknown_metric_raises() -> None:
    states = np.zeros((3, 2, 4))
    with pytest.raises(ValueError, match="unknown metric"):
        compute_rdm(states, _full_mask(3, 2), metric="cosine")  # type: ignore[arg-type]


def test_too_few_cells_raises() -> None:
    states = np.zeros((1, 1, 5))
    with pytest.raises(ValueError, match="at least 2"):
        compute_rdm(states, _full_mask(1, 1))


def test_mask_shape_mismatch_raises() -> None:
    states = np.zeros((3, 4, 5))
    with pytest.raises(ValueError, match="mask shape"):
        compute_rdm(states, np.ones((3, 5), dtype=bool))


def test_states_must_be_3d() -> None:
    with pytest.raises(ValueError, match="3-D states"):
        compute_rdm(np.zeros((4, 5)), np.ones((4, 5), dtype=bool))


def test_mask_name_passed_through() -> None:
    states = np.zeros((3, 2, 4))
    res = compute_rdm(states, _full_mask(3, 2), mask_name="occ_end_pad0")
    assert res.mask_name == "occ_end_pad0"


def test_handles_nans_via_nan_euclidean() -> None:
    """NaN entries in valid cells should not crash; sklearn handles them."""
    rng = np.random.default_rng(4)
    states = rng.normal(size=(3, 4, 5))
    states[0, 0, 2] = np.nan  # one NaN in an otherwise valid cell
    res = compute_rdm(states, _full_mask(3, 4))
    assert np.isfinite(res.rdm).all()
