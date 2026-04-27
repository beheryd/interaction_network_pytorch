"""Identity tests for ``dmfc.analysis.validate_pipeline``.

The script is a one-shot validation that our RDM / NC / SI implementations
match Rajalingham's reference functions on synthetic input. These tests
exercise the math identity directly (no CLI), and skip cleanly when the
unpacked Zenodo source tree isn't present.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dmfc.analysis.validate_pipeline import (
    DEFAULT_ZENODO_ROOT,
    _make_synthetic,
    import_reference,
    validate_neural_consistency,
    validate_rdm,
    validate_simulation_index,
)


def _zenodo_available() -> bool:
    return (DEFAULT_ZENODO_ROOT / "code" / "utils" / "phys_utils.py").exists()


@pytest.mark.skipif(not _zenodo_available(), reason="Zenodo source tree not available")
def test_rdm_matches_reference() -> None:
    phys_utils, _, _ = import_reference(DEFAULT_ZENODO_ROOT)
    synth = _make_synthetic(seed=0)
    diffs = validate_rdm(synth, phys_utils)
    for side, d in diffs.items():
        assert d == 0.0, f"RDM mismatch for {side}: {d}"


@pytest.mark.skipif(not _zenodo_available(), reason="Zenodo source tree not available")
def test_neural_consistency_matches_reference() -> None:
    _, comparer_cls, _ = import_reference(DEFAULT_ZENODO_ROOT)
    if comparer_cls is None:
        pytest.skip("RnnNeuralComparer could not be imported in this environment")
    synth = _make_synthetic(seed=0)
    diffs = validate_neural_consistency(synth, comparer_cls)
    # All five quantities (r_xy, r_xx, r_yy, r_xy_n, r_xy_n_sb) should match
    # bit-for-bit since both pipelines reduce to the same scipy.pearsonr calls.
    for k, v in diffs.items():
        assert v == 0.0, f"NC mismatch for {k}: {v}"


def test_simulation_index_matches_inline_reference() -> None:
    """SI identity does not depend on the Zenodo tree (inline reference)."""
    synth = _make_synthetic(seed=0)
    diffs = validate_simulation_index(synth)
    assert diffs["mae_per_coord_max_diff"] == 0.0
    assert diffs["si_diff"] == 0.0


def test_synthetic_inputs_are_deterministic() -> None:
    a = _make_synthetic(seed=0)
    b = _make_synthetic(seed=0)
    for k in a:
        assert (
            (a[k] == b[k]).all()
            if a[k].dtype != float or not _has_nan(a[k])
            else _nan_eq(a[k], b[k])
        ), f"non-deterministic key: {k}"


def _has_nan(x):  # type: ignore[no-untyped-def]
    import numpy as np

    return bool(np.isnan(x).any())


def _nan_eq(a, b) -> bool:  # type: ignore[no-untyped-def]
    import numpy as np

    return bool(((a == b) | (np.isnan(a) & np.isnan(b))).all())


_ = Path  # keep Path imported for parity with other test files
