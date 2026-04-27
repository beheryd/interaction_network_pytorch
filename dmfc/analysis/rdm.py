"""Pairwise-distance representational dissimilarity matrices.

Building block for ``neural_consistency.py`` and any RSA-style comparison.
Mirrors Rajalingham's ``get_state_pairwise_distances`` in
``phys_utils.py:371``: stack all valid ``(condition, timestep)`` cells into a
single sample matrix, then compute pairwise distances between samples.

The returned RDM is the strict lower triangle (no diagonal) flattened to a
1-D vector. Their code stores the whole flat matrix with NaN in the
upper-triangle + diagonal positions and relies on a NaN-safe Pearson r to
filter; we drop the redundant entries up front so downstream callers can use
plain ``pearsonr``.

Pure numerics on numpy arrays — no I/O, no torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.metrics import nan_euclidean_distances


@dataclass(frozen=True)
class RDMResult:
    """Flat lower-triangle RDM with provenance for downstream comparisons."""

    rdm: np.ndarray  # (n_pairs,) where n_pairs = n_cells * (n_cells - 1) // 2
    n_cells: int  # number of (condition, timestep) cells included
    n_features: int
    metric: str
    mask_name: str | None = None


def _mask_to_bool(mask: np.ndarray) -> np.ndarray:
    """Accept either a boolean mask or Rajalingham's NaN/1.0 mask."""
    if mask.dtype == np.bool_:
        return mask
    return np.isfinite(mask)


def compute_rdm(
    states: np.ndarray,
    mask: np.ndarray,
    metric: Literal["euclidean", "corr"] = "euclidean",
    mask_name: str | None = None,
) -> RDMResult:
    """Pairwise distances between all valid ``(condition, timestep)`` cells.

    Parameters
    ----------
    states
        Shape ``(n_cond, T, n_features)`` (the ``flatten_receivers`` output for
        IN states; for DMFC neural data, transpose ``(n_units, n_cond, T)`` to
        ``(n_cond, T, n_units)`` before calling).
    mask
        Shape ``(n_cond, T)``. Either boolean (``True`` = valid) or float with
        NaN at invalid positions and finite values elsewhere (Rajalingham's
        convention). The two conventions are interchangeable.
    metric
        ``"euclidean"`` (default, NaN-safe via ``nan_euclidean_distances``) or
        ``"corr"`` (Pearson distance, computed as ``np.corrcoef``). The paper
        reports euclidean for Fig. 4D.
    mask_name
        Optional identifier for traceability in the result (e.g.
        ``"occ_end_pad0"``).

    Returns
    -------
    RDMResult
        ``rdm`` is the flattened strict lower triangle, length
        ``n_cells * (n_cells - 1) // 2``.

    Notes
    -----
    Cells are pooled across conditions and time. So an RDM under
    ``occ_end_pad0`` for the 79 conditions has ``n_cells`` ≈ sum of
    occluded-epoch lengths across the 79.
    """
    if states.ndim != 3:
        raise ValueError(f"expected 3-D states (n_cond, T, n_features), got {states.shape}")
    if mask.shape != states.shape[:2]:
        raise ValueError(f"mask shape {mask.shape} does not match states[:2]={states.shape[:2]}")

    bool_mask = _mask_to_bool(mask)
    cells = states[bool_mask]  # (n_cells, n_features)
    n_cells, n_features = cells.shape
    if n_cells < 2:
        raise ValueError(f"need at least 2 valid cells to form an RDM, got {n_cells}")

    if metric == "euclidean":
        full = nan_euclidean_distances(cells)
    elif metric == "corr":
        # corrcoef is "similarity"; convert to distance (1 - r) so larger ⇒ farther.
        full = 1.0 - np.corrcoef(cells)
    else:
        raise ValueError(f"unknown metric {metric!r}; expected 'euclidean' or 'corr'")

    tril_idx = np.tril_indices(n_cells, k=-1)
    flat = full[tril_idx].astype(np.float64)

    return RDMResult(
        rdm=flat,
        n_cells=n_cells,
        n_features=n_features,
        metric=metric,
        mask_name=mask_name,
    )
