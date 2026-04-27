"""Noise-corrected representational similarity (Fig. 4D metric).

Compares an IN's representational geometry to the DMFC neural population by
correlating their pairwise-distance RDMs, then dividing out the geometric
mean of split-half reliabilities (Spearman-Brown corrected). Mirrors
Rajalingham's ``RnnNeuralComparer.get_noise_corrected_corr`` at
``analyses/rnn/RnnNeuralComparer.py:53``.

For a deterministic IN (no trial noise), ``Y1 == Y2 == Y`` so ``r_yy = 1``
and ``SB(r_yy) = 1``; the denominator simplifies to ``sqrt(SB(r_xx))``. We
keep the general form here for symmetry with stochastic-model comparisons
(noise-injected ablations) that may land later.

Restricted to the ``occ_end_pad0`` mask in the canonical Fig. 4D pipeline
(occluded epoch only). Pure numerics — no I/O, no torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from scipy.stats import pearsonr

from dmfc.analysis.rdm import RDMResult, compute_rdm


@dataclass(frozen=True)
class ConsistencyResult:
    """Outputs of one noise-corrected RSA comparison."""

    r_xy: float  # raw Pearson r between RDMs
    r_xx: float  # split-half reliability of X (neural)
    r_yy: float  # split-half reliability of Y (model)
    r_xy_n_sb: float  # noise-corrected, Spearman-Brown
    r_xy_n: float  # noise-corrected without Spearman-Brown (alternative form)
    n_pairs: int  # length of the RDM vectors


def _sb(r: float) -> float:
    """Spearman-Brown upgrade for a half-length reliability."""
    if not np.isfinite(r):
        return float("nan")
    if r <= -1.0:
        return float("nan")
    return float(2.0 * r / (1.0 + r))


def _safe_pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r over finite-in-both positions; returns NaN on degenerate inputs."""
    t = np.isfinite(x) & np.isfinite(y)
    if t.sum() < 2:
        return float("nan")
    xt = x[t]
    yt = y[t]
    if xt.std() == 0.0 or yt.std() == 0.0:
        return float("nan")
    return float(cast(float, pearsonr(xt, yt)[0]))


def neural_consistency(
    model_rdm: np.ndarray,
    neural_rdm: np.ndarray,
    neural_rdm_sh1: np.ndarray,
    neural_rdm_sh2: np.ndarray,
    model_rdm_sh1: np.ndarray | None = None,
    model_rdm_sh2: np.ndarray | None = None,
) -> ConsistencyResult:
    """Compute the noise-corrected RSA score from pre-computed flat RDMs.

    Parameters
    ----------
    model_rdm, neural_rdm
        Full-data flat RDMs (from :func:`dmfc.analysis.rdm.compute_rdm`).
    neural_rdm_sh1, neural_rdm_sh2
        Split-half neural RDMs (computed on ``responses_sh1`` / ``responses_sh2``).
    model_rdm_sh1, model_rdm_sh2
        Optional model split-halves. If omitted, both default to ``model_rdm``,
        which is appropriate for a deterministic model — ``r_yy`` becomes 1.

    Returns
    -------
    ConsistencyResult
        ``r_xy_n_sb`` is the headline score (Rajalingham Fig. 4D).
    """
    rdms = (model_rdm, neural_rdm, neural_rdm_sh1, neural_rdm_sh2)
    n_pairs = rdms[0].shape[0]
    for r in rdms:
        if r.shape != (n_pairs,):
            raise ValueError(f"RDM shape mismatch: expected ({n_pairs},), got {r.shape}")

    if model_rdm_sh1 is None:
        model_rdm_sh1 = model_rdm
    if model_rdm_sh2 is None:
        model_rdm_sh2 = model_rdm

    r_xy = _safe_pearsonr(neural_rdm, model_rdm)
    r_xy_v2 = float(
        np.nanmean(
            [
                _safe_pearsonr(neural_rdm_sh2, model_rdm_sh1),
                _safe_pearsonr(neural_rdm_sh1, model_rdm_sh2),
            ]
        )
    )
    r_xx = _safe_pearsonr(neural_rdm_sh1, neural_rdm_sh2)
    r_yy = _safe_pearsonr(model_rdm_sh1, model_rdm_sh2)

    denom = float(np.sqrt(r_xx * r_yy)) if (r_xx > 0 and r_yy > 0) else float("nan")
    sb_xx = _sb(r_xx)
    sb_yy = _sb(r_yy)
    denom_sb = float(np.sqrt(sb_xx * sb_yy)) if (sb_xx > 0 and sb_yy > 0) else float("nan")

    r_xy_n = r_xy_v2 / denom if (denom and np.isfinite(denom)) else float("nan")
    r_xy_n_sb = r_xy / denom_sb if (denom_sb and np.isfinite(denom_sb)) else float("nan")

    return ConsistencyResult(
        r_xy=r_xy,
        r_xx=r_xx,
        r_yy=r_yy,
        r_xy_n_sb=r_xy_n_sb,
        r_xy_n=r_xy_n,
        n_pairs=n_pairs,
    )


def neural_consistency_from_states(
    model_states: np.ndarray,
    neural_responses: np.ndarray,
    neural_responses_sh1: np.ndarray,
    neural_responses_sh2: np.ndarray,
    mask: np.ndarray,
    metric: str = "euclidean",
    mask_name: str | None = None,
) -> ConsistencyResult:
    """Convenience: compute RDMs from state arrays then run the comparison.

    Parameters
    ----------
    model_states
        IN states, shape ``(n_cond, T, n_features)`` (typically the output of
        :func:`dmfc.analysis.endpoint_decoding.flatten_receivers`).
    neural_responses, neural_responses_sh1, neural_responses_sh2
        DMFC reliable-unit responses, shape ``(n_units, n_cond, T)`` (the
        Zenodo native layout). Internally transposed to ``(n_cond, T, n_units)``
        for RDM construction.
    mask
        ``(n_cond, T)`` boolean or NaN-mask defining which cells contribute
        to the RDM (typically ``occ_end_pad0`` for Fig. 4D).
    metric
        Distance metric for ``compute_rdm``; ``"euclidean"`` matches the paper.
    """
    if neural_responses.ndim != 3:
        raise ValueError(
            f"neural_responses must be 3-D (n_units, n_cond, T); got {neural_responses.shape}"
        )

    neural = np.transpose(neural_responses, (1, 2, 0))  # (n_cond, T, n_units)
    neural_sh1 = np.transpose(neural_responses_sh1, (1, 2, 0))
    neural_sh2 = np.transpose(neural_responses_sh2, (1, 2, 0))

    model_rdm = compute_rdm(model_states, mask, metric=metric, mask_name=mask_name).rdm  # type: ignore[arg-type]
    neural_rdm = compute_rdm(neural, mask, metric=metric, mask_name=mask_name).rdm  # type: ignore[arg-type]
    neural_rdm_sh1 = compute_rdm(neural_sh1, mask, metric=metric).rdm  # type: ignore[arg-type]
    neural_rdm_sh2 = compute_rdm(neural_sh2, mask, metric=metric).rdm  # type: ignore[arg-type]

    return neural_consistency(
        model_rdm=model_rdm,
        neural_rdm=neural_rdm,
        neural_rdm_sh1=neural_rdm_sh1,
        neural_rdm_sh2=neural_rdm_sh2,
    )


__all__ = [
    "ConsistencyResult",
    "neural_consistency",
    "neural_consistency_from_states",
    "RDMResult",
]
