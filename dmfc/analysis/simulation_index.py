"""Simulation Index — k-fold OLS decoder of occluded ball position (Fig. 4E).

Trains a linear decoder from per-bin model states to true ball ``(x, y)``
position over the visible+occluded epoch (``train_mask``), tests it during
the occluded epoch only (``test_mask``), reports per-coordinate decoding
error on held-out conditions. Mirrors Rajalingham's
``get_model_simulation_index`` at
``analyses/rnn/rnn_analysis/rnn_analysis_utils.py:290``.

Headline scalar (``si``) is the average MAE across the two coordinates,
matching their ``decode_vis-sim_to_sim_index_mae_k2`` in Fig. 4E. Lower
means better online simulation of the ball during occlusion.

Pure numerics — no I/O, no torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold


@dataclass(frozen=True)
class SimulationIndexResult:
    """Per-coordinate decoding error on the occluded epoch (held-out conditions)."""

    mae: np.ndarray  # (n_targets,) e.g. (2,) for (ball_x, ball_y)
    rmse: np.ndarray  # (n_targets,)
    rho: np.ndarray  # (n_targets,)
    si: float  # mean(mae) — headline scalar
    k: int  # k-fold splits over conditions
    n_test_cells: int  # cells contributing to test metrics


def _flatten(states: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """``states[mask]`` collapsing ``(n_cond_local, T, ...)`` to ``(n_cells, ...)``."""
    return states[mask]


def simulation_index(
    states: np.ndarray,
    ball_xy: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    k: int = 2,
    seed: int = 0,
) -> SimulationIndexResult:
    """k-fold OLS decoder of ``ball_xy`` from ``states``, occluded-epoch metric.

    Parameters
    ----------
    states
        Shape ``(n_cond, T, n_features)``.
    ball_xy
        Shape ``(n_cond, T, n_targets)``, typically ``n_targets=2`` for
        ``(ball_x, ball_y)``.
    train_mask
        Shape ``(n_cond, T)`` boolean. Cells in this mask contribute to
        training (visible+occluded valid bins per Rajalingham's pipeline).
    test_mask
        Shape ``(n_cond, T)`` boolean. Cells here are scored on held-out
        conditions (occluded epoch only).
    k
        Number of KFold splits over the condition axis. ``k=2`` matches
        Fig. 4E's reported ``mae_k2``.
    seed
        ``KFold`` shuffle seed.

    Returns
    -------
    SimulationIndexResult
        ``si`` = mean of per-coordinate MAE; lower is better.

    Notes
    -----
    KFold splits over conditions (rows), not cells. A condition's bins are
    therefore either all-train or all-test within a fold, preventing leakage.
    Predictions across folds are concatenated into one held-out vector per
    coordinate before computing metrics, matching the paper's behavior.
    """
    if states.ndim != 3:
        raise ValueError(f"expected 3-D states (n_cond, T, n_features), got {states.shape}")
    if ball_xy.ndim != 3:
        raise ValueError(f"expected 3-D ball_xy (n_cond, T, n_targets), got {ball_xy.shape}")
    if states.shape[:2] != ball_xy.shape[:2]:
        raise ValueError(f"states[:2] {states.shape[:2]} != ball_xy[:2] {ball_xy.shape[:2]}")
    if train_mask.shape != states.shape[:2]:
        raise ValueError(f"train_mask shape {train_mask.shape} mismatches {states.shape[:2]}")
    if test_mask.shape != states.shape[:2]:
        raise ValueError(f"test_mask shape {test_mask.shape} mismatches {states.shape[:2]}")
    n_cond = states.shape[0]
    n_targets = ball_xy.shape[2]

    if k < 2 or k > n_cond:
        raise ValueError(f"k={k} out of range for n_cond={n_cond}")

    train_mask_b = train_mask.astype(bool)
    test_mask_b = test_mask.astype(bool)

    splitter = KFold(n_splits=k, shuffle=True, random_state=seed)

    y_true_chunks: list[np.ndarray] = []
    y_pred_chunks: list[np.ndarray] = []

    for train_idx, test_idx in splitter.split(np.arange(n_cond)):
        # Train on all train_mask cells of training conditions.
        X_tr = _flatten(states[train_idx], train_mask_b[train_idx])
        y_tr = _flatten(ball_xy[train_idx], train_mask_b[train_idx])
        if X_tr.shape[0] < 2:
            continue
        reg = LinearRegression()
        reg.fit(X_tr, y_tr)

        # Test on test_mask cells of held-out conditions (occluded epoch).
        X_te = _flatten(states[test_idx], test_mask_b[test_idx])
        y_te = _flatten(ball_xy[test_idx], test_mask_b[test_idx])
        if X_te.shape[0] == 0:
            continue
        y_hat = reg.predict(X_te)
        y_true_chunks.append(y_te)
        y_pred_chunks.append(y_hat)

    if not y_true_chunks:
        nan_targets = np.full(n_targets, np.nan)
        return SimulationIndexResult(
            mae=nan_targets,
            rmse=nan_targets.copy(),
            rho=nan_targets.copy(),
            si=float("nan"),
            k=k,
            n_test_cells=0,
        )

    y_true = np.concatenate(y_true_chunks, axis=0)  # (n_test_cells, n_targets)
    y_pred = np.concatenate(y_pred_chunks, axis=0)

    mae = np.zeros(n_targets, dtype=np.float64)
    rmse = np.zeros(n_targets, dtype=np.float64)
    rho = np.zeros(n_targets, dtype=np.float64)
    for i in range(n_targets):
        a = y_true[:, i]
        b = y_pred[:, i]
        finite = np.isfinite(a) & np.isfinite(b)
        if finite.sum() < 2:
            mae[i] = np.nan
            rmse[i] = np.nan
            rho[i] = np.nan
            continue
        a_f = a[finite]
        b_f = b[finite]
        mae[i] = float(np.mean(np.abs(a_f - b_f)))
        rmse[i] = float(np.sqrt(np.mean((a_f - b_f) ** 2)))
        if a_f.std() == 0.0 or b_f.std() == 0.0:
            rho[i] = np.nan
        else:
            rho[i] = float(cast(float, pearsonr(a_f, b_f)[0]))

    si = float(np.nanmean(mae))
    return SimulationIndexResult(
        mae=mae,
        rmse=rmse,
        rho=rho,
        si=si,
        k=k,
        n_test_cells=int(y_true.shape[0]),
    )
