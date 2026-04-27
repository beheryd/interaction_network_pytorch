"""Time-resolved endpoint decoding from per-condition state trajectories.

Primary metric for the Fig. 5B replication. Trains a linear decoder at each
timestep ``t`` to predict the trial's eventual ball-y endpoint from the state
at time ``t``, cross-validated across conditions (``GroupKFold`` with one
condition per group). Returns Pearson r and RMSE vectors over time.

Pure numerics on numpy arrays — no I/O, no torch. Loaders in
``dmfc.rajalingham.load`` and the convenience helper ``load_pilot_states``
below sit on top of this module; the figure-reproduction script glues them.

Mirrors Rajalingham's ``linear_regress_grouped`` (in their ``phys_utils.py``)
in the cross-validation strategy. The decoder itself is a plain
``LinearRegression`` because the target is one-dimensional (the eventual
endpoint y) — PLS, which their helper uses for multi-output targets, is
unnecessary here.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

# Output index 6 of the 7-d IN target vector is the final intercept y. See
# SCRATCHPAD M3 closeout § "Pinned design choices".
ENDPOINT_OUTPUT_INDEX: int = 6


@dataclass(frozen=True)
class DecodingResult:
    """Per-timestep decoding of a scalar target from per-condition states."""

    r: np.ndarray  # (T,) Pearson r per timestep, mean across folds
    rmse: np.ndarray  # (T,) RMSE per timestep, mean across folds
    r_per_fold: np.ndarray  # (n_splits, T)
    rmse_per_fold: np.ndarray  # (n_splits, T)
    n_conditions: int
    n_timesteps: int
    n_features: int


def flatten_receivers(effect_receivers: np.ndarray) -> np.ndarray:
    """Collapse the (object, effect) axes of ``effect_receivers``.

    Input  shape: ``(n_conditions, T, n_objects, effect_dim)``
    Output shape: ``(n_conditions, T, n_objects * effect_dim)``
    """
    if effect_receivers.ndim != 4:
        raise ValueError(f"expected 4-D effect_receivers, got shape {effect_receivers.shape}")
    n_cond, t, n_obj, eff = effect_receivers.shape
    return effect_receivers.reshape(n_cond, t, n_obj * eff)


def _pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r on 1-D arrays. Returns NaN if either input has zero variance."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xm = x - x.mean()
    ym = y - y.mean()
    denom = float(np.sqrt(np.sum(xm * xm) * np.sum(ym * ym)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(xm * ym) / denom)


def decode_endpoint(
    states: np.ndarray,
    endpoint_y: np.ndarray,
    valid_mask: np.ndarray | None = None,
    n_splits: int = 5,
) -> DecodingResult:
    """Per-timestep linear decoding of ``endpoint_y`` from ``states``.

    Parameters
    ----------
    states
        Float array of shape ``(n_conditions, T, n_features)``.
    endpoint_y
        Float array of shape ``(n_conditions,)``. The target is constant within
        a condition (it is the trial's eventual endpoint), so the decoder is
        asked: "from the time-t state, predict the endpoint."
    valid_mask
        Optional ``(n_conditions, T)`` mask. If a ``(c, t)`` entry is zero the
        condition is excluded from the per-t Pearson/RMSE at that timestep
        only; the decoder is still trained on all available training-fold
        rows at that timestep. Use this to skip post-end padding produced by
        ragged trial lengths.
    n_splits
        Number of GroupKFold splits across the 79 conditions. Default 5
        matches the smallest sensible split for 79 groups.

    Returns
    -------
    DecodingResult
        Per-fold and fold-averaged ``r`` and ``rmse`` vectors of length T.

    Notes
    -----
    Splits are over conditions only (no CV across timesteps), matching
    Rajalingham's ``linear_regress_grouped``. This prevents leakage in which
    different timesteps from the same condition could land in train and test
    folds of the same split.
    """
    if states.ndim != 3:
        raise ValueError(f"expected 3-D states (n_cond, T, n_features), got {states.shape}")
    if endpoint_y.ndim != 1:
        raise ValueError(f"expected 1-D endpoint_y (n_cond,), got {endpoint_y.shape}")
    n_cond, T, n_feat = states.shape
    if endpoint_y.shape[0] != n_cond:
        raise ValueError(
            f"endpoint_y has {endpoint_y.shape[0]} entries but states has {n_cond} conditions"
        )
    if valid_mask is not None and valid_mask.shape != (n_cond, T):
        raise ValueError(
            f"valid_mask shape {valid_mask.shape} does not match (n_cond, T)=({n_cond}, {T})"
        )

    groups = np.arange(n_cond)
    splitter = GroupKFold(n_splits=n_splits)

    r_per_fold = np.full((n_splits, T), np.nan, dtype=np.float64)
    rmse_per_fold = np.full((n_splits, T), np.nan, dtype=np.float64)

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(states, endpoint_y, groups)):
        y_train = endpoint_y[train_idx]
        y_test = endpoint_y[test_idx]
        for t in range(T):
            X_train = states[train_idx, t, :]
            X_test = states[test_idx, t, :]

            if valid_mask is not None:
                m_train = valid_mask[train_idx, t].astype(bool)
                m_test = valid_mask[test_idx, t].astype(bool)
                if m_train.sum() < 2 or m_test.sum() < 2:
                    # Not enough valid points to fit or to compute r; leave NaN.
                    continue
                X_train = X_train[m_train]
                y_train_t = y_train[m_train]
                X_test = X_test[m_test]
                y_test_t = y_test[m_test]
            else:
                y_train_t = y_train
                y_test_t = y_test

            reg = LinearRegression()
            reg.fit(X_train, y_train_t)
            y_pred = reg.predict(X_test)
            r_per_fold[fold_idx, t] = _pearsonr(y_test_t, y_pred)
            rmse_per_fold[fold_idx, t] = float(np.sqrt(np.mean((y_test_t - y_pred) ** 2)))

    # All-NaN columns can occur if a timestep has too few valid points in every
    # fold; nanmean's "Mean of empty slice" warning is the documented signal for
    # that case and we already encode it as NaN in the result.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"Mean of empty slice", RuntimeWarning)
        r_mean = np.nanmean(r_per_fold, axis=0)
        rmse_mean = np.nanmean(rmse_per_fold, axis=0)

    return DecodingResult(
        r=r_mean,
        rmse=rmse_mean,
        r_per_fold=r_per_fold,
        rmse_per_fold=rmse_per_fold,
        n_conditions=n_cond,
        n_timesteps=T,
        n_features=n_feat,
    )


def load_pilot_states(run_dir: Path | str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read ``hidden_states.npz`` from a run directory and unpack for decoding.

    Returns
    -------
    states_flat
        ``(79, T, 2 * effect_dim)`` — receivers flattened across objects.
    endpoint_y
        ``(79,)`` — true endpoint y per condition, taken from
        ``targets[:, 0, ENDPOINT_OUTPUT_INDEX]``. Constant within a valid
        condition trace; index 0 is always inside the valid window.
    valid_mask
        ``(79, T)`` — same as the trained model's valid_mask.
    """
    npz_path = Path(run_dir) / "hidden_states.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"no hidden_states.npz at {npz_path}")
    with np.load(npz_path) as data:
        receivers = np.asarray(data["effect_receivers"], dtype=np.float64)
        targets = np.asarray(data["targets"], dtype=np.float64)
        valid_mask = np.asarray(data["valid_mask"], dtype=np.float64)

    states_flat = flatten_receivers(receivers)
    endpoint_y = targets[:, 0, ENDPOINT_OUTPUT_INDEX]
    return states_flat, endpoint_y, valid_mask
