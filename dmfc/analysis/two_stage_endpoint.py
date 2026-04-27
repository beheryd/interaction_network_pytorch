"""Two-stage endpoint decoding control (Rajalingham Supplementary Fig. S8D analog).

The IN's per-step input features include ground-truth ball position AND
velocity from frame 1 (``(x, y, dx, dy)`` per ``observation_to_object_features``
in ``dmfc.models.interaction_network``). With Mental Pong's deterministic
kinematics, those four scalars determine the trial endpoint in closed form,
so a linear decoder of IN states at any timestep can in principle recover
the endpoint *purely* from instantaneous kinematics — no genuine simulation
needed. The single-stage Fig. 5B curve cannot distinguish "the IN computes
the endpoint" from "the IN's hidden state contains decodable kinematics
that, by linear closed-form, happen to give the endpoint."

This module implements the same control Rajalingham used to test whether
DMFC's rapid early prediction was an initial-position artifact (paper page
9, Supplementary Fig. S8D):

* **Stage 1**: at each time t, fit a linear decoder ``state(t) → (x, y, dx, dy)(t)``.
* **Stage 2**: fit a linear decoder ``(x, y, dx, dy) → endpoint`` (per condition).
* The "kinematics-mediated" curve at time t is the Pearson r between true
  endpoint and the endpoint predicted from Stage-1's *predicted* kinematics
  via Stage 2.

If the kinematics-mediated curve sits at the same height as the direct
state→endpoint curve (``decode_endpoint``), the IN's apparent rapid rise
is *fully* explained by the kinematic shortcut. If it sits below, the
residual gap is the IN's genuine endpoint-encoding contribution beyond
what's mechanically inferable from instantaneous kinematics.

Pure numerics — no I/O, no torch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

from dmfc.analysis.endpoint_decoding import _pearsonr
from dmfc.envs.conditions import ConditionSpec, load_conditions
from dmfc.envs.mental_pong import _resample_to_bins, integrate_trajectory

# Default kinematic feature order: (x, y, dx, dy).
KIN_DIM: int = 4


@dataclass(frozen=True)
class TwoStageResult:
    """Direct vs. kinematics-mediated endpoint decoding, per timestep."""

    direct_r: np.ndarray  # (T,) — state → endpoint, mean across folds
    kinematics_mediated_r: np.ndarray  # (T,) — state → kinematics → endpoint
    kinematics_only_r: np.ndarray  # (T,) — true kinematics → endpoint (upper bound)
    state_to_kinematics_r: np.ndarray  # (T, K) — per-axis state→kinematics r
    n_conditions: int
    n_timesteps: int
    n_features: int
    kin_dim: int


def kinematics_for_canonical_79(
    T_max: int, dt_ms: int = 50, conditions: list[ConditionSpec] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``(x, y, dx, dy)`` over time for the canonical 79 conditions.

    Parameters
    ----------
    T_max
        Number of bins per condition. Conditions shorter than ``T_max`` are
        NaN-padded after their final valid bin.
    dt_ms
        Bin width in ms; default 50 to match the IN training grid.
    conditions
        Optional list of ``ConditionSpec`` (defaults to the canonical 79
        from ``load_conditions``).

    Returns
    -------
    kinematics : np.ndarray, shape (n_cond, T_max, 4)
        Columns are ``(x, y, dx, dy)`` in MWK degrees; ``dx``, ``dy`` are in
        deg per RNN-step (matches ``ConditionSpec`` units).
    valid : np.ndarray, shape (n_cond, T_max), bool
        ``True`` where the bin lies inside the integrated trajectory.
    """
    from dmfc.envs.conditions import RNN_STEP_MS

    conds = conditions if conditions is not None else load_conditions()
    n_cond = len(conds)
    out = np.full((n_cond, T_max, KIN_DIM), np.nan, dtype=np.float64)
    valid = np.zeros((n_cond, T_max), dtype=bool)
    for i, spec in enumerate(conds):
        traj_steps = integrate_trajectory(spec)
        _, traj_bins = _resample_to_bins(traj_steps, dt_ms=dt_ms, integrator_dt_ms=RNN_STEP_MS)
        n_bins = min(traj_bins.shape[0], T_max)
        out[i, :n_bins, :] = traj_bins[:n_bins]
        valid[i, :n_bins] = True
    return out, valid


def _fit_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    return reg.predict(X_test)


def two_stage_decode(
    states: np.ndarray,
    kinematics: np.ndarray,
    endpoint_y: np.ndarray,
    valid_mask: np.ndarray | None = None,
    n_splits: int = 5,
) -> TwoStageResult:
    """Per-timestep two-stage endpoint decoding.

    Parameters
    ----------
    states
        ``(n_cond, T, n_features)``. The model's per-bin hidden states
        (e.g., flattened ``effect_receivers``).
    kinematics
        ``(n_cond, T, K)``. Per-bin kinematic features. With ``K = 4`` and
        the canonical Mental Pong layout, columns are ``(x, y, dx, dy)``;
        any other K works as long as the same K is used as the Stage-1
        target and Stage-2 input. NaN cells are dropped from both stages.
    endpoint_y
        ``(n_cond,)``. The per-condition scalar endpoint (constant within
        a condition).
    valid_mask
        Optional ``(n_cond, T)`` mask. Cells where the mask is zero (or the
        kinematics are NaN) are excluded from per-t Pearson r computation.
        Decoder fitting at timestep t uses whatever rows are valid in the
        train fold at that timestep.
    n_splits
        ``GroupKFold`` splits across the ``n_cond`` conditions. Default 5.

    Returns
    -------
    TwoStageResult
        Three per-timestep curves and a per-axis state→kinematics curve.
    """
    if states.ndim != 3:
        raise ValueError(f"expected 3-D states (n_cond, T, n_features), got {states.shape}")
    if kinematics.ndim != 3:
        raise ValueError(f"expected 3-D kinematics (n_cond, T, K), got {kinematics.shape}")
    if states.shape[:2] != kinematics.shape[:2]:
        raise ValueError(f"states[:2] {states.shape[:2]} != kinematics[:2] {kinematics.shape[:2]}")
    if endpoint_y.shape != (states.shape[0],):
        raise ValueError(f"endpoint_y shape {endpoint_y.shape} != (n_cond,)=({states.shape[0]},)")
    n_cond, T, n_feat = states.shape
    K = kinematics.shape[2]

    if valid_mask is None:
        valid_mask = np.ones((n_cond, T), dtype=bool)
    elif valid_mask.shape != (n_cond, T):
        raise ValueError(f"valid_mask shape {valid_mask.shape} != (n_cond, T)=({n_cond}, {T})")

    # A cell is usable iff it's flagged valid AND every kinematic feature is finite.
    kin_finite = np.all(np.isfinite(kinematics), axis=2)
    cell_ok = valid_mask.astype(bool) & kin_finite

    splitter = GroupKFold(n_splits=n_splits)
    groups = np.arange(n_cond)

    direct_per_fold = np.full((n_splits, T), np.nan, dtype=np.float64)
    kinmed_per_fold = np.full((n_splits, T), np.nan, dtype=np.float64)
    kinonly_per_fold = np.full((n_splits, T), np.nan, dtype=np.float64)
    s2k_per_fold = np.full((n_splits, T, K), np.nan, dtype=np.float64)

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(states, endpoint_y, groups)):
        y_train = endpoint_y[train_idx]
        y_test = endpoint_y[test_idx]
        for t in range(T):
            train_ok = cell_ok[train_idx, t]
            test_ok = cell_ok[test_idx, t]
            if train_ok.sum() < max(K + 1, 2) or test_ok.sum() < 2:
                continue
            X_train = states[train_idx, t][train_ok]
            kin_train = kinematics[train_idx, t][train_ok]
            y_train_t = y_train[train_ok]

            X_test = states[test_idx, t][test_ok]
            kin_test = kinematics[test_idx, t][test_ok]
            y_test_t = y_test[test_ok]

            # Direct: state → endpoint
            y_pred_direct = _fit_predict(X_train, y_train_t, X_test)
            direct_per_fold[fold_idx, t] = _pearsonr(y_test_t, y_pred_direct)

            # Stage 1: state → kinematics
            kin_pred_test = _fit_predict(X_train, kin_train, X_test)
            for k in range(K):
                s2k_per_fold[fold_idx, t, k] = _pearsonr(kin_test[:, k], kin_pred_test[:, k])

            # Stage 2: kinematics → endpoint (trained on TRUE kinematics).
            y_pred_kinmed = _fit_predict(kin_train, y_train_t, kin_pred_test)
            kinmed_per_fold[fold_idx, t] = _pearsonr(y_test_t, y_pred_kinmed)

            # Reference: kinematics → endpoint, evaluated on TRUE test kinematics.
            # Upper bound on what the kinematic shortcut alone can achieve.
            y_pred_kinonly = _fit_predict(kin_train, y_train_t, kin_test)
            kinonly_per_fold[fold_idx, t] = _pearsonr(y_test_t, y_pred_kinonly)

    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.filterwarnings("ignore", r"Mean of empty slice", RuntimeWarning)
        direct_r = np.nanmean(direct_per_fold, axis=0)
        kinmed_r = np.nanmean(kinmed_per_fold, axis=0)
        kinonly_r = np.nanmean(kinonly_per_fold, axis=0)
        s2k_r = np.nanmean(s2k_per_fold, axis=0)

    return TwoStageResult(
        direct_r=direct_r,
        kinematics_mediated_r=kinmed_r,
        kinematics_only_r=kinonly_r,
        state_to_kinematics_r=s2k_r,
        n_conditions=n_cond,
        n_timesteps=T,
        n_features=n_feat,
        kin_dim=K,
    )
