"""Driver: emit the statistical numbers that back ``fig5b_full_sweep.png`` and ``fig4_paper_replica.png``.

The two figures are descriptive on their own; this script produces the
numerical summaries the writeup needs to match the reporting style used
in Rajalingham et al. 2025 for the homologous panels.

What this driver computes:

* **Fig. 5B — DMFC vs shuffled control** (paper Fig. 5B caption: *"DMFC
  predictions were significantly more accurate than the shuffled control
  for all time points following t = 250 ms (p < 10^-10, two-sided
  permutation test)"*). We re-run the DMFC endpoint-y decoding on the
  real labels and on ``N_PERMUTATIONS`` permutations of the endpoint-y
  vector across the 79 conditions, then compute a per-timestep two-sided
  p-value from the permutation distribution. The first time the
  per-timestep p falls below the paper's threshold gives the "p < … for
  all t > X ms" claim. The smallest p resolvable from a permutation null
  is ``1 / N_PERMUTATIONS`` so we faithfully report a floor rather than
  fabricating ``p < 10^-10``.

* **Fig. 4 — NC ~ SI and NC ~ task-performance** (paper Supp. Fig. S6B/E:
  *"Raw R² and Partial R²"* of NC explained by intermediate-states
  decode performance and by overall task performance). We compute these
  on the IN swarm (40 runs) using OLS. Partial R² uses the simple form
  ``R²_full - R²_base``, matching ``dmfc.analysis.stats.partial_r2``.

Output
------
* ``<out_dir>/m5_stats.txt`` — human-readable summary table.
* ``<out_dir>/m5_stats.json`` — same numbers as JSON for programmatic use.

CPU-only.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold

from dmfc.analysis.endpoint_decoding import decode_endpoint
from dmfc.analysis.reproduce_fig4 import (
    _expand_run_dirs,
    in_swarm,
)
from dmfc.analysis.stats import partial_r2
from dmfc.rajalingham.load import (
    DEFAULT_DATA_DIR,
    load_dmfc_neural,
)

# Paper-aligned defaults.
DEFAULT_WINDOW_MS: tuple[int, int] = (0, 1200)
DEFAULT_N_PERMUTATIONS: int = 10_000
DEFAULT_PERM_SEED: int = 0
DEFAULT_DECODE_SPLITS: int = 5
DMFC_MASK_NAME: str = "start_end_pad0"


@dataclass(frozen=True)
class Fig5BPermResult:
    """Per-timestep permutation-test result for the DMFC endpoint decoder."""

    t_ms: np.ndarray  # (T,)
    r_real: np.ndarray  # (T,)
    rmse_real: np.ndarray  # (T,)
    p_two_sided: np.ndarray  # (T,) per-timestep p-value, two-sided
    first_t_below_alpha: float  # ms; first t at which p stays below alpha through window end
    alpha: float
    n_permutations: int
    p_floor: float  # 1 / n_permutations — smallest resolvable p from this null

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "t_ms": self.t_ms.tolist(),
            "r_real": _nan_to_none(self.r_real).tolist(),
            "rmse_real": _nan_to_none(self.rmse_real).tolist(),
            "p_two_sided": _nan_to_none(self.p_two_sided).tolist(),
            "first_t_below_alpha_ms": (
                None
                if not np.isfinite(self.first_t_below_alpha)
                else float(self.first_t_below_alpha)
            ),
            "alpha": self.alpha,
            "n_permutations": self.n_permutations,
            "p_floor": self.p_floor,
        }


@dataclass(frozen=True)
class Fig4R2Result:
    """Raw R² + Partial R² of NC ~ SI and NC ~ task-MAE on the IN swarm."""

    n_in_runs: int
    raw_r2_nc_from_si: float
    partial_r2_nc_from_si_after_mae: float  # SI's unique contribution beyond task-MAE
    raw_r2_nc_from_mae: float
    partial_r2_nc_from_mae_after_si: float  # task-MAE's unique contribution beyond SI
    raw_r2_nc_from_both: float

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class M5StatsBundle:
    """Everything emitted by this driver."""

    fig5b: Fig5BPermResult
    fig4: Fig4R2Result
    in_run_dirs: list[str] = field(default_factory=list)


def _nan_to_none(arr: np.ndarray) -> np.ndarray:
    """Pass-through; JSON serialization handles NaN→null via ``allow_nan``."""
    return arr


def _decode_real_and_perm(
    states: np.ndarray,
    endpoint_y: np.ndarray,
    valid_mask: np.ndarray,
    n_permutations: int,
    n_splits: int,
    perm_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the real decoder + ``n_permutations`` shuffled-label decoders.

    Returns
    -------
    r_real
        ``(T,)`` — Pearson r at each timestep on the true labels.
    rmse_real
        ``(T,)`` — RMSE at each timestep on the true labels.
    r_perm
        ``(n_permutations, T)`` — Pearson r at each timestep under
        per-condition label permutations of ``endpoint_y``. Permutation
        is across the 79 conditions (preserving the per-condition group
        structure ``GroupKFold`` relies on); within a permutation, every
        timestep sees the same shuffled label vector.

    Implementation note
    -------------------
    The real curve is computed via :func:`decode_endpoint` (full per-fold
    detail incl. RMSE). The permutation null is computed via a fast path
    that factorizes the per-(fold, t) train-feature matrix **once**, then
    reuses the factorization across all permutations — the OLS coefficients
    for a permuted label vector are just ``pinv(X_train) @ y_perm``, and
    the predictions are ``X_test @ coefs``. This collapses the ~17 GFLOPs
    of pseudoinverse work down to one factorization per (fold, t), giving
    a ~100× speedup over re-running ``decode_endpoint`` from scratch
    per permutation.
    """
    rng = np.random.default_rng(perm_seed)
    n_cond, T, _ = states.shape

    # Real: same code path as the figure curves — keeps the real numbers
    # bit-for-bit identical to whatever the figure script produced.
    real = decode_endpoint(
        states=states,
        endpoint_y=endpoint_y,
        valid_mask=valid_mask,
        n_splits=n_splits,
    )

    # Permutation null via cached pseudoinverses.
    splitter = GroupKFold(n_splits=n_splits)
    groups = np.arange(n_cond)
    valid = valid_mask.astype(bool) if valid_mask is not None else np.ones((n_cond, T), dtype=bool)

    # Pre-generate the permutation matrix; (n_permutations, n_cond).
    perms = np.empty((n_permutations, n_cond), dtype=np.int64)
    for k in range(n_permutations):
        perms[k] = rng.permutation(n_cond)
    y_all_perm = endpoint_y[perms]  # (n_permutations, n_cond)

    # r_perm_sum[t] aggregates per-fold r values; we average across folds
    # at the end to match decode_endpoint's reporting (mean across folds,
    # NaNs ignored).
    r_per_fold_perm = np.full((n_splits, n_permutations, T), np.nan, dtype=np.float64)

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(states, endpoint_y, groups)):
        # Train/test labels for all permutations at once.
        y_train_perm = y_all_perm[:, train_idx]  # (n_permutations, n_train)
        y_test_perm = y_all_perm[:, test_idx]  # (n_permutations, n_test)

        for t in range(T):
            m_train = valid[train_idx, t]
            m_test = valid[test_idx, t]
            if m_train.sum() < 2 or m_test.sum() < 2:
                continue
            X_train = states[train_idx, t, :][m_train]  # (n_tr_valid, n_feat)
            X_test = states[test_idx, t, :][m_test]  # (n_te_valid, n_feat)

            # Center features and build pseudoinverse once for this (fold, t).
            xt_mean = X_train.mean(axis=0)
            Xc = X_train - xt_mean
            try:
                pinv = np.linalg.pinv(Xc)
            except np.linalg.LinAlgError:
                continue

            # Predictions = (X_test - xt_mean) @ pinv @ y_train_centered + y_train_mean
            # Vectorized across permutations.
            yt = y_train_perm[:, m_train]  # (n_perm, n_tr_valid)
            yt_mean = yt.mean(axis=1, keepdims=True)  # (n_perm, 1)
            yt_c = yt - yt_mean
            coefs = yt_c @ pinv.T  # (n_perm, n_feat)
            X_test_c = X_test - xt_mean  # (n_te_valid, n_feat)
            preds = coefs @ X_test_c.T + yt_mean  # (n_perm, n_te_valid)

            y_te = y_test_perm[:, m_test]  # (n_perm, n_te_valid)

            # Per-permutation Pearson r between preds and held-out perm labels.
            r_per_fold_perm[fold_idx, :, t] = _row_pearson(preds, y_te)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"Mean of empty slice", RuntimeWarning)
        r_perm = np.nanmean(r_per_fold_perm, axis=0)  # (n_permutations, T)

    return real.r, real.rmse, r_perm


def _row_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-row Pearson r of two 2-D arrays of the same shape.

    Args:
        a, b: ``(n_rows, n_cols)``.

    Returns:
        ``(n_rows,)`` — Pearson r computed per row. Rows where either
        side has zero variance are NaN.
    """
    am = a - a.mean(axis=1, keepdims=True)
    bm = b - b.mean(axis=1, keepdims=True)
    num = np.sum(am * bm, axis=1)
    denom = np.sqrt(np.sum(am * am, axis=1) * np.sum(bm * bm, axis=1))
    out = np.full(a.shape[0], np.nan, dtype=np.float64)
    nz = denom > 0
    out[nz] = num[nz] / denom[nz]
    return out


def _two_sided_p(real: np.ndarray, perm: np.ndarray, p_floor: float) -> np.ndarray:
    """Two-sided permutation p-value, per timestep, on |statistic| against |null|.

    Computes ``p_t = mean(|perm[:, t]| >= |real[t]|)`` with the standard
    ``(c+1)/(n+1)`` smoothing so a strictly larger real statistic still
    receives a non-zero p.
    """
    T = real.shape[0]
    out = np.full(T, np.nan, dtype=np.float64)
    for t in range(T):
        r = real[t]
        if not np.isfinite(r):
            continue
        col = perm[:, t]
        col = col[np.isfinite(col)]
        if col.size == 0:
            continue
        c = int(np.sum(np.abs(col) >= abs(r)))
        n = int(col.size)
        out[t] = max(p_floor, (c + 1.0) / (n + 1.0))
    return out


def _first_t_persistently_below(p: np.ndarray, t_ms: np.ndarray, alpha: float) -> float:
    """First ``t_ms`` after which ``p`` stays strictly below ``alpha``.

    Returns ``+inf`` if no such t exists. Mirrors the paper's "for all
    time points following t = X ms" framing — we want the earliest X
    such that no later in-window bin violates the threshold.
    """
    finite = np.isfinite(p)
    if not finite.any():
        return float("inf")
    T = p.shape[0]
    for i in range(T):
        if not finite[i]:
            continue
        tail = p[i:]
        tail_finite = np.isfinite(tail)
        if not tail_finite.any():
            continue
        if np.all((~tail_finite) | (tail < alpha)):
            return float(t_ms[i])
    return float("inf")


def fig5b_permutation_test(
    data_dir: Path = DEFAULT_DATA_DIR,
    window_ms: tuple[int, int] = DEFAULT_WINDOW_MS,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    perm_seed: int = DEFAULT_PERM_SEED,
    n_splits: int = DEFAULT_DECODE_SPLITS,
    alpha: float = 1e-4,
    mask_name: str = DMFC_MASK_NAME,
) -> Fig5BPermResult:
    """DMFC endpoint-y decoder vs shuffled-control permutation null.

    Mirrors the test caption in paper Fig. 5B. The DMFC time axis is
    aligned to motion onset (paper Methods + ``reproduce_fig5b.dmfc_curve``
    handle the 300 ms pre-trial pad), and the per-timestep p-values are
    reported on that motion-aligned grid restricted to ``window_ms``.
    """
    dmfc = load_dmfc_neural(data_dir=data_dir)

    states = np.transpose(dmfc.responses, (1, 2, 0)).astype(np.float64)  # (79, T_full, n_units)
    mask = np.isfinite(dmfc.masks[mask_name]).astype(bool)

    t_from_start = np.asarray(dmfc.behavioral["t_from_start"])
    motion_onset_bin = int(np.argmax(t_from_start[0] >= 0))
    endpoint_y = np.asarray(dmfc.behavioral["ball_final_y"][:, motion_onset_bin], dtype=np.float64)

    print(
        f"[fig5b-perm] DMFC states {states.shape}, motion_onset_bin={motion_onset_bin}, "
        f"running real + {n_permutations} permutations ..."
    )
    r_real, rmse_real, r_perm = _decode_real_and_perm(
        states=states,
        endpoint_y=endpoint_y,
        valid_mask=mask,
        n_permutations=n_permutations,
        n_splits=n_splits,
        perm_seed=perm_seed,
    )

    p_floor = 1.0 / float(n_permutations + 1)
    p_full = _two_sided_p(r_real, r_perm, p_floor=p_floor)

    # Restrict to the paper's reporting window, on the motion-onset-aligned grid.
    full_grid_ms = (np.arange(states.shape[1]) - motion_onset_bin) * dmfc.bin_ms
    in_win = (full_grid_ms >= window_ms[0]) & (full_grid_ms <= window_ms[1])
    t_ms = full_grid_ms[in_win].astype(np.float64)
    p_in_win = p_full[in_win]
    r_in_win = r_real[in_win]
    rmse_in_win = rmse_real[in_win]

    first_t = _first_t_persistently_below(p_in_win, t_ms, alpha=alpha)

    return Fig5BPermResult(
        t_ms=t_ms,
        r_real=r_in_win,
        rmse_real=rmse_in_win,
        p_two_sided=p_in_win,
        first_t_below_alpha=first_t,
        alpha=alpha,
        n_permutations=n_permutations,
        p_floor=p_floor,
    )


def fig4_r2_summary(in_run_dirs: list[Path], data_dir: Path = DEFAULT_DATA_DIR) -> Fig4R2Result:
    """Raw R² + Partial R² for NC ~ SI and NC ~ task-MAE on the IN swarm.

    Mirrors paper Supp. Fig. S6B/E, with the IN points (40 runs after
    diverged-run skipping) standing in for the RNN points the paper uses
    in that figure.
    """
    print(f"[fig4-r2] loading DMFC + computing IN swarm over {len(in_run_dirs)} run dir(s) ...")
    dmfc = load_dmfc_neural(data_dir=data_dir)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"Mean of empty slice", RuntimeWarning)
        ins = in_swarm(in_run_dirs, dmfc)
    if len(ins) < 4:
        raise ValueError(f"need at least 4 non-diverged IN runs for OLS, got {len(ins)}")

    nc = np.array([p.nc for p in ins], dtype=np.float64)
    si = np.array([p.si for p in ins], dtype=np.float64)
    mae = np.array([p.task_mae for p in ins], dtype=np.float64)

    raw_r2_si = _ols_r2(nc, si.reshape(-1, 1))
    raw_r2_mae = _ols_r2(nc, mae.reshape(-1, 1))
    raw_r2_both = _ols_r2(nc, np.column_stack([si, mae]))

    partial_si = partial_r2(nc, base_predictors=mae, extra_predictors=si)
    partial_mae = partial_r2(nc, base_predictors=si, extra_predictors=mae)

    return Fig4R2Result(
        n_in_runs=len(ins),
        raw_r2_nc_from_si=float(raw_r2_si),
        partial_r2_nc_from_si_after_mae=float(partial_si),
        raw_r2_nc_from_mae=float(raw_r2_mae),
        partial_r2_nc_from_mae_after_si=float(partial_mae),
        raw_r2_nc_from_both=float(raw_r2_both),
    )


def _ols_r2(y: np.ndarray, X: np.ndarray) -> float:
    """OLS R² of ``y`` on ``X`` (intercept included). Drops non-finite rows."""
    from sklearn.linear_model import LinearRegression

    finite = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if finite.sum() < X.shape[1] + 2:
        return float("nan")
    return float(LinearRegression().fit(X[finite], y[finite]).score(X[finite], y[finite]))


def format_text_report(bundle: M5StatsBundle) -> str:
    """Render the bundle as a plain-text table for the writeup."""
    f5 = bundle.fig5b
    f4 = bundle.fig4
    lines: list[str] = []
    lines.append("M5 statistical summary")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Fig. 5B — DMFC endpoint decoder vs shuffled control")
    lines.append("-" * 60)
    lines.append(f"  Permutations         : {f5.n_permutations}")
    lines.append(f"  Permutation p floor  : {f5.p_floor:.2e}")
    lines.append(f"  Alpha threshold      : {f5.alpha:.2e}")
    if np.isfinite(f5.first_t_below_alpha):
        lines.append(
            f"  First t with p<alpha and stays below for the rest of the window: "
            f"{f5.first_t_below_alpha:.0f} ms"
        )
    else:
        lines.append("  No t in window had p < alpha sustained to window end.")
    finite_p = f5.p_two_sided[np.isfinite(f5.p_two_sided)]
    if finite_p.size:
        lines.append(
            f"  Per-timestep p (in window): min={finite_p.min():.2e}, "
            f"median={np.median(finite_p):.2e}, max={finite_p.max():.2e}"
        )
    finite_r = f5.r_real[np.isfinite(f5.r_real)]
    if finite_r.size:
        lines.append(
            f"  Real DMFC r over window  : min={finite_r.min():.3f}, "
            f"median={np.median(finite_r):.3f}, max={finite_r.max():.3f}"
        )
    lines.append("")
    lines.append("Fig. 4 — Raw R² and Partial R² of NC on the IN swarm")
    lines.append("-" * 60)
    lines.append(f"  N (non-diverged IN runs)            : {f4.n_in_runs}")
    lines.append(f"  Raw R² of NC ~ SI                   : {f4.raw_r2_nc_from_si:.4f}")
    lines.append(f"  Raw R² of NC ~ task MAE             : {f4.raw_r2_nc_from_mae:.4f}")
    lines.append(f"  Raw R² of NC ~ (SI + task MAE)      : {f4.raw_r2_nc_from_both:.4f}")
    lines.append(
        f"  Partial R² of SI given task MAE     : {f4.partial_r2_nc_from_si_after_mae:.4f}"
    )
    lines.append(
        f"  Partial R² of task MAE given SI     : {f4.partial_r2_nc_from_mae_after_si:.4f}"
    )
    lines.append("")
    lines.append(f"IN run dirs consumed: {len(bundle.in_run_dirs)}")
    return "\n".join(lines) + "\n"


def write_outputs(bundle: M5StatsBundle, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / "m5_stats.txt"
    json_path = out_dir / "m5_stats.json"
    txt_path.write_text(format_text_report(bundle))
    payload = {
        "fig5b": bundle.fig5b.to_jsonable(),
        "fig4": bundle.fig4.to_jsonable(),
        "in_run_dirs": bundle.in_run_dirs,
    }
    with json_path.open("w") as fh:
        json.dump(payload, fh, indent=2, allow_nan=True)
    return txt_path, json_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0] if __doc__ else "")
    parser.add_argument(
        "--in-runs",
        nargs="+",
        required=True,
        help="One or more glob patterns for IN run directories (e.g. runs_m5/in_*).",
    )
    parser.add_argument(
        "--rajalingham-data",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory holding the Zenodo release files. Default: data/dmfc.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures"),
        help="Directory for m5_stats.txt and m5_stats.json. Default: figures/",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=DEFAULT_N_PERMUTATIONS,
        help=f"Number of label permutations for Fig. 5B test (default {DEFAULT_N_PERMUTATIONS}).",
    )
    parser.add_argument(
        "--perm-seed",
        type=int,
        default=DEFAULT_PERM_SEED,
        help="RNG seed for the permutation null (default 0).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1e-4,
        help=(
            "Significance threshold for the 'first t persistently below alpha' "
            "summary stat. Default 1e-4. The smallest resolvable p is "
            "1 / (n_permutations + 1)."
        ),
    )
    parser.add_argument(
        "--window-ms",
        type=int,
        nargs=2,
        default=list(DEFAULT_WINDOW_MS),
        metavar=("MIN", "MAX"),
        help="Reporting window in ms relative to motion onset (default 0 1200).",
    )
    parser.add_argument(
        "--skip-fig5b",
        action="store_true",
        help="Skip the Fig. 5B permutation test (Fig. 4 R² only).",
    )
    args = parser.parse_args(argv)

    in_run_dirs = _expand_run_dirs(args.in_runs)
    if not in_run_dirs:
        raise SystemExit(f"No IN run directories matched: {args.in_runs}")

    if args.skip_fig5b:
        f5: Fig5BPermResult = Fig5BPermResult(
            t_ms=np.zeros(0),
            r_real=np.zeros(0),
            rmse_real=np.zeros(0),
            p_two_sided=np.zeros(0),
            first_t_below_alpha=float("inf"),
            alpha=args.alpha,
            n_permutations=0,
            p_floor=float("inf"),
        )
    else:
        f5 = fig5b_permutation_test(
            data_dir=args.rajalingham_data,
            window_ms=tuple(args.window_ms),  # type: ignore[arg-type]
            n_permutations=args.n_permutations,
            perm_seed=args.perm_seed,
            alpha=args.alpha,
        )

    f4 = fig4_r2_summary(in_run_dirs, data_dir=args.rajalingham_data)

    bundle = M5StatsBundle(fig5b=f5, fig4=f4, in_run_dirs=[str(d) for d in in_run_dirs])
    txt_path, json_path = write_outputs(bundle, out_dir=args.out_dir)
    print(f"[stats] wrote {txt_path}")
    print(f"[stats] wrote {json_path}")
    print()
    print(format_text_report(bundle))


if __name__ == "__main__":
    main()


__all__: list[Any] = [
    "Fig5BPermResult",
    "Fig4R2Result",
    "M5StatsBundle",
    "fig5b_permutation_test",
    "fig4_r2_summary",
    "format_text_report",
    "write_outputs",
    "main",
]
