"""Reproduce Fig. 5B (extended): DMFC, four RNN classes, and IN.

Primary deliverable for the project (PRD F3). Composes:

* **DMFC curve** — per-timestep linear decoding of the trial endpoint y from
  the reliable-unit population, computed by us via
  :func:`dmfc.analysis.endpoint_decoding.decode_endpoint`. The Zenodo release
  ships per-target r values for DMFC, but not per-timestep curves; the only
  honest way to get one is to run our own decoder on
  ``DMFCData.responses_reliable``.
* **Four RNN class curves** — pre-computed in
  ``offline_rnn_neural_responses_reliable_50.pkl`` as
  ``r_start1_all`` shape ``(100 iter, 90 t)`` per model. Aggregated within
  each ``loss_weight_type`` class on the ``gabor_pca`` subset (the relevant
  one for neural comparison; ``pixel_pca`` is not on the figure).
* **IN curve(s)** — for each ``runs/in_*`` directory, load
  ``hidden_states.npz`` and run the same per-timestep decoder.

All three curves natively live on slightly different time axes (DMFC and IN
at 50 ms bins; RNN at 41 ms bins). Default ``--align start`` interpolates
the RNN's 41 ms grid onto a shared 50 ms axis with t=0 at trial start —
matching the alignment used in the published Fig. 5B (Rajalingham's
``r_start1_all`` is named for "start at 1 timestep" alignment).

CPU-only. Plotting depends on matplotlib only; no torch import here.
"""

from __future__ import annotations

import argparse
import glob
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from dmfc.analysis.endpoint_decoding import (
    DecodingResult,
    decode_endpoint,
    flatten_receivers,
)
from dmfc.envs.conditions import RNN_STEP_MS
from dmfc.rajalingham.load import (
    DEFAULT_DATA_DIR,
    DMFC_BIN_MS,
    DMFCData,
    RNNMetrics,
    load_dmfc_neural,
    load_rnn_metrics,
)
from dmfc.training.config import load_config

# RNN classes in the four-loss order. Labels match the published Fig. 4 axis.
RNN_CLASS_ORDER: tuple[str, ...] = ("mov", "vis-mov", "vis-sim-mov", "sim-mov")
RNN_CLASS_LABELS: dict[str, str] = {
    "mov": "Intercept",
    "vis-mov": "Vis",
    "vis-sim-mov": "Vis+Occ",
    "sim-mov": "Vis&Occ",
}

# Default time axis for the figure: 50 ms bins covering the full DMFC window.
COMMON_BIN_MS: int = DMFC_BIN_MS  # 50
DEFAULT_T_MAX_MS: int = 5000  # 100 × 50 ms

INPUT_REPRESENTATION_FOR_NEURAL: str = "gabor_pca"


@dataclass(frozen=True)
class CurveOnGrid:
    """A decoding curve on the shared time axis."""

    t_ms: np.ndarray  # (n_bins,)
    r_mean: np.ndarray  # (n_bins,) Pearson r per bin
    r_sem: np.ndarray | None  # (n_bins,) or None when there's no across-model spread
    label: str


def common_time_axis(t_max_ms: int = DEFAULT_T_MAX_MS, bin_ms: int = COMMON_BIN_MS) -> np.ndarray:
    return np.arange(0, t_max_ms, bin_ms, dtype=np.float64)


def _interp_curve(curve: np.ndarray, src_grid: np.ndarray, tgt_grid: np.ndarray) -> np.ndarray:
    """Linear interpolation onto ``tgt_grid``; out-of-range bins become NaN."""
    out = np.interp(tgt_grid, src_grid, curve, left=np.nan, right=np.nan)
    return out


def dmfc_curve(
    dmfc: DMFCData,
    t_axis: np.ndarray,
    mask_name: str = "start_end_pad0",
    n_splits: int = 5,
) -> CurveOnGrid:
    """Per-timestep DMFC decoding of the trial endpoint y on the shared time axis.

    The DMFC dataset includes a 300 ms pre-trial fixation/stationary period
    (paper Methods, page 10), so bin 0 is *not* motion onset — bin 6 is.
    We read the motion-onset bin from ``behavioral['t_from_start']`` (which
    equals 0 at motion onset) and shift the source grid accordingly so the
    DMFC curve is aligned with the IN/RNN curves at t=0 = motion onset.
    The IN's hidden_states.npz starts at motion onset (env's integrator
    step 0 is the initial moving state) and the RNN ``r_start1_all`` is
    likewise aligned to start = motion onset.

    The ``ball_final_y`` target is constant within a condition; take any
    finite column. We use the column at the motion-onset bin.
    """
    # (n_units, 79, 100) → (79, 100, n_units) for decode_endpoint.
    states = np.transpose(dmfc.responses, (1, 2, 0)).astype(np.float64)
    mask = np.isfinite(dmfc.masks[mask_name]).astype(bool)

    # Motion-onset bin: first bin where t_from_start >= 0. Same across all
    # 79 conditions (uniform pre-trial of 300 ms).
    t_from_start = np.asarray(dmfc.behavioral["t_from_start"])
    motion_onset_bin = int(np.argmax(t_from_start[0] >= 0))

    endpoint_y = np.asarray(dmfc.behavioral["ball_final_y"][:, motion_onset_bin], dtype=np.float64)

    res = decode_endpoint(states=states, endpoint_y=endpoint_y, valid_mask=mask, n_splits=n_splits)
    src_grid = (np.arange(res.n_timesteps) - motion_onset_bin) * dmfc.bin_ms
    r_on_axis = _interp_curve(res.r, src_grid, t_axis)
    return CurveOnGrid(t_ms=t_axis, r_mean=r_on_axis, r_sem=None, label="DMFC")


def rnn_class_curves(
    rnn: RNNMetrics,
    t_axis: np.ndarray,
    input_representation: str = INPUT_REPRESENTATION_FOR_NEURAL,
) -> list[CurveOnGrid]:
    """One curve per ``loss_weight_type`` class on the shared time axis."""
    df = rnn.df
    sub_repr = df[df["input_representation"] == input_representation]

    curves: list[CurveOnGrid] = []
    for loss_class in RNN_CLASS_ORDER:
        sub = sub_repr[sub_repr["loss_weight_type"] == loss_class]
        per_model_means: list[np.ndarray] = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", r"Mean of empty slice", RuntimeWarning)
            warnings.filterwarnings("ignore", r"Degrees of freedom", RuntimeWarning)
            for fn in sub["filename"]:
                entry = rnn.per_model.get(fn)
                if entry is None:
                    continue
                r_iter_t = entry["r_start1_all"]  # (n_iter, 90)
                per_model_means.append(np.nanmean(r_iter_t, axis=0))
            if not per_model_means:
                continue
            stacked = np.stack(per_model_means, axis=0)  # (n_models, 90)
            class_mean = np.nanmean(stacked, axis=0)
            n_models = stacked.shape[0]
            class_sem = (
                np.nanstd(stacked, axis=0, ddof=1) / np.sqrt(n_models) if n_models > 1 else None
            )

        src_grid = np.arange(rnn.n_timesteps) * RNN_STEP_MS  # 41 ms native
        r_on_axis = _interp_curve(class_mean, src_grid, t_axis)
        sem_on_axis = _interp_curve(class_sem, src_grid, t_axis) if class_sem is not None else None
        label = f"RNN: {RNN_CLASS_LABELS[loss_class]}"
        curves.append(CurveOnGrid(t_ms=t_axis, r_mean=r_on_axis, r_sem=sem_on_axis, label=label))
    return curves


def _in_curve_for_run(run_dir: Path, t_axis: np.ndarray, n_splits: int = 5) -> CurveOnGrid:
    """One IN run → one decoding curve on the shared time axis."""
    cfg = load_config(run_dir / "config.yaml")
    loss_class = cfg.training.loss_variant
    label = (
        f"IN: {RNN_CLASS_LABELS.get(loss_class, loss_class)} (h{cfg.model.effect_dim} s{cfg.seed})"
    )

    npz_path = run_dir / "hidden_states.npz"
    with np.load(npz_path) as data:
        receivers = np.asarray(data["effect_receivers"], dtype=np.float64)
        targets = np.asarray(data["targets"], dtype=np.float64)
        valid_mask = np.asarray(data["valid_mask"], dtype=np.float64)

    states = flatten_receivers(receivers)
    endpoint_y = targets[:, 0, 6]  # ENDPOINT_OUTPUT_INDEX from endpoint_decoding
    res: DecodingResult = decode_endpoint(
        states=states, endpoint_y=endpoint_y, valid_mask=valid_mask, n_splits=n_splits
    )
    src_grid = np.arange(res.n_timesteps) * COMMON_BIN_MS
    r_on_axis = _interp_curve(res.r, src_grid, t_axis)
    return CurveOnGrid(t_ms=t_axis, r_mean=r_on_axis, r_sem=None, label=label)


def in_curves(run_dirs: list[Path], t_axis: np.ndarray) -> list[CurveOnGrid]:
    return [_in_curve_for_run(d, t_axis) for d in run_dirs]


def aggregate_in_curves_by_class(
    run_dirs: list[Path],
    t_axis: np.ndarray,
    n_splits: int = 5,
) -> list[CurveOnGrid]:
    """Per-loss-class IN curves: mean + SEM across all runs in the class.

    Replaces the spaghetti-of-40-lines fig5b plot with one line per loss
    class (the M5 sweep produces 10 runs per class — 2 hidden × 5 seeds —
    so the legend stays manageable and the seed-to-seed variance shows up
    as a SEM band rather than a curve cloud).

    Run dirs grouped by ``config.training.loss_variant``; classes with no
    runs are skipped. Curves with NaN at a given timestep contribute
    nothing there (``np.nanmean`` / ``np.nanstd``) so trial-end volatility
    doesn't bias the mean.
    """
    by_class: dict[str, list[np.ndarray]] = {c: [] for c in RNN_CLASS_ORDER}
    for d in run_dirs:
        cfg = load_config(d / "config.yaml")
        loss_class = cfg.training.loss_variant
        if loss_class not in by_class:
            continue
        curve = _in_curve_for_run(d, t_axis, n_splits=n_splits)
        by_class[loss_class].append(curve.r_mean)

    out: list[CurveOnGrid] = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"Mean of empty slice", RuntimeWarning)
        warnings.filterwarnings("ignore", r"Degrees of freedom", RuntimeWarning)
        for loss_class in RNN_CLASS_ORDER:
            stacked = by_class[loss_class]
            if not stacked:
                continue
            arr = np.stack(stacked, axis=0)  # (n_runs_in_class, T)
            mean = np.nanmean(arr, axis=0)
            n_runs = arr.shape[0]
            sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(n_runs) if n_runs > 1 else None
            label = f"IN: {RNN_CLASS_LABELS[loss_class]} (n={n_runs})"
            out.append(CurveOnGrid(t_ms=t_axis, r_mean=mean, r_sem=sem, label=label))
    return out


def plot_fig5b(
    dmfc: CurveOnGrid,
    rnns: list[CurveOnGrid],
    ins: list[CurveOnGrid],
    out_path: Path,
    title: str = "Endpoint decoding over time (Fig. 5B extended)",
    xlim_ms: tuple[float, float] = (0.0, 1200.0),
) -> None:
    """Render the extended Fig. 5B.

    Color convention (per user direction 2026-04-28):

    * **DMFC** — black, bold.
    * **RNN classes** — grey family, kept since the rapid-vs-slow rise vs
      DMFC is the central comparison this figure exists for.
    * **IN classes** — palette matched to Fig. 4 panel D
      (:data:`IN_CLASS_LINE_PALETTE`): Intercept blue, Vis green, Vis+Occ
      red, Vis&Occ orange. Each line has a translucent ±SEM band when
      ``r_sem`` is set (i.e. for class-aggregated curves with n_runs > 1).

    ``ins`` may be either per-run curves (one line per ``runs/in_*``) or
    per-class aggregates (one mean+SEM curve per loss class — see
    :func:`aggregate_in_curves_by_class`). For the M5 sweep with 40 runs
    the per-class form is the only legible option.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.7))

    # DMFC — black, bold.
    ax.plot(dmfc.t_ms, dmfc.r_mean, color="black", lw=2.5, label=dmfc.label)

    # RNN classes — grey family.
    rnn_palette = ["#9ca3af", "#6b7280", "#4b5563", "#1f2937"]
    for i, c in enumerate(rnns):
        color = rnn_palette[i % len(rnn_palette)]
        ax.plot(c.t_ms, c.r_mean, color=color, lw=1.5, label=c.label)
        if c.r_sem is not None:
            ax.fill_between(
                c.t_ms,
                c.r_mean - c.r_sem,
                c.r_mean + c.r_sem,
                color=color,
                alpha=0.15,
                linewidth=0,
            )

    # IN curves — Fig. 4 palette per loss class. The label format
    # "IN: <Class> (n=...)" exposes the loss class so we can route color.
    for c in ins:
        color = _in_curve_color(c.label)
        ax.plot(c.t_ms, c.r_mean, color=color, lw=2.0, label=c.label)
        if c.r_sem is not None:
            ax.fill_between(
                c.t_ms,
                c.r_mean - c.r_sem,
                c.r_mean + c.r_sem,
                color=color,
                alpha=0.20,
                linewidth=0,
            )

    ax.set_xlabel("Time from trial start (ms)")
    ax.set_ylabel("Endpoint decoding (Pearson r)")
    ax.set_title(title)
    ax.set_xlim(*xlim_ms)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.0, color="black", lw=0.5, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# IN palette resolution — match Fig. 4 panel D


# Per-class IN line color, matched to Fig. 4 panel D (see
# ``IN_CLASS_PALETTE`` in ``reproduce_fig4.py``). Defined here too so this
# module doesn't import from reproduce_fig4 (would be a circular import).
IN_CLASS_LINE_PALETTE: dict[str, str] = {
    "mov": "#1f77b4",
    "vis-mov": "#2ca02c",
    "vis-sim-mov": "#d62728",
    "sim-mov": "#ff7f0e",
}
IN_FALLBACK_COLOR: str = "#1f77b4"


def _in_curve_color(label: str) -> str:
    """Resolve a CurveOnGrid label like 'IN: Vis+Occ (...)' to its class color.

    We look up the human-readable class name (the value of
    :data:`RNN_CLASS_LABELS`) to find the underlying ``loss_weight_type``
    key, then index :data:`IN_CLASS_LINE_PALETTE`. Falls back to blue if
    no match (e.g. if a caller passes a non-standard label).

    Substring matching is order-sensitive: ``"Vis"`` is a substring of
    ``"Vis+Occ"`` and ``"Vis&Occ"``, so we iterate longest-label-first to
    let the more specific names ("Vis&Occ", "Vis+Occ") win before the
    bare "Vis". Exact-match would also work but breaks if a future label
    suffixes the class name with extra metadata.
    """
    label_to_class = sorted(
        ((v, k) for k, v in RNN_CLASS_LABELS.items()),
        key=lambda pair: -len(pair[0]),
    )
    for human, machine in label_to_class:
        if human in label:
            return IN_CLASS_LINE_PALETTE.get(machine, IN_FALLBACK_COLOR)
    return IN_FALLBACK_COLOR


def _expand_run_dirs(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pat in patterns:
        matched = sorted(glob.glob(pat))
        for p in matched:
            d = Path(p)
            if (d / "hidden_states.npz").exists() and (d / "config.yaml").exists():
                out.append(d)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0] if __doc__ else "")
    parser.add_argument(
        "--in-runs",
        nargs="+",
        required=True,
        help="One or more glob patterns for IN run directories (e.g. runs/in_*)",
    )
    parser.add_argument(
        "--rajalingham-data",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory holding the Zenodo release files. Default: data/dmfc",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("figures/fig5b_extended.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--align",
        choices=("start", "onset"),
        default="start",
        help=(
            "Time-axis alignment. 'start' aligns curves at trial start (matches the "
            "published Fig. 5B). 'onset' would re-align each condition to its own "
            "occluder-onset time and is not implemented in this script (requires "
            "per-condition RNN predictions; planned for an M5 follow-up)."
        ),
    )
    parser.add_argument("--t-max-ms", type=int, default=DEFAULT_T_MAX_MS)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument(
        "--xlim-ms",
        type=float,
        nargs=2,
        default=(0.0, 1200.0),
        metavar=("MIN_MS", "MAX_MS"),
        help="Display range for the x-axis (curves are still computed on the full window).",
    )
    parser.add_argument(
        "--in-aggregation",
        choices=("by-class", "per-run"),
        default="by-class",
        help=(
            "How to summarize IN runs in the figure. 'by-class' (default): "
            "one line per loss class with ±SEM band across all runs in the "
            "class — the right view when there are many runs (M5 sweep). "
            "'per-run': one line per run, no aggregation — only legible for "
            "≤4 runs."
        ),
    )
    args = parser.parse_args(argv)

    if args.align == "onset":
        raise NotImplementedError(
            "--align onset is not yet implemented (see module docstring). Use 'start'."
        )

    run_dirs = _expand_run_dirs(args.in_runs)
    if not run_dirs:
        raise SystemExit(f"No IN run directories matched: {args.in_runs}")

    print(f"[fig5b] resolved {len(run_dirs)} IN run dir(s):")
    for d in run_dirs:
        print(f"  - {d}")

    t_axis = common_time_axis(t_max_ms=args.t_max_ms)
    print("[fig5b] loading DMFC neural data ...")
    dmfc_data = load_dmfc_neural(data_dir=args.rajalingham_data)
    print("[fig5b] decoding DMFC per-timestep ...")
    dmfc = dmfc_curve(dmfc_data, t_axis, n_splits=args.n_splits)

    print("[fig5b] loading RNN per-model metrics ...")
    rnn_data = load_rnn_metrics(data_dir=args.rajalingham_data)
    print("[fig5b] aggregating RNN per-class curves ...")
    rnns = rnn_class_curves(rnn_data, t_axis)

    if args.in_aggregation == "by-class":
        print(
            f"[fig5b] decoding IN run(s) and aggregating by loss class ({len(run_dirs)} runs) ..."
        )
        ins = aggregate_in_curves_by_class(run_dirs, t_axis, n_splits=args.n_splits)
    else:
        print("[fig5b] decoding IN run(s) ...")
        ins = in_curves(run_dirs, t_axis)

    print(f"[fig5b] writing figure to {args.out}")
    plot_fig5b(dmfc, rnns, ins, out_path=args.out, xlim_ms=tuple(args.xlim_ms))  # type: ignore[arg-type]
    print("[fig5b] done.")


if __name__ == "__main__":
    main()


__all__: list[Any] = [
    "CurveOnGrid",
    "common_time_axis",
    "dmfc_curve",
    "rnn_class_curves",
    "in_curves",
    "aggregate_in_curves_by_class",
    "plot_fig5b",
    "main",
    "RNN_CLASS_ORDER",
    "RNN_CLASS_LABELS",
    "IN_CLASS_LINE_PALETTE",
]
