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


def plot_fig5b(
    dmfc: CurveOnGrid,
    rnns: list[CurveOnGrid],
    ins: list[CurveOnGrid],
    out_path: Path,
    title: str = "Endpoint decoding over time (Fig. 5B extended)",
    xlim_ms: tuple[float, float] = (0.0, 1200.0),
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    # DMFC: bold orange
    ax.plot(dmfc.t_ms, dmfc.r_mean, color="#d97706", lw=2.5, label=dmfc.label)

    # RNN classes: greys with shaded SEM bands
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
                alpha=0.18,
                linewidth=0,
            )

    # IN runs: blue family
    in_palette = ["#1d4ed8", "#2563eb", "#3b82f6", "#60a5fa", "#93c5fd"]
    for i, c in enumerate(ins):
        color = in_palette[i % len(in_palette)]
        ax.plot(c.t_ms, c.r_mean, color=color, lw=2.0, label=c.label)

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
    "plot_fig5b",
    "main",
    "RNN_CLASS_ORDER",
    "RNN_CLASS_LABELS",
]
