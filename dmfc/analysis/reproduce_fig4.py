"""Reproduce Fig. 4D + 4E (extended): RNN swarm + IN points.

Secondary deliverable for the project (PRD F4). Two side-by-side panels:

* **Panel D — Neural Consistency** (RSA, Spearman-Brown noise-corrected,
  ``occ_end_pad0`` mask). RNN values are read straight from
  ``rnn_compare_*.pkl``'s summary DataFrame (column
  ``pdist_similarity_occ_end_pad0_euclidean_r_xy_n_sb``); IN values are
  computed via :func:`dmfc.analysis.neural_consistency.neural_consistency_from_states`.
* **Panel E — Simulation Index** (k=2 OLS decoder of occluded ball
  position from model states). RNN values are read from the existing
  ``offline_rnn_*.pkl`` DataFrame (column
  ``decode_vis-sim_to_sim_index_mae_k2``); IN values are computed via
  :func:`dmfc.analysis.simulation_index.simulation_index`.

Both panels x-stratify by ``loss_weight_type`` (mov / vis-mov / vis-sim-mov
/ sim-mov, mapped to the published Intercept / Vis / Vis+Occ / Vis&Occ
labels). The RNN swarm is restricted to the ``gabor_pca`` subset (96 of
192 published RNNs), matching the input representation used for neural
comparison throughout the paper.

Time-axis alignment for the IN side: the IN's hidden_states.npz starts
at motion onset (env step 0); the DMFC dataset has a 300 ms pre-trial
pad so DMFC bin 6 is motion onset. We slice DMFC to
``[:, motion_onset_bin : motion_onset_bin + T_in]`` so model and neural
states share the same ``(n_cond, T)`` lattice.

CPU-only.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dmfc.analysis.endpoint_decoding import flatten_receivers
from dmfc.analysis.neural_consistency import (
    neural_consistency,
    neural_consistency_from_states,
)
from dmfc.analysis.rdm import compute_rdm
from dmfc.analysis.reproduce_fig5b import (
    RNN_CLASS_LABELS,
    RNN_CLASS_ORDER,
    _expand_run_dirs,
)
from dmfc.analysis.simulation_index import simulation_index
from dmfc.rajalingham.load import (
    DEFAULT_DATA_DIR,
    DMFCData,
    load_dmfc_neural,
    load_rnn_compare,
    load_rnn_metrics,
)
from dmfc.training.config import load_config

INPUT_REPRESENTATION_FOR_NEURAL: str = "gabor_pca"
NC_COLUMN: str = "pdist_similarity_occ_end_pad0_euclidean_r_xy_n_sb"
SI_COLUMN: str = "decode_vis-sim_to_sim_index_mae_k2"
NC_MASK_NAME: str = "occ_end_pad0"
SI_K_FOLDS: int = 2
# The IN's targets[:, :, 0] and [:, :, 1] hold true ball x/y regardless of
# supervision (see SCRATCHPAD M3 closeout § "Pinned design choices").
BALL_X_INDEX: int = 0
BALL_Y_INDEX: int = 1


@dataclass(frozen=True)
class INPoint:
    """One IN run reduced to a (NC, SI, task-MAE) triple."""

    loss_class: str  # one of RNN_CLASS_ORDER
    nc: float
    si: float
    task_mae: float
    label: str
    n_hidden: int
    seed: int
    run_dir: Path


def task_performance_mae(run_dir: Path) -> float:
    """Mean absolute error of the IN's predicted intercept y, in degrees.

    Operationalizes paper Fig. 4F's "task performance / mean absolute error"
    for the IN: at the last valid timestep of each condition, compare
    output[6] (predicted final intercept y) to targets[6] (y_f_oracle), then
    average the absolute error across the 79 conditions. The IN's paddle is
    held at y=0 throughout training (no controller; SCRATCHPAD M3 closeout),
    so the prediction-vs-oracle MAE is the right "how well does the model
    solve the task" scalar.
    """
    with np.load(run_dir / "hidden_states.npz") as data:
        outputs = np.asarray(data["outputs"], dtype=np.float64)
        targets = np.asarray(data["targets"], dtype=np.float64)
        valid_mask = np.asarray(data["valid_mask"], dtype=bool)

    n_cond = outputs.shape[0]
    last_valid_t = (valid_mask.sum(axis=1) - 1).astype(int)  # (n_cond,)
    rows = np.arange(n_cond)
    pred = outputs[rows, last_valid_t, 6]
    true = targets[rows, last_valid_t, 6]
    return float(np.mean(np.abs(pred - true)))


def rnn_swarm(
    df_offline: pd.DataFrame,
    df_compare: pd.DataFrame,
    input_representation: str = INPUT_REPRESENTATION_FOR_NEURAL,
) -> pd.DataFrame:
    """Per-RNN (NC, SI) DataFrame restricted to one input representation.

    The two source DataFrames are row-aligned (verified live: identical
    ``filename`` columns at row 0..2). We merge on ``filename`` so a future
    re-export from Zenodo with reordered rows still works.
    """
    if SI_COLUMN not in df_offline.columns:
        raise KeyError(f"SI column {SI_COLUMN!r} not in df_offline")
    if NC_COLUMN not in df_compare.columns:
        raise KeyError(f"NC column {NC_COLUMN!r} not in df_compare")
    if "filename" not in df_offline.columns or "filename" not in df_compare.columns:
        raise KeyError("expected 'filename' on both DataFrames")

    left = df_offline[["filename", "loss_weight_type", "input_representation", SI_COLUMN]]
    right = df_compare[["filename", NC_COLUMN]]
    merged = left.merge(right, on="filename", how="inner")

    sub = merged[merged["input_representation"] == input_representation].copy()
    sub = sub.rename(columns={SI_COLUMN: "si", NC_COLUMN: "nc", "loss_weight_type": "loss_class"})
    return sub.reset_index(drop=True)


@dataclass(frozen=True)
class NeuralRDMCache:
    """Pre-computed DMFC neural RDMs for one ``(t_in, mask_name)`` slice.

    The DMFC neural RDMs (full + 2 split-halves) are identical across all
    40 IN runs in the M5 sweep — same DMFC dataset, same motion-onset
    alignment, same ``occ_end_pad0`` mask, same ``T_in=72``. Computing
    them inside ``in_point_for_run`` (via ``neural_consistency_from_states``)
    repeats ~12 GFLOPs of pdist work 40 times for no reason. Caching
    cuts ~80% of the wall clock on Fig. 4.

    Constructed via :func:`compute_neural_rdm_cache`.
    """

    rdm: np.ndarray  # full neural RDM (n_pairs,)
    rdm_sh1: np.ndarray  # split-half 1
    rdm_sh2: np.ndarray  # split-half 2
    mask: np.ndarray  # boolean (n_cond, t_in) — passed downstream for the model RDM
    t_in: int
    mask_name: str


def compute_neural_rdm_cache(
    dmfc: DMFCData,
    t_in: int,
    mask_name: str = NC_MASK_NAME,
    metric: Literal["euclidean", "corr"] = "euclidean",
) -> NeuralRDMCache:
    """Compute the 3 DMFC neural RDMs once for a given ``t_in`` window.

    Slices DMFC at ``[motion_onset_bin : motion_onset_bin + t_in]`` to
    match the IN's temporal extent (M3-era IN trains on env trajectories
    integrated to bin ``t_in=72``). The full + split-half RDMs returned
    are flat vectors over the same ``(n_cond, t_in)`` cell pairs.
    """
    motion_onset_bin = _dmfc_motion_onset_bin(dmfc)
    sl = slice(motion_onset_bin, motion_onset_bin + t_in)

    nc_neural = np.transpose(dmfc.responses[:, :, sl], (1, 2, 0))
    nc_neural_sh1 = np.transpose(dmfc.responses_sh1[:, :, sl], (1, 2, 0))
    nc_neural_sh2 = np.transpose(dmfc.responses_sh2[:, :, sl], (1, 2, 0))
    nc_mask = dmfc.masks[mask_name][:, sl].astype(bool)

    rdm = compute_rdm(nc_neural, nc_mask, metric=metric, mask_name=mask_name).rdm
    rdm_sh1 = compute_rdm(nc_neural_sh1, nc_mask, metric=metric).rdm
    rdm_sh2 = compute_rdm(nc_neural_sh2, nc_mask, metric=metric).rdm

    return NeuralRDMCache(
        rdm=rdm,
        rdm_sh1=rdm_sh1,
        rdm_sh2=rdm_sh2,
        mask=nc_mask,
        t_in=t_in,
        mask_name=mask_name,
    )


def in_point_for_run(
    run_dir: Path,
    dmfc: DMFCData,
    mask_name: str = NC_MASK_NAME,
    si_k: int = SI_K_FOLDS,
    si_seed: int = 0,
    neural_rdm_cache: NeuralRDMCache | None = None,
) -> INPoint:
    """Compute (NC, SI) for one IN run on the canonical 79 conditions.

    When ``neural_rdm_cache`` is provided we reuse the pre-computed DMFC
    RDMs rather than recomputing — this is the ~5× speedup that turns a
    1-hour fig4 render into a 10-minute one. The cache is constructed
    via :func:`compute_neural_rdm_cache`. Without a cache (default), we
    fall back to the original :func:`neural_consistency_from_states`
    path, which is slower but doesn't require the caller to know t_in
    ahead of time.
    """
    cfg = load_config(run_dir / "config.yaml")
    loss_class = cfg.training.loss_variant

    with np.load(run_dir / "hidden_states.npz") as data:
        receivers = np.asarray(data["effect_receivers"], dtype=np.float64)
        targets = np.asarray(data["targets"], dtype=np.float64)
        visible_mask = np.asarray(data["visible_mask"], dtype=bool)
        valid_mask = np.asarray(data["valid_mask"], dtype=bool)

    states = flatten_receivers(receivers)  # (79, T_in, 2*eff)
    n_cond, t_in, _ = states.shape

    # NC — RSA against DMFC, occ_end_pad0 mask.
    if neural_rdm_cache is not None:
        if neural_rdm_cache.t_in != t_in:
            raise ValueError(
                f"NeuralRDMCache built for t_in={neural_rdm_cache.t_in} but "
                f"IN run has t_in={t_in}; rebuild the cache for this t_in"
            )
        if neural_rdm_cache.mask_name != mask_name:
            raise ValueError(
                f"NeuralRDMCache built for mask {neural_rdm_cache.mask_name!r} "
                f"but call asked for {mask_name!r}"
            )
        model_rdm = compute_rdm(
            states, neural_rdm_cache.mask, metric="euclidean", mask_name=mask_name
        ).rdm
        nc_result = neural_consistency(
            model_rdm=model_rdm,
            neural_rdm=neural_rdm_cache.rdm,
            neural_rdm_sh1=neural_rdm_cache.rdm_sh1,
            neural_rdm_sh2=neural_rdm_cache.rdm_sh2,
        )
    else:
        motion_onset_bin = _dmfc_motion_onset_bin(dmfc)
        nc_neural = dmfc.responses[:, :, motion_onset_bin : motion_onset_bin + t_in]
        nc_neural_sh1 = dmfc.responses_sh1[:, :, motion_onset_bin : motion_onset_bin + t_in]
        nc_neural_sh2 = dmfc.responses_sh2[:, :, motion_onset_bin : motion_onset_bin + t_in]
        nc_mask = dmfc.masks[mask_name][:, motion_onset_bin : motion_onset_bin + t_in]
        if nc_mask.shape != (n_cond, t_in):
            raise ValueError(
                f"DMFC mask slice {nc_mask.shape} does not match IN states {n_cond, t_in}; "
                f"motion_onset_bin={motion_onset_bin}, dmfc has {dmfc.n_timesteps} bins"
            )

        nc_result = neural_consistency_from_states(
            model_states=states,
            neural_responses=nc_neural,
            neural_responses_sh1=nc_neural_sh1,
            neural_responses_sh2=nc_neural_sh2,
            mask=nc_mask.astype(bool),
            metric="euclidean",
            mask_name=mask_name,
        )

    # SI — k=2 OLS decoder of true ball (x, y) on occluded held-out cells.
    ball_xy = np.stack([targets[:, :, BALL_X_INDEX], targets[:, :, BALL_Y_INDEX]], axis=-1)
    train_mask = valid_mask
    test_mask = valid_mask & (~visible_mask)
    si_result = simulation_index(
        states=states,
        ball_xy=ball_xy,
        train_mask=train_mask,
        test_mask=test_mask,
        k=si_k,
        seed=si_seed,
    )

    task_mae = task_performance_mae(run_dir)

    label = (
        f"IN: {RNN_CLASS_LABELS.get(loss_class, loss_class)} (h{cfg.model.effect_dim} s{cfg.seed})"
    )
    return INPoint(
        loss_class=loss_class,
        nc=float(nc_result.r_xy_n_sb),
        si=float(si_result.si),
        task_mae=task_mae,
        label=label,
        n_hidden=int(cfg.model.effect_dim),
        seed=int(cfg.seed),
        run_dir=run_dir,
    )


def _has_diverged_states(run_dir: Path) -> bool:
    """True if ``hidden_states.npz`` contains NaN — i.e. the training diverged.

    The Intercept loss variant has a known recurrent-stability failure mode
    on default lr/clip (SCRATCHPAD M4 progress 5); the seed=0 ``mov`` pilot
    on disk has all-NaN effect_receivers. We skip such runs rather than
    crash sklearn deep in the SI decoder.
    """
    with np.load(run_dir / "hidden_states.npz") as data:
        return bool(np.isnan(np.asarray(data["effect_receivers"])).any())


def in_swarm(
    run_dirs: list[Path],
    dmfc: DMFCData,
    mask_name: str = NC_MASK_NAME,
) -> list[INPoint]:
    """Compute (NC, SI, task_MAE) for each ``runs/in_*`` dir.

    Builds the DMFC neural RDM cache once on the first non-diverged run's
    ``t_in`` and reuses it for the whole list. All M5 runs share
    ``t_in=72`` so the cache hits 100% of the time; the only cost is
    computing the RDMs once (~40s) instead of 40 times.
    """
    cache: NeuralRDMCache | None = None
    out: list[INPoint] = []
    for d in run_dirs:
        if _has_diverged_states(d):
            print(f"[fig4] skipping diverged run (NaN states): {d}")
            continue
        # Build the cache lazily on the first run so we know t_in.
        if cache is None:
            with np.load(d / "hidden_states.npz") as data:
                t_in = int(data["effect_receivers"].shape[1])
            print(
                f"[fig4] precomputing DMFC neural RDM cache for t_in={t_in}, "
                f"mask={mask_name!r} (one-time cost) ..."
            )
            cache = compute_neural_rdm_cache(dmfc, t_in=t_in, mask_name=mask_name)
        out.append(in_point_for_run(d, dmfc, mask_name=mask_name, neural_rdm_cache=cache))
    return out


def _dmfc_motion_onset_bin(dmfc: DMFCData) -> int:
    """Bin where DMFC's ``t_from_start`` first turns non-negative (motion onset)."""
    t_from_start = np.asarray(dmfc.behavioral["t_from_start"])
    return int(np.argmax(t_from_start[0] >= 0))


_RNN_CLASS_X: dict[str, int] = {c: i for i, c in enumerate(RNN_CLASS_ORDER)}

# Per-class palette for the paper-replica IN-only figure (per user direction
# 2026-04-28). Mirrors the paper's class-color encoding: Intercept=blue,
# Vis=green, Vis+Occ=red, Vis&Occ=orange.
IN_CLASS_PALETTE: dict[str, str] = {
    "mov": "#1f77b4",  # Intercept — blue
    "vis-mov": "#2ca02c",  # Vis — green
    "vis-sim-mov": "#d62728",  # Vis+Occ — red
    "sim-mov": "#ff7f0e",  # Vis&Occ — orange
}

# Paper Fig. 4 axis limits, read from the figure (panel D y, panels E/F x and y).
PAPER_FIG4_LIMITS: dict[str, tuple[float, float]] = {
    "panel_d_y": (0.05, 0.4),
    "panel_e_x": (5.0, 1.0),  # flipped: 5 left, 1 right
    "panel_e_y": (0.1, 0.4),
    "panel_f_x": (4.0, 1.0),  # flipped: 4 left, 1 right
    "panel_f_y": (0.1, 0.4),
}


def plot_fig4(
    rnn_df: pd.DataFrame,
    ins: list[INPoint],
    out_path: Path,
    title: str = "Fig. 4 extended (NC + SI)",
    jitter: float = 0.18,
    rng_seed: int = 0,
) -> None:
    """Two-panel scatter: NC on the left, SI on the right."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    rng = np.random.default_rng(rng_seed)

    # X positions: one per loss class, jittered for the swarm.
    def x_for(class_name: str, n: int) -> np.ndarray:
        x0 = _RNN_CLASS_X[class_name]
        return x0 + rng.uniform(-jitter, jitter, size=n)

    # Per-class palette for IN points (matches reproduce_fig5b's blue family).
    in_palette = {
        "mov": "#1d4ed8",
        "vis-mov": "#2563eb",
        "vis-sim-mov": "#3b82f6",
        "sim-mov": "#60a5fa",
    }

    for ax, value_col, ylabel, panel_title in (
        (axes[0], "nc", "Neural Consistency (r_xy_n_sb)", "D. Neural Consistency"),
        (axes[1], "si", "Simulation Index (MAE, deg)", "E. Simulation Index"),
    ):
        # RNN swarm (light grey, jittered).
        for class_name in RNN_CLASS_ORDER:
            sub = rnn_df[rnn_df["loss_class"] == class_name]
            if sub.empty:
                continue
            xs = x_for(class_name, len(sub))
            ax.scatter(xs, sub[value_col].to_numpy(), s=18, color="#9ca3af", alpha=0.6, lw=0)

        # IN points (one marker per run, larger and on top).
        for p in ins:
            x0 = _RNN_CLASS_X.get(p.loss_class)
            if x0 is None:
                continue
            v = p.nc if value_col == "nc" else p.si
            ax.scatter(
                [x0],
                [v],
                s=110,
                color=in_palette.get(p.loss_class, "#1d4ed8"),
                edgecolors="black",
                lw=1.0,
                marker="D",
                zorder=5,
            )

        ax.set_xticks(list(_RNN_CLASS_X.values()))
        ax.set_xticklabels([RNN_CLASS_LABELS[c] for c in RNN_CLASS_ORDER], rotation=20)
        ax.set_xlabel("Loss class")
        ax.set_ylabel(ylabel)
        ax.set_title(panel_title)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    # SI is "lower is better" (MAE in degrees); annotate so a reader can't miss it.
    axes[1].text(
        0.02,
        0.98,
        "lower = better",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#374151",
    )

    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _hidden_to_size(n_hidden: int, small: float = 60.0, large: float = 130.0) -> float:
    """Marker size as a function of hidden-unit count (10 → small, 20 → large).

    Per user direction 2026-04-28: hidden size is encoded by point size
    rather than marker shape, so the figure stays single-marker-family while
    still distinguishing the two sweep values Rajalingham used.
    """
    if n_hidden <= 10:
        return small
    return large


def _fit_line(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares line through (xs, ys); returns (xs_sorted, y_pred)."""
    if len(xs) < 2:
        return xs, ys
    finite = np.isfinite(xs) & np.isfinite(ys)
    if finite.sum() < 2:
        return xs, ys
    xs_f = xs[finite]
    ys_f = ys[finite]
    slope, intercept = np.polyfit(xs_f, ys_f, 1)
    xs_sorted = np.sort(xs_f)
    y_pred = slope * xs_sorted + intercept
    return xs_sorted, y_pred


def _auto_extend_xlim(
    paper_lim: tuple[float, float],
    data_xs: np.ndarray,
    pad_frac: float = 0.05,
) -> tuple[float, float]:
    """Extend ``paper_lim`` if ``data_xs`` falls outside; preserve flip orientation.

    Per user direction 2026-04-28: keep paper-strict limits when IN data fits
    inside the paper's RNN range, but silently extend either bound when IN
    points spill out so the figure stays informative even when IN strongly
    outperforms the paper RNNs (e.g. task-MAE near 0° vs RNN range 1–4°).

    The paper panels are flipped (e.g. ``(5, 1)`` for SI), so ``paper_lim[0]``
    may be larger than ``paper_lim[1]``. We compute axis-domain min/max from
    the data, then assemble the returned tuple in the same orientation.
    """
    finite = data_xs[np.isfinite(data_xs)]
    if len(finite) == 0:
        return paper_lim
    paper_lo = min(paper_lim)
    paper_hi = max(paper_lim)
    data_lo = float(finite.min())
    data_hi = float(finite.max())
    span = max(paper_hi - paper_lo, 1e-9)
    pad = pad_frac * span
    new_lo = min(paper_lo, data_lo - pad)
    new_hi = max(paper_hi, data_hi + pad)
    flipped = paper_lim[0] > paper_lim[1]
    return (new_hi, new_lo) if flipped else (new_lo, new_hi)


def plot_fig4_paper_replica(
    ins: list[INPoint],
    out_path: Path,
    title: str = "Fig. 4 (IN, paper-replica)",
    jitter: float = 0.18,
    rng_seed: int = 0,
    show_regression: bool = True,
) -> None:
    """Three-panel D / E / F replica of paper Fig. 4 — IN-only, no RNN data.

    Per user direction 2026-04-28: replicate the paper's panel layout with
    matched colors (Intercept=blue, Vis=green, Vis+Occ=red, Vis&Occ=orange)
    and matched axis limits, but display only the IN swarm — RNN data is
    deliberately omitted.

    * **Panel D** — NC swarm by loss class, y∈[0.05, 0.4], jittered points
      with one median bar per class.
    * **Panel E** — NC vs SI, x flipped (5→1 left-to-right) so increasing
      simulation capacity moves rightward, matching the paper.
    * **Panel F** — NC vs Task MAE, x flipped (4→1) for the same reason.

    Hidden-unit size is encoded by point size (10 = small, 20 = large) per
    user direction, so the figure stays single-marker-family.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
    rng = np.random.default_rng(rng_seed)

    by_class: dict[str, list[INPoint]] = {c: [] for c in RNN_CLASS_ORDER}
    for p in ins:
        if p.loss_class in by_class:
            by_class[p.loss_class].append(p)

    # Panel D — NC swarm.
    ax_d = axes[0]
    for class_name in RNN_CLASS_ORDER:
        pts = by_class[class_name]
        if not pts:
            continue
        x0 = _RNN_CLASS_X[class_name]
        xs = x0 + rng.uniform(-jitter, jitter, size=len(pts))
        ys = np.array([p.nc for p in pts])
        sizes = np.array([_hidden_to_size(p.n_hidden) for p in pts])
        color = IN_CLASS_PALETTE[class_name]
        ax_d.scatter(xs, ys, s=sizes, color=color, alpha=0.85, edgecolors="white", lw=0.6)
        # Translucent class-median bar (mirrors paper's box overlay).
        median = float(np.median(ys))
        ax_d.hlines(median, x0 - 0.32, x0 + 0.32, colors=color, lw=2.2, alpha=0.7)
    ax_d.set_xticks(list(_RNN_CLASS_X.values()))
    ax_d.set_xticklabels([RNN_CLASS_LABELS[c] for c in RNN_CLASS_ORDER], rotation=20)
    ax_d.set_ylabel("Neural Consistency Score")
    ax_d.set_ylim(*_auto_extend_xlim(PAPER_FIG4_LIMITS["panel_d_y"], np.array([p.nc for p in ins])))
    ax_d.set_title("D")
    ax_d.grid(axis="y", linestyle="--", alpha=0.3)

    # Helper to draw a class-colored scatter on panels E/F.
    def _scatter_panel(ax: Any, x_attr: str, y_attr: str) -> tuple[np.ndarray, np.ndarray]:
        xs_all: list[float] = []
        ys_all: list[float] = []
        for class_name in RNN_CLASS_ORDER:
            pts = by_class[class_name]
            if not pts:
                continue
            xs = np.array([getattr(p, x_attr) for p in pts])
            ys = np.array([getattr(p, y_attr) for p in pts])
            sizes = np.array([_hidden_to_size(p.n_hidden) for p in pts])
            color = IN_CLASS_PALETTE[class_name]
            ax.scatter(xs, ys, s=sizes, color=color, alpha=0.85, edgecolors="white", lw=0.6)
            xs_all.extend(xs.tolist())
            ys_all.extend(ys.tolist())
        return np.asarray(xs_all), np.asarray(ys_all)

    # Panel E — NC vs SI.
    ax_e = axes[1]
    xs_e, ys_e = _scatter_panel(ax_e, "si", "nc")
    if show_regression and len(xs_e) >= 2:
        xf, yf = _fit_line(xs_e, ys_e)
        ax_e.plot(xf, yf, ls="--", color="#374151", lw=1.5, alpha=0.8)
    ax_e.set_xlabel("Simulation Index (SI)\nVis-occ. transfer error (°)")
    ax_e.set_ylabel("Neural Consistency Score")
    ax_e.set_xlim(*_auto_extend_xlim(PAPER_FIG4_LIMITS["panel_e_x"], xs_e))
    ax_e.set_ylim(*_auto_extend_xlim(PAPER_FIG4_LIMITS["panel_e_y"], ys_e))
    ax_e.set_title("E")
    ax_e.grid(linestyle="--", alpha=0.3)

    # Panel F — NC vs Task MAE.
    ax_f = axes[2]
    xs_f_data, ys_f_data = _scatter_panel(ax_f, "task_mae", "nc")
    if show_regression and len(xs_f_data) >= 2:
        xf, yf = _fit_line(xs_f_data, ys_f_data)
        ax_f.plot(xf, yf, ls="--", color="#374151", lw=1.5, alpha=0.8)
    ax_f.set_xlabel("Task performance\nMean absolute error (°)")
    ax_f.set_ylabel("Neural Consistency Score")
    ax_f.set_xlim(*_auto_extend_xlim(PAPER_FIG4_LIMITS["panel_f_x"], xs_f_data))
    ax_f.set_ylim(*_auto_extend_xlim(PAPER_FIG4_LIMITS["panel_f_y"], ys_f_data))
    ax_f.set_title("F")
    ax_f.grid(linestyle="--", alpha=0.3)

    # Compact legend showing class colors + size convention.
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            lw=0,
            color=IN_CLASS_PALETTE[c],
            markersize=8,
            label=f"IN: {RNN_CLASS_LABELS[c]}",
        )
        for c in RNN_CLASS_ORDER
    ] + [
        plt.Line2D([0], [0], marker="o", lw=0, color="#6b7280", markersize=6, label="hidden=10"),
        plt.Line2D([0], [0], marker="o", lw=0, color="#6b7280", markersize=10, label="hidden=20"),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.02)
    )

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0] if __doc__ else "")
    parser.add_argument(
        "--in-runs",
        nargs="+",
        required=True,
        help="One or more glob patterns for IN run directories (e.g. runs/in_*).",
    )
    parser.add_argument(
        "--rajalingham-data",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory holding the Zenodo release files. Default: data/dmfc.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("figures/fig4_extended.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--input-representation",
        default=INPUT_REPRESENTATION_FOR_NEURAL,
        help="RNN input representation to subset on (default: gabor_pca).",
    )
    parser.add_argument(
        "--style",
        choices=("extended", "paper-replica"),
        default="paper-replica",
        help=(
            "extended: 2-panel NC+SI with RNN swarm overlay (M4 deliverable). "
            "paper-replica: 3-panel D/E/F IN-only matching paper Fig. 4 layout (default)."
        ),
    )
    args = parser.parse_args(argv)

    run_dirs = _expand_run_dirs(args.in_runs)
    if not run_dirs:
        raise SystemExit(f"No IN run directories matched: {args.in_runs}")

    print(f"[fig4] resolved {len(run_dirs)} IN run dir(s):")
    for d in run_dirs:
        print(f"  - {d}")

    print("[fig4] loading DMFC neural data ...")
    dmfc = load_dmfc_neural(data_dir=args.rajalingham_data)

    print("[fig4] computing IN points ...")
    with warnings.catch_warnings():
        # KFold can issue a benign warning when a fold has 0 test cells for a coord.
        warnings.filterwarnings("ignore", r"Mean of empty slice", RuntimeWarning)
        ins = in_swarm(run_dirs, dmfc)
    for p in ins:
        print(f"  {p.label}: NC={p.nc:.4f}  SI={p.si:.4f} deg  task_MAE={p.task_mae:.4f} deg")

    if args.style == "extended":
        print("[fig4] loading RNN swarm (df + rnn_compare) ...")
        rnn_metrics = load_rnn_metrics(data_dir=args.rajalingham_data)
        rnn_compare = load_rnn_compare(data_dir=args.rajalingham_data)
        rnn_df = rnn_swarm(rnn_metrics.df, rnn_compare, args.input_representation)
        print(f"[fig4] RNN swarm: {len(rnn_df)} models on {args.input_representation!r}")
        print(f"[fig4] writing extended figure to {args.out}")
        plot_fig4(rnn_df, ins, out_path=args.out)
    else:
        print(f"[fig4] writing paper-replica figure to {args.out}")
        plot_fig4_paper_replica(ins, out_path=args.out)
    print("[fig4] done.")


if __name__ == "__main__":
    main()


__all__: list[Any] = [
    "INPoint",
    "NeuralRDMCache",
    "compute_neural_rdm_cache",
    "rnn_swarm",
    "in_point_for_run",
    "in_swarm",
    "plot_fig4",
    "plot_fig4_paper_replica",
    "task_performance_mae",
    "main",
    "NC_COLUMN",
    "SI_COLUMN",
    "IN_CLASS_PALETTE",
    "PAPER_FIG4_LIMITS",
]
