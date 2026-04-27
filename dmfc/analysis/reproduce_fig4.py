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
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dmfc.analysis.endpoint_decoding import flatten_receivers
from dmfc.analysis.neural_consistency import neural_consistency_from_states
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
    """One IN run reduced to a single (NC, SI) point."""

    loss_class: str  # one of RNN_CLASS_ORDER
    nc: float
    si: float
    label: str
    n_hidden: int
    seed: int
    run_dir: Path


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


def in_point_for_run(
    run_dir: Path,
    dmfc: DMFCData,
    mask_name: str = NC_MASK_NAME,
    si_k: int = SI_K_FOLDS,
    si_seed: int = 0,
) -> INPoint:
    """Compute (NC, SI) for one IN run on the canonical 79 conditions."""
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

    label = (
        f"IN: {RNN_CLASS_LABELS.get(loss_class, loss_class)} (h{cfg.model.effect_dim} s{cfg.seed})"
    )
    return INPoint(
        loss_class=loss_class,
        nc=float(nc_result.r_xy_n_sb),
        si=float(si_result.si),
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


def in_swarm(run_dirs: list[Path], dmfc: DMFCData) -> list[INPoint]:
    out: list[INPoint] = []
    for d in run_dirs:
        if _has_diverged_states(d):
            print(f"[fig4] skipping diverged run (NaN states): {d}")
            continue
        out.append(in_point_for_run(d, dmfc))
    return out


def _dmfc_motion_onset_bin(dmfc: DMFCData) -> int:
    """Bin where DMFC's ``t_from_start`` first turns non-negative (motion onset)."""
    t_from_start = np.asarray(dmfc.behavioral["t_from_start"])
    return int(np.argmax(t_from_start[0] >= 0))


_RNN_CLASS_X: dict[str, int] = {c: i for i, c in enumerate(RNN_CLASS_ORDER)}


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
    args = parser.parse_args(argv)

    run_dirs = _expand_run_dirs(args.in_runs)
    if not run_dirs:
        raise SystemExit(f"No IN run directories matched: {args.in_runs}")

    print(f"[fig4] resolved {len(run_dirs)} IN run dir(s):")
    for d in run_dirs:
        print(f"  - {d}")

    print("[fig4] loading DMFC neural data ...")
    dmfc = load_dmfc_neural(data_dir=args.rajalingham_data)

    print("[fig4] loading RNN swarm (df + rnn_compare) ...")
    rnn_metrics = load_rnn_metrics(data_dir=args.rajalingham_data)
    rnn_compare = load_rnn_compare(data_dir=args.rajalingham_data)
    rnn_df = rnn_swarm(rnn_metrics.df, rnn_compare, args.input_representation)
    print(f"[fig4] RNN swarm: {len(rnn_df)} models on {args.input_representation!r}")

    print("[fig4] computing IN points ...")
    with warnings.catch_warnings():
        # KFold can issue a benign warning when a fold has 0 test cells for a coord.
        warnings.filterwarnings("ignore", r"Mean of empty slice", RuntimeWarning)
        ins = in_swarm(run_dirs, dmfc)
    for p in ins:
        print(f"  {p.label}: NC={p.nc:.4f}  SI={p.si:.4f} deg")

    print(f"[fig4] writing figure to {args.out}")
    plot_fig4(rnn_df, ins, out_path=args.out)
    print("[fig4] done.")


if __name__ == "__main__":
    main()


__all__: list[Any] = [
    "INPoint",
    "rnn_swarm",
    "in_point_for_run",
    "in_swarm",
    "plot_fig4",
    "main",
    "NC_COLUMN",
    "SI_COLUMN",
]
