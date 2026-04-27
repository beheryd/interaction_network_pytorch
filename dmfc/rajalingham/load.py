"""Pure-I/O adapters over the Rajalingham et al. 2025 Zenodo release.

Three loaders, each isolated to one pickle so a format change at the source only
breaks one function:

* ``load_dmfc_neural`` — ``all_hand_dmfc_dataset_50ms.pkl`` (pooled DMFC population)
* ``load_rnn_metrics`` — ``offline_rnn_neural_responses_reliable_50.pkl`` (per-RNN
  per-timestep r/MAE curves and predictions)
* ``load_decode_dmfc`` — ``decode_*.pkl`` (precomputed decoder results across
  23 behavioral targets)

All loaders return frozen dataclasses backed by plain numpy arrays. No analysis
logic lives here. Consumers should copy arrays before mutating.

Pickle compatibility: the Zenodo files were written under an older pandas, and
plain ``pickle.load`` raises ``ModuleNotFoundError: pandas.core.indexes.numeric``.
``pandas.read_pickle`` handles the shim transparently, so we use it throughout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

DEFAULT_DATA_DIR = Path("data/dmfc")

DMFC_NEURAL_PKL = "all_hand_dmfc_dataset_50ms.pkl"
RNN_METRICS_PKL = "offline_rnn_neural_responses_reliable_50.pkl"
DECODE_PKL = "decode_all_hand_dmfc_occ_start_end_pad0_50ms_0.50_neural_responses_reliable_FactorAnalysis_50.pkl"
RNN_COMPARE_PKL = (
    "rnn_compare_all_hand_dmfc_occ_50ms_neural_responses_reliable_FactorAnalysis_50.pkl"
)

# Bin width used throughout the DMFC dataset and our IN training (matching
# Rajalingham's neural binning). RNN-native bin width is 41 ms — see conditions.py.
DMFC_BIN_MS: int = 50

# Number of timesteps stored per condition in the DMFC dataset.
DMFC_N_TIMESTEPS: int = 100

# Number of canonical evaluation conditions (PONG_BASIC_META_IDX in conditions.py).
N_CONDITIONS: int = 79


@dataclass(frozen=True)
class DMFCData:
    """Pooled DMFC neural population on the 79 evaluation conditions.

    All arrays are organized as ``(..., condition, timestep)`` with timesteps in
    50-ms bins. ``meta_index`` aligns the condition axis with
    ``dmfc.envs.conditions.PONG_BASIC_META_IDX``.
    """

    responses: np.ndarray  # (n_neurons, 79, 100)
    responses_sh1: np.ndarray  # (n_neurons, 79, 100), split-half 1
    responses_sh2: np.ndarray  # (n_neurons, 79, 100), split-half 2
    masks: dict[str, np.ndarray]  # canonical mask names → (79, 100)
    behavioral: dict[str, np.ndarray]  # behavior name → (79, 100)
    meta_index: np.ndarray  # (79,) PONG_BASIC_META_IDX
    bin_ms: int
    n_timesteps: int
    epoch: str  # "occ" (occlusion-aligned recordings)


@dataclass(frozen=True)
class RNNMetrics:
    """Per-RNN time-resolved metrics for the 192 published RNNs.

    ``per_model`` is keyed by the file-path string from ``df['name']``; each
    entry is a dict with ``yp``, ``yt``, ``r_start1_all``, ``mae_start1_all``,
    and the bounce-stratified variants.
    """

    df: pd.DataFrame  # (192, 695)
    per_model: dict[str, dict[str, np.ndarray]]
    n_iterations: int  # 100, the CV-iteration axis of r_start1_all etc.
    n_timesteps: int  # 90, the RNN's native time axis (41 ms bins)


@dataclass(frozen=True)
class DecodeResult:
    """Precomputed DMFC decoder output across 23 behavioral targets.

    The Zenodo pickle bundles 11 ``res_decode`` entries (one per neural-data
    variant or scaling sweep). We expose the full list as ``entries``; most
    callers want ``entries[-1]`` (the last sweep value), but exposing all eleven
    keeps this loader honest about what's on disk.
    """

    beh_targets: list[str]
    entries: list[dict[str, Any]]  # 11 entries; each has r_mu, mae_mu, r_dist, etc.
    decoder_specs: dict[str, Any]
    neural_data_key: str
    n_conditions: int


def _resolve(data_dir: Path | str, filename: str) -> Path:
    path = Path(data_dir) / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Zenodo file not found: {path}\n"
            f"Expected `{filename}` under `{data_dir}` (typically a symlink to the "
            f"unpacked release at https://doi.org/10.5281/zenodo.13952210)."
        )
    return path


# Mask names we expose by default. The Zenodo pickle ships hundreds of "_roll{N}"
# shifted variants used for cross-validated time-shift analyses; analysis code
# rarely needs them, so only the canonical set is included here. Add more by
# passing them in via the loader if needed.
_CANONICAL_MASKS: tuple[str, ...] = (
    "pretrial_pad0",
    "start_end_pad0",
    "start_occ_pad0",
    "occ_end_pad0",
    "f_pad0",
    "occ_pad0",
    "start_pad0",
    "half_pad0",
)


def load_dmfc_neural(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    epoch: str = "occ",
    extra_masks: tuple[str, ...] = (),
) -> DMFCData:
    """Load the pooled DMFC neural dataset for the 79 conditions.

    Parameters
    ----------
    data_dir
        Directory containing the Zenodo files. Defaults to ``data/dmfc``.
    epoch
        ``"occ"`` (default) for occlusion-aligned trials. ``"vis"`` is also
        present in the pickle but only ``"occ"`` carries the reliable-neuron
        split-halves needed for noise-adjusted analyses.
    extra_masks
        Additional mask names beyond the canonical set (e.g. specific
        ``"_roll{N}"`` shifts) to include in the returned ``masks`` dict.
    """
    path = _resolve(data_dir, DMFC_NEURAL_PKL)
    raw = cast(dict[str, Any], pd.read_pickle(path))

    if epoch not in raw["neural_responses_reliable"]:
        raise KeyError(
            f"epoch={epoch!r} not in dataset; available: {list(raw['neural_responses_reliable'])}"
        )

    responses = np.asarray(raw["neural_responses_reliable"][epoch])
    responses_sh1 = np.asarray(raw["neural_responses_reliable_sh1"][epoch])
    responses_sh2 = np.asarray(raw["neural_responses_reliable_sh2"][epoch])

    mask_names = list(_CANONICAL_MASKS) + [m for m in extra_masks if m not in _CANONICAL_MASKS]
    masks = {
        name: np.asarray(raw["masks"][epoch][name])
        for name in mask_names
        if name in raw["masks"][epoch]
    }

    behavioral = {name: np.asarray(arr) for name, arr in raw["behavioral_responses"][epoch].items()}

    # Lazy import to avoid a hard cycle if envs ever depends on rajalingham.
    from dmfc.envs.conditions import PONG_BASIC_META_IDX

    return DMFCData(
        responses=responses,
        responses_sh1=responses_sh1,
        responses_sh2=responses_sh2,
        masks=masks,
        behavioral=behavioral,
        meta_index=np.asarray(PONG_BASIC_META_IDX, dtype=np.int64),
        bin_ms=DMFC_BIN_MS,
        n_timesteps=DMFC_N_TIMESTEPS,
        epoch=epoch,
    )


def load_rnn_metrics(data_dir: Path | str = DEFAULT_DATA_DIR) -> RNNMetrics:
    """Load per-RNN per-timestep r/MAE curves and predictions for all 192 RNNs.

    Returned ``per_model`` is keyed by the model's checkpoint path; the
    ``df`` column ``name`` lists those keys. Each value has shapes:

    * ``yp``, ``yt``: ``(100, 79, 90)`` — per-iteration predictions and targets
    * ``r_start1_all``, ``mae_start1_all``: ``(100, 90)`` — endpoint-decoding
      curves on the 79 conditions, mirrored by ``_bounce`` / ``_no_bounce``
      condition-stratified variants
    """
    path = _resolve(data_dir, RNN_METRICS_PKL)
    raw = cast(dict[str, Any], pd.read_pickle(path))

    df = raw["df"]
    all_metrics = raw["all_metrics"]
    per_model: dict[str, dict[str, np.ndarray]] = {
        key: {sub_k: np.asarray(sub_v) for sub_k, sub_v in entry.items()}
        for key, entry in all_metrics.items()
    }

    sample = next(iter(per_model.values()))
    n_iter, n_t = sample["r_start1_all"].shape
    return RNNMetrics(df=df, per_model=per_model, n_iterations=int(n_iter), n_timesteps=int(n_t))


def load_rnn_compare(data_dir: Path | str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load the per-RNN representational-comparison summary DataFrame.

    Carries the published Fig. 4D Neural Consistency column
    ``pdist_similarity_occ_end_pad0_euclidean_r_xy_n_sb`` (and many other
    epoch/metric variants) that ``load_rnn_metrics``' DataFrame does not
    expose. Shape ``(192, 90)``, integer-indexed; row order is identical to
    ``load_rnn_metrics().df`` and rows can be matched on the shared
    ``filename`` column.
    """
    path = _resolve(data_dir, RNN_COMPARE_PKL)
    raw = cast(dict[str, Any], pd.read_pickle(path))
    return cast(pd.DataFrame, raw["summary"])


def load_decode_dmfc(data_dir: Path | str = DEFAULT_DATA_DIR) -> DecodeResult:
    """Load the precomputed DMFC decoder results across 23 behavioral targets."""
    path = _resolve(data_dir, DECODE_PKL)
    raw = cast(dict[str, Any], pd.read_pickle(path))

    return DecodeResult(
        beh_targets=list(raw["beh_to_decode"]),
        entries=list(raw["res_decode"]),
        decoder_specs=dict(raw["decoder_specs"]),
        # The pickle key has a trailing space — preserve as-is when reading.
        neural_data_key=str(raw["neural_data_to_use "]),
        n_conditions=int(raw["ncond"]),
    )
