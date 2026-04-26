"""Canonical 79-condition Mental Pong evaluation set.

Adapter over Rajalingham et al.'s `valid_meta_sample_full.pkl`. The 79 condition
indices are the `PONG_BASIC_META_IDX` list from their `code/utils/phys_utils.py`;
order is preserved so per-condition arrays line up across our env, their neural
data, and their RNN outputs.

All kinematic quantities use the visual-degree (MWK) frame: x, y in [-10, +10],
velocities in degrees per RNN-step (RNN-step = 41 ms).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Native RNN timestep used throughout the Zenodo release; trajectory integers
# (`t_f`, `t_occ`) are in these units. See `phys_utils.py:54`.
RNN_STEP_MS: int = 41

# Frozen 79-condition order from Rajalingham `phys_utils.PONG_BASIC_META_IDX`.
PONG_BASIC_META_IDX: tuple[int, ...] = (
    29337,
    55062,
    58244,
    59920,
    68220,
    72780,
    77046,
    77944,
    90751,
    93053,
    99902,
    105802,
    118879,
    122957,
    126479,
    158705,
    159613,
    163248,
    167848,
    179553,
    182748,
    183171,
    184642,
    187617,
    197632,
    199564,
    204372,
    204678,
    206128,
    217817,
    218791,
    226533,
    232533,
    241919,
    242380,
    244437,
    245662,
    248856,
    251629,
    258471,
    269179,
    270569,
    273073,
    273515,
    287936,
    297652,
    301600,
    320246,
    325813,
    328386,
    330797,
    332165,
    340684,
    351612,
    356164,
    356765,
    370777,
    377134,
    379400,
    381514,
    388571,
    394672,
    396358,
    400995,
    401673,
    406460,
    413189,
    413513,
    420020,
    428615,
    435203,
    448412,
    454477,
    456038,
    460651,
    463174,
    464785,
    465431,
    469268,
)


@dataclass(frozen=True)
class ConditionSpec:
    """Per-condition stimulus parameters and validation oracles.

    Coordinates are in visual degrees (MWK frame). Velocities are deg per RNN-step.
    """

    meta_index: int
    x0: float
    y0: float
    dx: float
    dy: float
    t_f_steps: int
    t_occ_steps: int
    y_occ_oracle: float
    y_f_oracle: float
    n_bounce: int

    @property
    def t_f_ms(self) -> int:
        return self.t_f_steps * RNN_STEP_MS

    @property
    def t_occ_ms(self) -> int:
        return self.t_occ_steps * RNN_STEP_MS


DEFAULT_META_PICKLE = Path("data/dmfc/valid_meta_sample_full.pkl")


def load_conditions(meta_pickle_path: Path | str = DEFAULT_META_PICKLE) -> list[ConditionSpec]:
    """Load the canonical 79 conditions in `PONG_BASIC_META_IDX` order."""
    meta = pd.read_pickle(Path(meta_pickle_path))
    by_index = meta.set_index("meta_index")
    specs: list[ConditionSpec] = []
    for idx in PONG_BASIC_META_IDX:
        row = by_index.loc[idx]
        specs.append(
            ConditionSpec(
                meta_index=int(idx),
                x0=float(row["x0_mwk"]),
                y0=float(row["y0_mwk"]),
                dx=float(row["dx_rnn"]),
                dy=float(row["dy_rnn"]),
                t_f_steps=int(row["t_f"]),
                t_occ_steps=int(row["t_occ"]),
                y_occ_oracle=float(row["y_occ_rnn_mwk"]),
                y_f_oracle=float(row["y_f_rnn_mwk"]),
                n_bounce=int(row["n_bounce_correct"]),
            )
        )
    return specs
