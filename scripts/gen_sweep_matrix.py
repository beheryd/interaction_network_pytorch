"""Generate scripts/sweep_matrix.csv — the M5 experimental matrix.

The 40-run sweep per PLANNING.md is fixed at:
    4 loss variants × 2 hidden sizes (10, 20) × 5 seeds = 40 runs

Per-variant HP overrides:
    mov (Intercept) requires grad_clip_norm=0.5 to converge across all 5 seeds
        (validated 2026-04-28 in the Intercept HP pass; default 1.0 NaN-diverged
        seed=0 by step ~800). Other variants stay at the default 1.0.

Run from repo root:
    python scripts/gen_sweep_matrix.py > scripts/sweep_matrix.csv
"""

from __future__ import annotations

import csv
import sys
from itertools import product

# (variant, config_filename) — the 4 base YAMLs already exist under configs/
VARIANTS: list[tuple[str, str]] = [
    ("mov", "configs/in_intercept.yaml"),
    ("vis-mov", "configs/in_vis.yaml"),
    ("vis-sim-mov", "configs/in_vis_occ.yaml"),
    ("sim-mov", "configs/in_vis_and_occ.yaml"),
]
HIDDEN_SIZES: list[int] = [10, 20]
SEEDS: list[int] = [0, 1, 2, 3, 4]


def main() -> int:
    writer = csv.writer(sys.stdout)
    writer.writerow(["task_id", "variant", "config", "hidden", "seed", "grad_clip_norm"])
    for task_id, ((variant, config), hidden, seed) in enumerate(
        product(VARIANTS, HIDDEN_SIZES, SEEDS)
    ):
        grad_clip_norm = 0.5 if variant == "mov" else 1.0
        writer.writerow([task_id, variant, config, hidden, seed, grad_clip_norm])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
