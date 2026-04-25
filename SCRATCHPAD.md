# Development scratchpad

- Use this file to keep notes on ongoing development work.
- Open problems marked with [ ]
- Fixed problems marked with [x]

## NOTES

## Milestone 1 findings — Rajalingham Zenodo release inspection (2026-04-22)

The Zenodo release is at `/Users/david/Downloads/MentalPong`. Key findings below.

### [x] Data format

Primary format is **Python pickle (.pkl)**. MATLAB .mat versions of the neural datasets are also
present but the analysis pipeline uses the .pkl files throughout.

```
data/
  all_hand_dmfc_dataset_50ms.pkl        # pooled neural (Perle + Mahler), 50ms bins, 1.2 GB
  perle_hand_dmfc_dataset_50ms.pkl      # Monkey Perle only, 1.0 GB
  mahler_hand_dmfc_dataset_50ms.pkl     # Monkey Mahler only, 228 MB
  offline_rnn_neural_responses_reliable_50.pkl   # all 192 RNN models, 2.1 GB
  valid_meta_sample_full.pkl            # 79 condition metadata, 72 KB
  decode_*.pkl                          # decoding results (100 CV splits), 1.3–1.8 GB each
  offline_*_neural_responses_reliable_50.pkl     # offline analysis results
  rnn_compare_*.pkl                     # RNN-vs-DMFC RSA
```

### [x] Per-RNN Fig. 5B curves — individually released (not just aggregated)

All 192 RNN models have individual per-timestep predictions in
`offline_rnn_neural_responses_reliable_50.pkl`:

```python
res_rnn['all_metrics'][model_key]['r_start1_all']  # shape (100 splits, 79 conditions, 90 timesteps)
res_rnn['all_metrics'][model_key]['mae_start1_all'] # same shape
res_rnn['state'][model_key]['data_neur_nxcxt']      # hidden states (n_hidden, 79, 90)
res_rnn['df']                                        # DataFrame, 192 rows × 695 columns
```

This means our Fig. 5B comparison can use per-RNN individual curves, not just mean±SEM.

### [x] Hidden-unit sweep values — CORRECTED (2026-04-24)

**CORRECTION**: Earlier note said "Exactly 3 values: 10, 20, 40" — this was wrong.
Actual: `res['df']['n_hidden'].unique()` = `[10.0, 20.0]` — **only 2 hidden sizes in the Zenodo release**.

Corrected 192-model breakdown:
- **4 loss types × 2 architectures (GRU, LSTM) × 2 n_hidden (10, 20) × 2 input types (gabor_pca, pixel_pca) × 6 seeds = 192**
- For our neural comparison, the `gabor_pca` subset (96 models) is the relevant reference.
- PLANNING.md experimental matrix has been updated accordingly.

### [x] RNN architecture

- **Type**: GRU and LSTM (both present in release)
- **Input dim**: 100 (Gabor-filtered visual frames → PCA → 100D for `gabor_pca` models;
  pixel features → PCA for `pixel_pca` models)
- **Output dim**: 7 (indices 0–1 vis-sim, 2–3 vis, 4–5 sim, 6 final position)
- **Input types in data**: `gabor_pca` (relevant for neural comparison), `pixel_pca`

### [x] Loss-type label mapping — CORRECTED (2026-04-24)

**CORRECTION**: Earlier note proposed `vis-sim`→Intercept etc. — this was wrong.

Actual `loss_weight_type` values confirmed from `res['df']['loss_weight_type'].unique()`:

```
['mov', 'sim-mov', 'vis-mov', 'vis-sim-mov']
```

All 4 types include `mov` (final-position supervision); variants add supervision on visible/occluded:

| loss_weight_type | What is supervised | Rajalingham class |
|---|---|---|
| `mov` | final position only (output index 6) | **Intercept** |
| `vis-mov` | visible epoch (indices [2,3]) + final | **Vis** |
| `vis-sim-mov` | vis+occ jointly (indices [0,1]) + final | **Vis+Occ** |
| `sim-mov` | occluded epoch (indices [4,5]) + final | **Vis&Occ** |

Mapping inferred from `rnn_comparisons_2021.py` plot order `['mov','vis-mov','vis-sim-mov','sim-mov']`
matching Fig. 4 left→right (Intercept, Vis, Vis+Occ, Vis&Occ). Verify from paper supplementary
before IN training begins.

For our IN configs, the 4 loss variants supervise:

| IN config | Supervised outputs | Output indices |
|---|---|---|
| `in_intercept` | final position only | [6] |
| `in_vis` | visible epoch + final | [2,3] + [6] |
| `in_vis_occ` | vis+occ joint + final | [0,1] + [6] |
| `in_vis_and_occ` | occluded epoch + final | [4,5] + [6] |

### [x] RNN input — NOT raw pixels

RNNs receive **100-dimensional PCA-compressed Gabor features**, not raw pixels. This is important
for our input-spec decision (PRD NG3 / A3):
- We are using `(ball_x, ball_y, visible_flag, paddle_y, paddle_vy)` — 5 scalars
- This matches the *information content* of their 100D Gabor input (ball position + visibility),
  but is explicit rather than learned from pixels.
- The writeup must clearly acknowledge this difference and defend it as an architectural isolation.

### [x] Trial timing — CORRECTION to TASKS.md

TASKS.md Milestone 2 says "16.7 ms/step". **This is wrong.** Actual specs:
- **Neural data**: 50 ms bins, 100 timepoints per trial → ~5 s total
- **RNN timestep**: ~41 ms/step (confirmed: `scaling_factor = 41` in phys_utils.py), 90 timesteps → ~3.7 s total
- The 16.7 ms figure may come from the monitor refresh rate, not the analysis timestep.

**For our IN training**: use 50 ms bins to match the neural data (analysis compares hidden states
to neural activity at 50 ms resolution). Update Milestone 2 task description accordingly.

### [x] 79 evaluation conditions

Condition IDs are the `PONG_BASIC_META_IDX` list (79 integers) defined in `code/utils/phys_utils.py`.
Full kinematic parameters for each condition are in `valid_meta_sample_full.pkl`:
- `ball_speed`, `ball_heading` (direction degrees)
- `ball_pos_x`, `ball_pos_y` (initial position)
- `ball_pos_dx`, `ball_pos_dy` (velocity components)
- `n_bounce_correct` (0 = straight, >0 = bouncing)
- `t_f` (total trial duration in RNN timesteps)
- `t_occ` (occlusion onset in RNN timesteps)

Access pattern: `meta[meta['meta_index'] == condition_id]`

### [x] Neural data structure

```python
dataset = {
  'behavioral_responses': {
    'occ': {
      'ball_pos_x': (79, 100),   # 79 conditions × 100 timepoints
      'ball_pos_y': (79, 100),
      'ball_final_y': (79, 100),
      'paddle_pos_y': (79, 100),
      't_from_start': (79, 100), # ms
      't_from_occ': (79, 100),
      't_from_end': (79, 100),
      ...
    }
  },
  'neural_responses': {'occ': (n_neurons, 79, 100)},
  'neural_responses_sh1': ...,  # split-half 1 (for reliability)
  'neural_responses_sh2': ...,  # split-half 2
  'neural_responses_reliable': {'occ': (n_reliable, 79, 100)},
  'neural_responses_reliable_FactorAnalysis_50': {'occ': (50, 79, 100)},
  'masks': {
    'occ': {
      'start_end_pad0': (79, 100),   # full trial
      'start_occ_pad0': (79, 100),   # visible epoch
      'occ_end_pad0':   (79, 100),   # occluded epoch
      ...
    }
  },
  'reliable_neural_idx': {'occ': array}
}
```

### [x] Does upstream higgsfield code expose hidden states? (2026-04-24)

**Confirmed by reading `Interaction Network.ipynb`**: `InteractionNetwork.forward()` returns only
`predicted` (shape `[batch*n_objects, 2]` = next-state speedX/Y). No intermediate states exposed.

Internal tensors available for subclassing:
- `effects`: `[batch, n_relations, effect_dim]` — raw relational effects from RelationalModel
- `effect_receivers`: `[batch, n_objects, effect_dim]` — effects aggregated per object ← **use this**

**Action for Milestone 3**: subclass `InteractionNetwork` in `dmfc/models/interaction_network.py`.
Override `forward()` to return `(predicted, effect_receivers)`. Do NOT edit the upstream file.
`effect_receivers` is the per-object relational state analogous to the RNN hidden state for DMFC
comparison.

### [x] Wall geometry (2026-04-24)

Confirmed from `add_mpong_frame_to_axis()` in `code/utils/generic_plot_utils.py`:

```python
ax.plot([-10, 10, 10, -10, -10], [10, 10, -10, -10, 10], 'k-', lw=2)  # arena
ax.plot([5, 5], [-10, 10], 'k-', lw=1)                                 # paddle at x=5
```

- **Arena**: x ∈ [-10, 10]°, y ∈ [-10, 10]°
- **Ball reflects off y = ±10°** walls
- **Paddle at x = 5°** (right of center)
- Ball travels left→right: x₀ ∈ [-8, 0]°, dx₀ > 0

Wall reflection is handled env-level (ball velocity flipped on collision with y = ±10). This is the
default per PLANNING.md decision. Walls are NOT graph edges in the IN by default.

### [x] GRU vs LSTM for flat-RNN baseline — MOOT (flat-RNN cut)

Rajalingham used both GRU and LSTM (192 = 4×2×2×2×6). **Flat-RNN baseline is cut from scope
per project timeline.** Only Interaction Networks will be trained. The analysis pipeline compares
IN hidden states directly against Rajalingham's released RNN outputs.

### Open problems

(all resolved as of 2026-04-24)

## Milestone 1 closeout — repo scaffolding (2026-04-25)

Closed out the four remaining Milestone 1 items in one pass after confirming the Zenodo bundle at `~/Downloads/MentalPong/` already covers what the "clone jazlab/MentalPong" task asked for (it ships `code/` alongside `data/`).

### What got built

- `dmfc/` package with five subpackages (`envs`, `models`, `training`, `analysis`, `rajalingham`), each a stub `__init__.py`. Empty for now; first real code lands in Milestone 2.
- `configs/`, `runs/`, `tests/`, `notebooks/` directories. `runs/` is gitignored per CONSTITUTION (run artifacts only).
- `data/dmfc -> ~/Downloads/MentalPong/data` (symlink). `data/README.md` (tracked) documents source + recreate-on-other-machine steps. `data/dmfc` itself is gitignored.
- `pyproject.toml`: PEP 621 `[project]` with 9 runtime + 3 dev deps, all `==`-pinned. `uv lock` resolves a 50-package tree cleanly. `uv.lock` is tracked.
- `tests/test_smoke.py`: 7 passing tests (6 subpackage imports + 1 core-deps import).
- `.gitignore`: added `runs/`, `data/dmfc`, `*.pt`, `.venv/`, `__pycache__/`, ruff/mypy/pytest cache dirs.
- `VERSIONS.md`: rewritten with resolved exact versions and rationale for each pin.

### Pin choices worth remembering

- **numpy 1.26.4** (not 2.x) — deliberate. The Zenodo `.pkl` files were produced under numpy 1; numpy 2 has caused pickle-roundtrip dtype shifts in similar scientific stacks. If a future task forces numpy 2, regenerate any cached analysis pickles.
- **torch 2.4.1** — modern stable. Upstream higgsfield code is pre-PEP 518 and has no torch pin; per CONSTITUTION fork discipline, if the upstream `InteractionNetwork` doesn't run under torch 2.4 we copy the file into `dmfc/models/` and patch the copy rather than editing in place.
- **Python 3.11.15** with upper bound `<3.12` — keeps the torch/numpy/scipy lockstep predictable until we have a reason to bump.

### Validation done

- `uv sync --extra dev` installed 50 packages without conflict.
- `python -c "import torch, numpy, scipy, sklearn, pandas, matplotlib, seaborn, yaml, pytest, ruff, mypy"` — all import.
- `pytest`: 7 passed, 14 warnings (matplotlib pyparsing deprecations — known, not ours).
- `ruff check dmfc tests` + `ruff format --check dmfc tests` — both clean.

### Carry-forward note for Milestone 3

Upstream `Interaction Network.ipynb` has not been re-executed under torch 2.4. The read-through that documented `effect_receivers` as the per-object hidden state stands, but we'll find out in Milestone 3 whether the actual upstream `InteractionNetwork` class instantiates and runs unmodified. If it doesn't, copy `Physics_Engine.py` (or whichever upstream file breaks) into `dmfc/models/` with an upstream-origin comment and patch the copy.
