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

### [x] Wall geometry (2026-04-24; paddle/occluder identity corrected 2026-04-25)

Confirmed from `add_mpong_frame_to_axis()` in `code/utils/generic_plot_utils.py`:

```python
ax.plot([-10, 10, 10, -10, -10], [10, 10, -10, -10, 10], 'k-', lw=2)  # arena
ax.plot([5, 5], [-10, 10], 'k-', lw=1)                                 # OCCLUDER LEFT EDGE
```

- **Arena (visible frame)**: x ∈ [-10, 10]°, y ∈ [-10, 10]°
- **Ball reflects off y = ±9.6°** (NOT y = ±10°). Discovered 2026-04-25: integrating with y=±10° walls produces a 0.8° post-bounce error in all 27 of 27 one-bounce conditions; integrating with y=±9.6° matches `y_occ_rnn_mwk` and `y_f_rnn_mwk` for all 79 conditions to ~1e-14. Plausibly accounts for the rendered ball's finite extent (paper draws the ball as 0.5° x 0.5°). The visible-arena edge is still y=±10°; only the ball-center reflection threshold is inset.
- **Paddle at x = +10°** (right wall, full vertical span). Paper Methods: paddle "rendered in the central vertical position at the right edge of the screen"; the 20° frame puts the right edge at x=+10°.
- **Occluder spans x ∈ [+5°, +10°]**. The vertical line at x=5° in `add_mpong_frame_to_axis` is the occluder's left edge, NOT the paddle.
- Ball travels left→right: x₀ ∈ [-8, 0]°, dx₀ > 0

Verified against `valid_meta_sample_full.pkl` (loaded live 2026-04-25): per-condition `x_occ_*_mwk ≈ 5°` (ball x when entering occluder), `x_f_*_mwk ≈ 10°` (ball x at trial end / interception line).

**Correction note**: Earlier 2026-04-24 entries called the x=5° line the paddle. That was a misread of the plotting code — `generic_plot_utils.py` has no separate paddle marker; the paddle is implicit in the right wall.

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

## Milestone 2 closeout — Mental Pong environment (2026-04-25)

Built a deterministic Gym-style env that reproduces Rajalingham's 79-condition stimulus set exactly. Headline result: integrated trajectories match `y_occ_rnn_mwk` and `y_f_rnn_mwk` for all 79 conditions to ~1e-14.

### What got built

- `dmfc/envs/conditions.py` — `load_conditions()` adapter over `valid_meta_sample_full.pkl`. Returns 79 `ConditionSpec` dataclasses (frozen) in `PONG_BASIC_META_IDX` order. Reads MWK-frame columns (`x0_mwk`, `y0_mwk`, `dx_rnn`, `dy_rnn`, `t_f`, `t_occ`, `y_occ_rnn_mwk`, `y_f_rnn_mwk`, `n_bounce_correct`) — no re-derivation of kinematics needed.
- `dmfc/envs/mental_pong.py` — `MentalPongEnv` Gym-style class with `reset/step/render`, plus `integrate_trajectory` and module-level helpers for resampling and masking. Trajectory is integrated at 41 ms (Zenodo native), observations exposed at 50 ms bins (neural-data binning). Ball state is zeroed and `visible_flag=0` when `ball_x ≥ 5°`. Paddle is one of two object nodes per the IN graph design; held at y=0 in the smoke render until a controller exists.
- `tests/test_mental_pong.py` — 9 tests. The endpoint-oracle test sweeps all 79 conditions; if it goes red the integrator is wrong, do not move on.
- CLI extensions: `--render` (static PNG), `--animate` (real-time GIF), `--grid` (all 79 trajectories in one PNG, 9×9 layout, blue=no-bounce, red=one-bounce).

### Two corrections to Milestone-1 docs landed in this pass

Both turned out to be misreads from the earlier read-through-only inspection; verified live against `valid_meta_sample_full.pkl`. Both are now consistent across SCRATCHPAD.md, PLANNING.md, and the code.

1. **Paddle is at x=+10°, not x=+5°.** The vertical line at x=5° in `add_mpong_frame_to_axis` is the occluder's left edge, not the paddle. Verified: per-condition `x_occ_*_mwk ≈ 5°` (ball entering occluder), `x_f_*_mwk ≈ 10°` (ball at trial end). Paper Methods unambiguously place the paddle "at the right edge of the screen" — the 20° frame puts that at x=+10°.
2. **Ball reflects off y=±9.6°, not y=±10°.** Discovered by validating the integrator: y=±10° walls produce a 0.8° post-bounce error in all 27/27 one-bounce conditions, while y=±9.6° matches all 79 endpoint oracles to ~1e-14. Plausibly accounts for the ball's finite extent (0.5°-square sprite) vs. the wall edge. The visible arena box is still ±10°; only the ball-center reflection threshold is inset.

### Two design decisions confirmed with the user before coding

- **Trajectory source**: integrate kinematics ourselves (true Gym simulator) rather than replay pre-binned trajectories from `behavioral_responses`. Lets us run ablations and OOD tests later; validated by matching all 79 endpoint oracles.
- **Occluded-epoch input**: `visible_flag=0` and ball_x/ball_y/ball_dx/ball_dy zeroed (not last-seen-position frozen). Faithful analog of the paper's RNN setup, where Gabor frames literally hide the ball during occlusion. This is what makes the simulation-vs-no-simulation comparison meaningful.
- **Paddle in IN graph**: included as an object node with a ball↔paddle relation. Matches the IN's object-relational philosophy; the alternative ("ball-only graph, paddle as decoded output") would leave the graph with one node and zero relations.

### Carry-forward note for Milestone 3

The paddle currently has no controller — `step(action)` accepts a target-y but the smoke CLI passes `0.0` every tick. When an IN policy or training loop arrives in Milestone 3, the paddle dynamics in the env (`MAX_PADDLE_SPEED_DEG_PER_MS = 0.01` per the paper) become live. No changes to the env API expected.

## How to visualize the Mental Pong stimulus

The env CLI (`python -m dmfc.envs.mental_pong`) supports three visualization modes. Run these from the repo root after activating the venv:

```bash
# activate the venv (once per terminal)
source .venv/bin/activate

# create the output dir (gitignored — figures are regenerable from scripts)
mkdir -p figures

# 1. Static PNG of all 79 trajectories in a 9x9 grid
#    Blue = no-bounce (52), red = one-bounce (27). Green dot = start, black = end.
python -m dmfc.envs.mental_pong --grid --out figures/all_79.png
open figures/all_79.png

# 2. Animated GIF of one condition (real time, 20 fps)
#    Conditions are indexed 0..78. Condition 0 is no-bounce; condition 1 is one-bounce.
python -m dmfc.envs.mental_pong --animate --condition 1 --out figures/cond1.gif
open figures/cond1.gif

# 3. Static PNG of one condition with full trajectory + endpoint oracles drawn
python -m dmfc.envs.mental_pong --render --condition 1 --out figures/cond1.png
open figures/cond1.png
```

Notes:
- `open` is the macOS command; on Linux use `xdg-open`.
- If you don't want to activate the venv, prefix with the venv's python directly: `.venv/bin/python -m dmfc.envs.mental_pong --grid --out figures/all_79.png`.
- `--fps N` on `--animate` plays the GIF faster than real time (default 20 fps = real time at 50 ms bins).
- Loop a few conditions:
  ```bash
  for i in 0 1 14 17 22; do
      python -m dmfc.envs.mental_pong --animate --condition $i --out figures/cond_$i.gif
  done
  ```

The paddle in the GIFs is held at y=0 (action=0 every step) — placeholder until a controller is wired up in Milestone 3.

## Milestone 3 closeout — IN training infrastructure + pilot (2026-04-26)

End-to-end training loop running on CPU. Pilot (`sim-mov`, seed 0, `effect_dim=10`, 50K steps) trained to loss 0.05 EMA in ~13 min and produced a clean run artifact. The bones of the IN-Mental-Pong stack are solid; the two real surprises were architectural and are documented below.

### What got built

- `dmfc/models/_upstream_in.py` — vendored copy of upstream `RelationalModel`, `ObjectModel`, `InteractionNetwork` from `Interaction Network.ipynb` (cells 10/12/14). Modernized for torch 2.4 (`Variable(...)` removed; `forward` now returns `(predicted, effect_receivers)`). No mathematical changes to this file — it's a faithful upstream reference per CONSTITUTION fork discipline.
- `dmfc/models/interaction_network.py` — `MentalPongIN(nn.Module)`. 2 objects (ball, paddle), 2 directed relations (ball→paddle, paddle→ball). Per-step object features `(x, y, dx, dy, visible, is_ball, is_paddle)` are concatenated with the previous step's `effect_receivers` for that object → relational message passing → 7-d output head reads from concatenated effect_receivers. Recurrence is the prior-effect concat; `effect_receivers` is the published "hidden state" the M4 pipeline will consume.
- `dmfc/envs/random_conditions.py` — `sample_random_condition(rng)` and `sample_batch(...)`. Samples `(x0, y0, speed, heading)` from the 79's empirical envelope, integrates via `mental_pong.integrate_trajectory`, rejects `n_bounce > 1` and out-of-envelope `t_f_steps`. Returns `ConditionSpec` with `meta_index = -1`. The training loop builds infinite stream from this generator; the canonical 79 are held out for eval.
- `dmfc/training/losses.py` — `compute_loss(outputs, targets, variant, visible_mask, valid_mask)`. One implementation, four variants:
  - `mov` (Intercept): supervise output[6] every valid step.
  - `vis-mov` (Vis): output[2,3] visible-and-valid + output[6] every valid.
  - `vis-sim-mov` (Vis+Occ): output[0,1] every valid + output[6] every valid.
  - `sim-mov` (Vis&Occ): output[4,5] occluded-and-valid + output[6] every valid.
- `dmfc/training/config.py` — typed `RunConfig` dataclass + `load_config`/`dump_config` over YAML.
- `dmfc/training/train.py` — argparse → seed everything → make `runs/in_<variant>_h<eff>_s<seed>_<ts>_<git>/` → log to file + stdout → train (random conditions per step, gradient clipping) → final checkpoint → forward over the 79 → `hidden_states.npz`.
- `configs/in_{intercept,vis,vis_occ,vis_and_occ}.yaml` — one per loss variant. All start at `effect_dim=10`, lr=1e-3, batch=32, max_steps=50000.
- Tests: `tests/test_random_conditions.py`, `tests/test_losses.py`, `tests/test_training.py`. 17 new tests; total now 33/33.

### Two design deviations that landed mid-milestone

**1. Final ReLU dropped from the relational MLP.** First pilot loss plateaued at ~25 (= variance of mean-zero targets) and `effect_receivers std = 0.0000` everywhere. Probe showed 100% of pre-final-ReLU activations in the upstream `RelationalModel` were negative — the ReLU was clipping every effect to zero, so the IN was a constant-output model whose only learnable signal was the bias on the 7-d output head. Fix: build a local `_RelationalMLP` in the wrapper without the final ReLU. The vendored upstream copy stays untouched per CONSTITUTION.

The upstream notebook gets away with the final ReLU because its solar-system task's next-velocity targets happen to be in a regime where positive effects suffice. For Mental Pong the relational state must encode signed offsets (paddle above vs. below ball, etc.), so a non-negative effect vector is structurally inadequate.

**2. Gradient clipping at norm 1.0.** Second pilot trained beautifully to step ~10K (loss 4.4) then catastrophically diverged: by step 25K loss was 6.7e30 and effect_receivers std was 5e15. Cause: the recurrent loop where `prev_effect_receivers` is concatenated into next-step object features has no fixed point with a linear final layer — any positive Lyapunov component compounds. Fix: `torch.nn.utils.clip_grad_norm_` at 1.0 in `train.py` (constant `GRAD_CLIP_NORM`). Standard RNN-stability technique.

Tanh on the relational output was tried and rejected — pre-tanh activations have magnitude ~30+ at init, so tanh saturates everywhere and the network can't learn out of saturation. The right place to bound the dynamics is in the optimizer (clip_grad_norm), not in the architecture.

### Pilot — `runs/in_sim-mov_h10_s0_20260426-125840_fd614df-dirty/`

50K steps, ~13 min on CPU. Loss 36.8 → 0.05 EMA (400× reduction). On the held-out 79 conditions:

| Supervised quantity | MSE | Target var | R²    |
|---|---|---|---|
| Occluded ball_x (output[4]) | 0.037 | 2.39  | 0.985 |
| Occluded ball_y (output[5]) | 0.27  | 25.40 | 0.989 |
| Final intercept y (output[6]) | 0.08 | 26.95 | 0.997 |

`effect_receivers` shape `(79, 72, 2, 10)`, std 2.15 globally and 1.31 averaged across conditions — non-trivially varying, ready for M4.

A Fig. 5B preview using **the supervised intercept output** (not the proper hidden-state-decoded version M4 will compute) shows Pearson r ≈ 0.999 from t=0 onward. This is encouraging — it suggests the IN learned the kinematics so completely that initial-state information at t=0 already determines the final intercept — but it isn't the comparison we'll publish. The honest M4 metric trains a linear decoder on `effect_receivers` and reports decoding accuracy over time; that's where the rapid-vs-slow rise question gets settled.

### Pinned design choices (matter for M4 / M5)

- **Object features are 7-d**: `(x, y, dx, dy, visible, is_ball, is_paddle)`. Identity flags are necessary because the relational MLP sees both objects through the same weights.
- **Paddle held at y=0 throughout training**. M3 had no controller; the paddle is a fixed reference object so the relational graph has a peer for the ball. Revisit if a future ablation needs paddle motion.
- **Random condition generator uses the 79's empirical envelope**, not Rajalingham's full sampling distribution. Slight bias possible but the held-out 79 are inside the envelope by construction.
- **Output[0:2] = output[2:4] = output[4:6]** at the target level (all hold the true ball position); the loss-mask decides which is supervised when. This means a network can in principle learn three independent ball-position decoders that are forced to agree only on supervised positions. Worth keeping in mind as a degree of freedom if the M4 analysis surfaces oddities.

### Carry-forward to M4

- `runs/in_sim-mov_h10_s0_20260426-125840_fd614df-dirty/hidden_states.npz` is the first real artifact for the analysis pipeline. M4 should be able to read it without any model code in scope (CPU-only, npz only).
- The `meta_index` array in the .npz aligns row-by-row with `dmfc.envs.conditions.PONG_BASIC_META_IDX`, which means the 79 rows align exactly with Rajalingham's neural-data array. No re-ordering needed at the analysis layer.
- `effect_dim=10` matches the smaller of Rajalingham's hidden sizes; the M5 sweep will additionally do `effect_dim=20`.

### Carry-forward to M5 (cluster sweep)

- M5 needs SLURM scaffolding (deferred from M3 per the planning Q&A). When the time comes, the per-run wall-clock at `effect_dim=10` is ~13 min on CPU; with cluster GPUs the full 4×2×5=40-run matrix should finish in well under an hour.
- The training loop is already CPU/GPU agnostic (`device = "cuda" if available else "cpu"`); no code changes needed for the cluster move beyond a sbatch template and a runs-root override.
