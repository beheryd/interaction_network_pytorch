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

### [x] Hidden-unit sweep values

**Exactly 3 values: 10, 20, 40 hidden units.** Update PLANNING.md experimental matrix.

192 models = 3 hidden sizes × 4 loss types × 16 seeds.

### [x] RNN architecture

- **Type**: GRU or LSTM (both present in release)
- **Input dim**: 100 (Gabor-filtered visual frames → PCA → 100D)
- **Output dim**: 7 (indices 0–1 vis-sim, 2–3 vis, 4–5 sim, 6 final position)
- **Loss types in data**: `vis-sim`, `vis`, `sim`, `mov`
  - These map to Rajalingham's four classes (Intercept=sim, Vis=vis, Vis+Occ=vis-sim, Vis&Occ=mov)
  - Verify exact mapping against paper before training IN.

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
- **RNN timestep**: ~41 ms/step, 90 timesteps → ~3.7 s total
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

### [x] Does upstream higgsfield code expose hidden states?

Not yet inspected directly. From `Interaction Network.ipynb` — it's a Jupyter notebook.
Next step: read it to determine if hidden-state extraction exists or needs to be added in a subclass.

### Open problems

- [ ] Verify exact loss-type label mapping: `vis-sim`/`vis`/`sim`/`mov` → Intercept/Vis/Vis+Occ/Vis&Occ
- [ ] Inspect `Interaction Network.ipynb` for hidden-state API
- [ ] Decide whether to use GRU or LSTM for the flat-RNN baseline (or both)
- [ ] Confirm wall geometry from the condition metadata (what are the y-bounds for reflections?)
