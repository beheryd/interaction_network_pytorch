# Development scratchpad

- Use this file to keep notes on ongoing development work.
- Open problems marked with [ ]
- Fixed problems marked with [x]

## NOTES

## Milestone 5 sweep + figures + stats scaffolding (2026-04-29)

Three work blocks landed in one push: (1) the full 40-run M5 sweep, (2) the two M5 figures regenerated on it, (3) `dmfc/analysis/stats.py` so the formal Wilcoxon/partial-R² stage is unblocked. The M5 sweep ran clean from end to end and reproduces Rajalingham's main structural findings.

### M5 sweep — 40/40 converged

SLURM job 9168152 on `gershman_lab/shared` (`--array=5-39%10`, 2h walltime per task). Rows 0-4 of the matrix were symlinked from the Intercept HP-pass runs (functionally identical: `mov`, h=10, seeds 0–4, clip=0.5) so we didn't burn ~5h re-training them.

Headline numbers across the 40 runs:

- **State**: 35/35 array tasks `COMPLETED 0:0` (no TIMEOUTs, no NODE_FAILs).
- **Convergence**: final losses 0.009–0.087. Per-variant medians: vis-mov 0.015 (best), vis-sim-mov 0.033, sim-mov 0.041, mov 0.028. Zero NaN lines across all 40 × 50000 steps.
- **Wall clock**: 1:02–1:31 per task; ~3.5 h end-to-end. The throttle (`%10`) was the right setting — node-pool contention varied from 5 to 25 over the run, never starved.
- **Disk**: 40 × ~1.5 MB run dirs under `runs_m5/`.

### Two figures, two style choices the user pinned

`figures/fig5b_full_sweep.png` (primary deliverable, PRD F3):
- **Aggregated by loss class**, not 40 spaghetti lines. Refactored `reproduce_fig5b.py`: new `aggregate_in_curves_by_class()` helper pools the 10 runs per class (2 hidden × 5 seeds) into one mean+SEM curve. CLI flag `--in-aggregation {by-class,per-run}` defaults to `by-class`.
- **Palette matched to Fig. 4 panel D**: Intercept blue, Vis green, Vis+Occ red, Vis&Occ orange. **DMFC = black**, RNN = grey family.
- One color-routing bug surfaced and fixed: substring matching on labels routed all "Vis…" classes to green because "Vis" is a substring of "Vis+Occ" and "Vis&Occ". Fix: iterate longest-label-first in `_in_curve_color()`.
- **Result**: all 4 IN class lines stack at r ≈ 1.0 from t=0 (input-asymmetry effect, robust across loss variants — the M4 progress 4 finding holds at full sweep). DMFC rapid rise to ~0.85 by 200 ms; RNN classes slow rise to 0.5–0.65 by 1200 ms.

`figures/fig4_paper_replica.png` (secondary deliverable, PRD F4 — paper-replica variant):
- **3-panel D/E/F layout**, IN-only (no RNN data per user direction).
- Color = loss class (same Fig. 4 palette). Marker size = hidden units (10 small / 20 large). Regression lines on E and F.
- **Axis auto-extension**: paper-strict limits when IN data fits, silent extension when data spills out (option 2 from the user). Panel F is the visible case — IN's task MAE 0.1–0.5° vs paper's 1–4°, axis auto-stretched right to ~0.
- The 40-run version reproduces the structural findings:
  - **Panel D**: Vis+Occ (median NC ≈ 0.52) and Vis&Occ (≈ 0.55) substantially above Intercept (≈ 0.25) and Vis (≈ 0.16). Same direction as the paper's RNN finding.
  - **Panel E**: positive regression — better SI ↔ better NC. Reproduces paper Fig. 4E for IN.
  - **Panel F**: NC vs task MAE shows no within-IN gradient (all points cluster at low MAE), so task performance alone does not predict NC.

### The fig4 cache fix (60 min → 2 min)

First M5 fig4 render hung in computation for over 1h before being killed. Diagnosis: `in_swarm` calls `in_point_for_run` per run, which calls `neural_consistency_from_states` per run, and that function recomputes 3 DMFC-side RDMs (full + 2 split-halves) on every invocation. Each pdist is over (1889 reliable units × ~7000 cells), order GFLOPs, and we did it 40× redundantly because all 40 IN runs share the same DMFC slice (same `T_in=72`, same `occ_end_pad0` mask).

Fix: new `NeuralRDMCache` dataclass + `compute_neural_rdm_cache()` helper at the top of `reproduce_fig4.py`. `in_swarm()` builds the cache once (lazily, on the first non-diverged run so we know `T_in`) and threads it into `in_point_for_run` via an optional `neural_rdm_cache=` parameter. The non-cache path is preserved as a fallback so the original `neural_consistency_from_states` still composes from public surface.

Wall clock dropped from 60+ min (killed before completion) to **~2 min** end-to-end. The cache also exits cleanly when the run dirs have heterogeneous `T_in` (raises with a clear message rather than silently mismatching).

### Stats scaffolding — `dmfc/analysis/stats.py`

Four pure-numerics functions covering the M5 statistical tests. No statsmodels dependency (just scipy + sklearn, both already pinned):

- `time_to_threshold(curves, threshold, time_axis) → (n_models,)` — first time each curve reaches threshold; NaN if it never does.
- `rmse_auc(rmse_curves, time_axis, window_ms=None) → (n_models,)` — trapezoidal AUC over a window.
- `wilcoxon_rank_sum(group_a, group_b) → WilcoxonResult` — wraps `scipy.stats.ranksums` plus a rank-biserial effect-size proxy `|Z|/sqrt(n_a+n_b)`.
- `partial_r2(target, base_predictors, extra_predictors) → float` — R²_full − R²_base for the *change* in R² when adding extra predictors. Note: not the classical (R²_full − R²_base)/(1 − R²_base); M5 calls for the simpler form.

21 tests in `tests/test_stats.py`, including reference-impl checks against scipy/sklearn directly. Smoke-validated against the published 96-RNN gabor_pca subset:

- `time_to_threshold(r=0.5)` per RNN class: mov 984ms, vis-mov 861ms, vis-sim-mov 779ms, sim-mov 902ms — consistent with paper Fig. 5B class ordering.
- `(1−r)`-AUC over 0–1200 ms: vis-sim-mov is best (lowest accumulated error), matching its highest median r.
- **NC ~ SI R² = 0.7648** on the 96-RNN swarm — directly reproduces paper Fig. 4E qualitative claim.
- `wilcoxon_rank_sum(mov, sim-mov)` time-to-r=0.5: z=1.13, p=0.26, effect=0.165 — modest difference between mov and sim-mov on a single metric.

Now ready to plug `runs_m5/` IN points into the formal IN-vs-RNN-class Wilcoxons + partial R² with-vs-without-IN. That's a small driver script (~50 lines) the next session can build.

### Carry-forward

- Two-stage panel for `reproduce_fig5b.py` is still the open M4 task. Now more interesting since we have all 40 IN runs to feed it.
- Build `dmfc/analysis/run_m5_stats.py` — driver that loads `runs_m5/` IN points + the 96-model RNN swarm, runs the 4 stat tests, emits a results table for the writeup. Should land before M6 starts.
- The PRD S2/S3 questions now have visual answers landed in the figures; formal Wilcoxon answers need the driver above.

## Milestone 5 prereq — Intercept HP pass + sweep orchestration (2026-04-28)

The M5 prereq from the M4 closeout (Intercept variant `mov` reliably diverges to NaN on seed=0 at default grad_clip=1.0) is closed. Companion deliverable: full M5 sweep orchestration is built and dry-run-validated, ready to fire the moment we want a 40-run cluster sweep.

### Headline result: grad_clip_norm=0.5 wins

All three single-intervention HP candidates survived the seed=0 3K-step smoke (vs the original NaN by step ~800 at default):

| candidate | loss @ 3K | std(eff_receivers) | survived past step 800 |
|---|---:|---:|:---:|
| A: lr=5e-4 | 2.62 | 7.25 | ✓ |
| **B: grad_clip=0.5** | **1.21** | **0.93** | ✓ |
| C: weight_decay=1e-4 | 1.71 | 0.95 | ✓ |

B picked because (i) best step-3K loss → suggests room to fully converge by 50K, (ii) effect_receivers std (0.93) lands in the same ballpark as the M3 sim-mov pilot (1.31) so the M4 analysis pipeline behavior stays comparable, and (iii) directly addresses the diagnosed failure mode (recurrent gradient runaway from one-scalar supervision).

### Full-sweep confirmation: B is robust across all 5 seeds

Submitted `scripts/submit_intercept_hp.sbatch` as a SLURM array job (`--array=0-4`, `gershman_lab` account, `shared` partition, 2h walltime per task). Full-50K results:

| seed | final step | final loss | std | NaN lines | hidden_states.npz |
|---:|---:|---:|---:|---:|:---:|
| 0 | 50000 | 0.0619 | 1.42 | 0 | ✓ |
| 1 | 50000 | 0.0281 | 0.76 | 0 | ✓ |
| 2 | 50000 | 0.0423 | 0.83 | 0 | ✓ |
| 3 | 50000 | 0.0294 | 1.09 | 0 | ✓ |
| 4 | 50000 | 0.0260 | 1.31 | 0 | ✓ |

Final losses 0.026–0.062 are all in the same regime as the M3 sim-mov pilot's 0.05 EMA. Zero NaN across 5 × 50000 steps. Run dirs under `runs_intercept_hp/in_mov_h10_s{0..4}_20260428-152115_*/`.

### Two SLURM gotchas worth pinning

1. **The first run hit `--time=01:00:00` and TIMEOUT'd 5/5 tasks at step ~47–48K**, leaving no `hidden_states.npz`. CPU on `shared` is meaningfully slower than my 12-min/3K-step smoke estimate (smoke ran on the Claude-shell cluster node directly, not via SLURM). Real-job per-task wall on `shared`: 1:02–1:29. Bumping to `--time=02:00:00` was sufficient. The slow tail variance comes from node-pool contention — first attempt put 4 of 5 tasks on one node, second attempt got 5 different nodes. **For the M5 sweep we kept the 2h budget**.
2. **`kempner` partition rejects CPU-only jobs.** Their submit hook says: *"You must request a gpu using the --gpus or --gres option to use kempner partition, if you have CPU work for this hardware please use sapphire or shared."* Standard FASRC `shared` partition with `gershman_lab` account works (note the FASRC fairshare account, not `kempner_gershman_lab` which is Kempner-only).

### Sweep orchestration as it actually landed

Pattern follows the Kempner array-jobs handbook recommendation: CSV lookup keyed on `$SLURM_ARRAY_TASK_ID`, dispatch with CLI overrides. Three pieces:

- `scripts/gen_sweep_matrix.py` — generates the canonical 40-row matrix from intent. Records the per-variant HP override rule (mov gets clip=0.5, others stay at 1.0).
- `scripts/sweep_matrix.csv` — `task_id,variant,config,hidden,seed,grad_clip_norm`. Layout: rows 0–9 mov, 10–19 vis-mov, 20–29 vis-sim-mov, 30–39 sim-mov. Within each variant: 0–4 are hidden=10, 5–9 are hidden=20.
- `scripts/submit_m5_sweep.sbatch` — `--array=0-39%10` (10 concurrent), reads CSV row, dispatches train.py with `--config / --seed / --effect-dim / --grad-clip-norm`. Run artifacts at `runs_m5/in_<variant>_h<hidden>_s<seed>_<ts>_<git>/`.

Three new train.py CLI flags landed (mirroring the existing `--seed`, `--max-steps` pattern): `--effect-dim`, `--lr`, `--grad-clip-norm`, `--weight-decay`. `TrainingConfig.grad_clip_norm` and `weight_decay` are now first-class fields with defaults preserving prior behavior. The original module constant `GRAD_CLIP_NORM` is still in train.py as a default-source comment but no longer used.

### Validation done before declaring orchestration ready

- CSV-lookup awk logic verified for task IDs 0, 5, 10, 25, 39 (variant/hidden/seed boundaries).
- `--effect-dim 20` smoke run: produces `effect_receivers (79, 72, 2, 20)` artifact + saved `config.yaml` shows `effect_dim: 20`.
- `sbatch --test-only scripts/submit_m5_sweep.sbatch` accepted by SLURM (commits to a real start time).
- Did NOT submit for real yet — waiting for an explicit M5 launch decision.

### Carry-forward to M5 launch

When ready: `sbatch scripts/submit_m5_sweep.sbatch`. Then on completion:

- `python -m dmfc.analysis.reproduce_fig5b --in-runs 'runs_m5/in_*' --rajalingham-data data/dmfc/` regenerates Fig. 5B with all 40 IN points (per CLAUDE.md command convention).
- Same for `reproduce_fig4.py`. The diverged-run NaN-skipping path is already in place.
- M5 stat tests (Wilcoxon IN vs RNN class for time-to-threshold-r and SI/NC distributions) need a new `dmfc/analysis/stats.py` — separate task, can land before or after the sweep finishes.

The existing 5 Intercept HP run dirs at `runs_intercept_hp/in_mov_h10_*` are functionally equivalent to the first 5 rows of the M5 sweep (mov, h=10, seeds 0–4, clip=0.5). If we wanted to save the ~5h × 5 = ~25h of cluster time, we could symlink them into `runs_m5/` rather than re-train. **Open question for M5 launch**: do that vs re-train for cleanliness?

## Milestone 4 final close — Fig. 4 + pipeline validation (2026-04-27)

Two remaining M4 sub-tasks landed in one pass: `reproduce_fig4.py` (PRD F4 secondary deliverable) and pipeline validation (PRD S1, rescoped). Test count 94→104; ruff + mypy clean. Plan file: `~/.claude/plans/can-we-continue-looking-linear-lantern.md`.

### Headline correction to the validation plan

**The Zenodo release does not ship per-RNN hidden states.** The earlier note (line 38 above) referencing `res_rnn['state'][model_key]['data_neur_nxcxt']` was wrong — `pd.read_pickle('offline_rnn_neural_responses_reliable_50.pkl')` only has top-level keys `df` and `all_metrics`. There is no `state`. `per_model[fn]` carries `yp`, `yt`, `r_start1_all`, `mae_start1_all`, and bounce-stratified variants — no states.

Implication: per-RNN external validation (recompute our NC/SI on each RNN's hidden states, compare to the published columns within ~1e-3) is **infeasible without rerunning their training pipeline**. The validation pivoted to a math-identity check.

### Pipeline validation as it actually landed

`dmfc/analysis/validate_pipeline.py` imports Rajalingham's reference functions directly from the unpacked Zenodo source tree at `~/Downloads/MentalPong/`:

- `code/utils/phys_utils.get_state_pairwise_distances` — RDM construction.
- `analyses/rnn/RnnNeuralComparer.RnnNeuralComparer.get_noise_corrected_corr` — noise-corrected RSA (`r_xy_n_sb`, `r_xy_n`, `r_xx`, `r_yy`, `r_xy`).
- The SI math is small enough (KFold + OLS + per-coord MAE) that we re-state it inline with explicit attribution rather than wrestling with `rnn_analysis_utils.get_model_simulation_index`'s upstream-package imports.

Both pipelines are run on a fixed-seed synthetic input (`(20, 30, 8)` model states + `(15, 20, 30)` neural states with 0.3-noise split halves), and outputs compared element-by-element. **All three checks pass at 0.0e+00 max-absolute-diff** — bit-for-bit equivalence.

Two import-time gotchas worth preserving:
1. **`RnnNeuralComparer.py` imports `from phys import phys_utils, data_utils`** — a package layout that doesn't exist as `phys/` in the Zenodo bundle (the file is at `code/utils/phys_utils.py`). Workaround: alias `phys` → a synthetic `types.ModuleType("phys")` with `phys_utils` as a sub-attribute, plus a stub for `data_utils` (referenced at import but only used by methods we don't call). Logic in `validate_pipeline.import_reference`.
2. **`code/utils/utils.py` is referenced as just `utils`** by `rnn_analysis_utils.py`; we register the alias in `sys.modules` to keep its `flatten_to_mat` accessible.

For posterity: these workarounds are **only** needed inside `validate_pipeline.py`. The main analysis pipeline is self-contained and doesn't import any Zenodo source code.

### Fig. 4 (PRD F4) results

`figures/fig4_pilot.png` — two-panel scatter, NC on the left, SI on the right. RNN swarms = 96 gabor_pca models pulled from the published `pdist_similarity_occ_end_pad0_euclidean_r_xy_n_sb` (NC, lives in `rnn_compare_*.pkl['summary']`) and `decode_vis-sim_to_sim_index_mae_k2` (SI, lives in `offline_rnn_*.pkl['df']`). IN points = 4 pilot runs, computed on-the-fly via our pipeline.

| Loss class | IN NC | RNN NC mean | IN SI (deg) | RNN SI mean (deg) |
|---|---:|---:|---:|---:|
| Intercept (mov, seed=1) | 0.241 | ~0.16 | 1.79 | ~3.0 |
| Vis (vis-mov) | 0.066 | ~0.27 | 1.69 | ~2.5 |
| Vis+Occ (vis-sim-mov) | -0.082 | ~0.34 | 0.16 | ~1.7 |
| Vis&Occ (sim-mov) | 0.476 | ~0.32 | 0.77 | ~1.4 |

Two patterns the pilot already shows:

1. **IN's SI is uniformly lower (better) than the RNN swarm** across all four loss variants. Most extreme on Vis+Occ (IN 0.16° vs RNN ~1.7°) where the IN essentially solves online ball position decoding. Plausible reason: object-relational structure + explicit kinematic input gives the IN a structural advantage on the simulation task that RNNs trained on pixels can't easily match. This is a positive answer to PRD S3 ("does the IN swarm land within the RNN swarm range on Fig. 4?") — the IN is *outside* the RNN swarm, in the better-than direction.
2. **IN's NC (representational geometry alignment with DMFC) is variant-dependent and not uniformly better/worse**. Vis&Occ (sim-mov) sits well above the RNN swarm at 0.476 (vs RNN ~0.32); Intercept lands inside the RNN swarm; Vis and Vis+Occ are below the RNN swarm. So the IN can match RNN's NC on the variant that supervises *only* occluded position + endpoint, but underperforms on intermediate-supervision variants. M5's full sweep (5 seeds, 2 hidden sizes per variant) will tell us how seed-stable these are.

### Loader extension

`load_rnn_compare(data_dir)` added to `dmfc/rajalingham/load.py`. Returns the (192, 90) summary DataFrame from `rnn_compare_all_hand_dmfc_occ_50ms_neural_responses_reliable_FactorAnalysis_50.pkl`. Carries the published Fig. 4D NC column. Row order is identical to `load_rnn_metrics().df` and rows are joinable on `filename`.

### Carry-forward to M5

- **Intercept HP pass** is still the only M5 prereq. The `mov` variant on seed=0 diverged to NaN by step 800 (SCRATCHPAD M4 progress 4); seed=1 converges with default settings but reproducibility across all 5 seeds is not guaranteed. Lower lr / stricter clip / weight decay are the suggested first sweeps.
- **Diverged-run handling** in `reproduce_fig4.py`: `_has_diverged_states` skips run dirs whose `effect_receivers` contain NaN. Means the canonical `python -m dmfc.analysis.reproduce_fig4 --in-runs runs/in_*` (per CLAUDE.md) won't crash on the diverged seed=0 mov pilot.
- **Statistical tests for Fig. 4** (M5 sub-task): the IN's "uniformly lower SI" pattern is real on a single seed/hidden-size; with 5 seeds × 2 hidden sizes per variant we can test it formally (Wilcoxon rank-sum: IN SI distribution vs RNN class SI distribution). Same for NC. The current Vis&Occ-above-RNN-swarm finding on NC is the most interesting and the one most worth pinning down with the M5 statistics.

### Time-axis question (carry-forward from the user's prompt)

The user asked what "Time from trial start" represents on `figures/fig5b_4variants_aligned.png`, especially past the 1200 ms display crop. Plan file has the full explanation; the short version: **all 79 conditions exceed 1200 ms** (min t_f = 1353 ms motion-onset to interception, median 2132 ms, max 3526 ms), so within the displayed window every bin sees every condition. Beyond ~1350 ms conditions begin terminating; per-timestep r is computed only on the still-valid subset, hence the late-trial volatility. The 1200 ms crop is the right honest window. Optional follow-up: add a "n_valid(t)" overlay to the figure for the writeup. Skipped for now per the user's instruction to focus on M4 close.

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

## Milestone 4 progress — loader + endpoint decoder (2026-04-26)

First two M4 modules landed: the Zenodo adapter (`dmfc/rajalingham/load.py`) and the time-resolved endpoint decoder (`dmfc/analysis/endpoint_decoding.py`). Plan file at `~/.claude/plans/crystalline-cuddling-bentley.md`.

### What got built

- `dmfc/rajalingham/load.py` — three frozen-dataclass loaders, no analysis logic:
  - `load_dmfc_neural()` → `DMFCData` with `responses (n_neur, 79, 100)`, `responses_sh1`/`_sh2`, canonical `masks` (8 names; `extra_masks=` opens up the `_roll` shifts), `behavioral` dict, `meta_index` from `PONG_BASIC_META_IDX`. The pooled DMFC `neural_responses_reliable['occ']` is 1889 reliable units.
  - `load_rnn_metrics()` → `RNNMetrics` with `df (192, 695)` and `per_model[ckpt_path] -> {yp, yt, r_start1_all, mae_start1_all, ...}`. `r_start1_all` shape `(100 iter, 90 t)`.
  - `load_decode_dmfc()` → `DecodeResult` with the 23-target `beh_targets` list, `entries` (a list of 11 res_decode sub-dicts), `decoder_specs`, `neural_data_key`. The pickle key `'neural_data_to_use '` has a trailing space we preserve as-is when reading.
  - All loaders use `pd.read_pickle` (not raw `pickle.load`) — the Zenodo pickles need the old-pandas `pandas.core.indexes.numeric` shim that pandas handles transparently.
- `dmfc/analysis/endpoint_decoding.py` — pure numerics, no I/O:
  - `decode_endpoint(states, endpoint_y, valid_mask=None, n_splits=5)` → `DecodingResult` with per-fold and fold-averaged `r(T)` / `rmse(T)`. `GroupKFold` across the 79 conditions, per-timestep `LinearRegression` (1-D target → no need for PLS).
  - `flatten_receivers((n_cond, T, 2, eff))` → `(n_cond, T, 2*eff)` — first object's effect goes into the first half of the feature vector.
  - `load_pilot_states(run_dir)` reads `hidden_states.npz` and returns `(states_flat, endpoint_y, valid_mask)` ready to feed `decode_endpoint`. `endpoint_y` is `targets[:, 0, 6]` (output index 6 is the final intercept y per M3 closeout).
- Tests: `tests/test_rajalingham_load.py` (5 tests, 4 skip-if-data-missing) and `tests/test_endpoint_decoding.py` (9 tests: synthetic perfect-signal/noise/shape/determinism/mask + pilot smoke). Suite total: 33 → 47.

### One real surprise — M3's "r ≈ 0.999 from t=0" was true for the proper hidden-state decoder too

The M3 closeout flagged that the *supervised* output index 6 already gives r ≈ 0.999 from t=0. I expected the proper hidden-state-decoded version (LinearRegression on `effect_receivers`) to show a learning curve — early states uninformative, late states fully informative — so the pilot test originally asserted `late > early`.

It doesn't. The decoder run on the pilot returns r[0]=0.999, r[35]=0.999, r[71]=NaN (last timestep is past every condition's valid window — masked correctly). The IN's effect_receivers encode the deterministic kinematics so completely that the eventual endpoint is already linear-decodable from the very first step.

This is the `sim-mov` (Vis&Occ) loss variant, which directly supervises occluded ball position + final intercept. With deterministic kinematics + endpoint supervision, the network has a strong incentive to embed initial-condition info into the relational state from the start. The result isn't wrong — it's just that this particular variant's hidden states don't have the rapid-vs-slow rise structure that Fig. 5B exists to differentiate.

**The honest Fig. 5B comparison will need to look at curves re-aligned to occlusion onset**, where the slope before vs. after occlusion is the signal. With 79 conditions at different `t_occ` values, the per-timestep curve on the absolute time axis mixes pre- and post-occlusion phases together; only after re-alignment does "decodability rises after occlusion onset" become visible. That re-alignment lives in `reproduce_fig5b.py` (next M4 task), not in the decoder module.

The other three loss variants (`mov`, `vis-mov`, `vis-sim-mov`) may also show different baseline decodability — `mov` (Intercept) only supervises the endpoint, no per-timestep ball position, so its hidden states might not encode initial conditions as eagerly. M5's full sweep will tell us.

### Pinned design choices (matter for downstream M4)

- **Decoder is `LinearRegression`, not PLS.** Rajalingham's `linear_regress_grouped` uses PLS by default because its target is multi-output. Our endpoint target is 1-D (a single scalar per condition), so PLS adds no value over plain OLS. The CV strategy (`GroupKFold` across conditions) matches theirs exactly.
- **`valid_mask` excludes invalid (cond, t) cells from per-timestep r/RMSE only**, not from training. If condition c is masked invalid at t, c's row is dropped from the test-fold Pearson at t but the decoder still trains on whatever valid rows exist in the train fold at t. NaN propagates cleanly through `np.nanmean` for all-invalid columns; the documented "Mean of empty slice" warning is suppressed at source since the NaN itself is the signal.
- **Mask names exposed by `load_dmfc_neural`** are the canonical 8 (`pretrial_pad0`, `start_end_pad0`, `start_occ_pad0`, `occ_end_pad0`, `f_pad0`, `occ_pad0`, `start_pad0`, `half_pad0`). The Zenodo pickle ships ~600 `_rollN` shifted variants used for cross-validated time-shift analyses; pass them via `extra_masks=` if needed instead of paying the dict-construction cost by default.
- **`load_decode_dmfc` exposes all 11 `res_decode` entries**, not just the last. Looking at `r_mu[:3]` across the 11, they're a smooth sweep of some scaling parameter (likely a regularization or PLS-component count). The figure script can pick the right one when it lands; the loader stays neutral.

## Milestone 4 progress 5 — DMFC time-axis alignment + two-stage decoder + 4-variant figure (2026-04-27, end of session)

### What landed in this session

- **DMFC time-axis fix**. The original `figures/fig5b_pilot.png` had the DMFC curve shifted right by 300 ms — its rapid rise was visible at ~500 ms instead of the paper's reported ~200 ms. Root cause: bin 0 of `all_hand_dmfc_dataset_50ms.pkl` is **not** motion onset. Per Methods (paper page 10), trials include a 300 ms pre-trial period during which the ball is rendered stationary at `(x₀, y₀)` before motion begins. In the dataset:
  - `t_from_start[0, 0] = -6` (in 50 ms bin units) → bin 0 is at t = -300 ms.
  - `t_from_start[0, 6] = 0` → bin 6 is motion onset (uniform across all 79 conditions).
  - `ball_pos_x` is NaN for bins 0–5; first finite value at bin 6 confirms motion-onset alignment.
  - `start_end_pad0` mask becomes valid at bin 6 across all conditions.
  
  Fix in `dmfc_curve()`: read `motion_onset_bin = argmax(t_from_start[0] >= 0)` and offset the source time grid by `-motion_onset_bin × bin_ms`. The IN's `hidden_states.npz` (env-side `integrate_trajectory` step 0 = motion onset) and the RNN's `r_start1_all` (named "_start1_" = aligned to motion onset) were already correctly aligned, so only DMFC needed the shift. Post-fix DMFC reaches r ≈ 0.78 at 200 ms and plateaus at ~0.85 from 250–1000 ms — quantitatively consistent with the paper's published Fig. 5B.

- **All 4 IN loss-variant pilots trained**. Seeds 0, effect_dim=10, 50K steps each (~13 min CPU per run, three in parallel for ~25 min wall clock). `vis-mov` and `vis-sim-mov` converged cleanly on seed=0; **`mov` (Intercept) diverged to NaN by step 800 on seed=0** and was retrained on seed=1 (final loss 0.028, effect_receivers std 0.80). Failure mode: with only output[6] supervised, the gradient signal on the 7-d output head is too sparse to constrain the recurrent loop's unsupervised channels — they drift, the head re-tugs the recurrence, runaway. Existing grad-clip @ norm 1.0 isn't sufficient for Intercept. M5 needs an Intercept-specific HP pass (lower lr, stricter clip, or weight decay) before launching the full 40-run sweep.

- **`dmfc/analysis/two_stage_endpoint.py`** (Rajalingham Supplementary Fig. S8D analog). Two-stage decoder: `state(t) → (x, y, dx, dy)(t)` then `kinematics → endpoint`. Kinematics built by integrating the canonical 79 conditions on the 50 ms grid via the env's `integrate_trajectory` + `_resample_to_bins`. Returns three curves per timestep (direct, kinematics-mediated, kinematics-only) plus a per-axis state→kinematics curve. 7 tests including a synthetic factor-based design (per-condition kinematic factor + small bin noise so the kinematics-only baseline carries condition identity at every t) and pilot smoke. Pilot results across all 4 IN variants:

  | Variant | Direct r | Kinmed r | Kinonly r | Gap (direct − kinmed) |
  |---|---:|---:|---:|---:|
  | Intercept (seed=1)  | 0.823 | 0.674 | 0.787 | 0.149 |
  | Vis (vis-mov)       | 0.830 | 0.666 | 0.787 | 0.164 |
  | Vis+Occ (vis-sim-mov)| 0.976 | 0.843 | 0.787 | 0.133 |
  | Vis&Occ (sim-mov)   | 0.897 | 0.777 | 0.787 | 0.120 |

  Two structural findings:
  1. **Kinonly = 0.787 is uniform** across variants (it depends only on true kinematics → endpoint, not on the model). It's the linear ceiling for "endpoint inferred from instantaneous kinematics, ignoring bounce sign-flips".
  2. **Direct > kinonly in every variant** (gaps 0.04–0.19). The IN's effect_receivers contain endpoint-relevant information *beyond* what's mechanically inferable from instantaneous `(x, y, dx, dy)`. Most plausibly bounce-aware future-trajectory inference: the IN can resolve "will this bounce" / "did this bounce" from accumulated history, while a single-timestep linear decoder of `(x, y, dx, dy)` cannot. This is a defensible "the IN computes something" finding for the writeup.

- **`reproduce_fig5b.py` x-axis cropped to 0–1200 ms by default** (display-only). Curves are still computed on the full window; new `--xlim-ms MIN MAX` CLI flag overrides. Matches the paper's framing (early-prediction window) and visually de-emphasizes late-trial DMFC volatility from conditions running out of valid bins.

### Pinned design choices that matter for M5 / writeup

- **`figures/fig5b_4variants_aligned.png` is the current canonical figure**: motion-onset aligned, 0–1200 ms cropped, all 4 IN variants overlaid. The four IN curves overlap at r ≈ 1.0 from t=0 onward — input-asymmetry effect, robust across loss variants. The two-stage analysis is the honest control for that.
- **Two-stage panel for the figure is deferred** per user direction; the analysis exists and can be plotted at any time. When it lands, the natural form is a second panel beside the current one showing direct vs. kinematics-mediated curves per variant.
- **The Intercept failure with seed=0 is a real M5 risk.** Diverged at step 800 with the same lr/clip settings that converge for the other three variants. Adding an Intercept-specific config or HP override is a hard prerequisite for the M5 sweep.

### Carry-forward to M4 final close

- `reproduce_fig4.py` (secondary deliverable, PRD F4): now mostly assembly. Loop the 192 RNN models, compute IN's NC and SI on each pilot, plot the 2-panel scatter. The building blocks (`rdm.py`, `neural_consistency.py`, `simulation_index.py`) all exist and are tested.
- Pipeline validation (PRD S1): for each gabor_pca RNN, compute our NC and SI from `per_model[fn]['data_neur_nxcxt']` and compare to `df.loc[fn, 'pdist_similarity_occ_end_pad0_euclidean_r_xy_n_sb']` and `df.loc[fn, 'decode_vis-sim_to_sim_index_mae_k2']`. Match within ~1e-3 ⇒ pipeline correct. This is the gate to M5.

## Milestone 4 progress 4 — Fig. 5B input-asymmetry diagnosis (2026-04-27, post-figure)

Generated `figures/fig5b_pilot.png` (single sim-mov pilot) and immediately flagged a problem: the IN curve sits at r ≈ 1.0 from t=0 onward, well above DMFC's rapid-rise plateau (~0.85 starting ~500ms). Trained pilots for the other three loss variants and confirmed: **all four IN variants exhibit the same r ≈ 1 from t=0 behavior** (`figures/fig5b_3of4.png` — sim-mov, vis-mov, vis-sim-mov; mov diverged on seed=0 and is being retrained on seed=1). The behavior is structural to the IN's input format, not specific to which loss variant is in use.

### Root cause: input asymmetry vs. Rajalingham's RNNs

Confirmed by reading the paper (s41467-024-54688-y.pdf, page 12, "RNN models" section): **"RNNs were trained to map a series of visual inputs (pixel frames) to a movement output."** Their RNNs receive raw pixel frames (PCA-compressed Gabor features per the Zenodo release). Our IN receives `(x, y, dx, dy, visible, is_ball, is_paddle)` as object features — i.e., **ground-truth ball position AND velocity from frame 1**.

This is a structural head-start: a single pixel frame contains ball *position* but not *velocity* (you need ≥2 frames for finite-difference velocity). With Mental Pong's deterministic kinematics (constant speed, ≤1 reflection at y=±9.6°, paddle line at x=10°), the endpoint is closed-form computable from `(x₀, y₀, dx₀, dy₀)` — four scalars. So a linear decoder of the IN's t=0 effect_receivers (a non-linear transform of those four scalars) can trivially recover the endpoint.

### The paper itself flagged this exact confound

Page 9, ¶2: *"we considered the possibility that the early signals, which carry information about the ball's initial position, predict the endpoint because the initial and endpoints are correlated."* Rajalingham controlled for this via two-stage decoding (Supplementary Fig. S8D): they decoded position from early DMFC, then asked whether decoded position alone could predict endpoint. Position alone could not; position+velocity could. So even DMFC's rapid rise *might* leverage the same kinematic shortcut we're handing the IN for free.

### Implications for the writeup

1. **The figure is faithful to what we built**, but the rapid-vs-slow comparison vs. RNNs is not apples-to-apples on input format. PRD/NG3 already documented this as the Option-2-inputs decision; the writeup must carry the caveat front-and-center.
2. **All four IN variants showing r ≈ 1 from t=0 is informative**: the input-asymmetry effect dominates over loss-variant differences. If we want to study how loss-variant choice affects representational geometry, Fig. 4 (Neural Consistency × Simulation Index) is the right axis — Fig. 5B is dominated by the input format.
3. **Honest mitigation paths** (in order of cost):
   - (a) Two-stage decoding control (S8D analog): decode `(x, y, dx, dy)` from IN states at each time t, then decode endpoint from those — this isolates the IN's *additional* contribution beyond raw initial-condition decodability. Modest scope; lives in the writeup methods.
   - (b) **Pixel-input IN** (PRD NG3 future work): drop a small CNN front-end before the IN's object-graph stage. Big scope; not for this project.
   - (c) Discuss in writeup limitations section without modifying the analysis. Cheapest; honest if (a) and (b) are out of scope.

### IN training stability — Intercept variant divergence

Intercept (`mov`, supervise only output[6] = final intercept y) diverged to NaN by step 800 with seed=0. Vis, Vis+Occ, Vis&Occ all converged cleanly at the same lr/grad-clip settings. Diagnosis: with only one supervised output (a single scalar per condition), the gradient signal is too weak to constrain the unsupervised output channels (output[0:6]); these drift, the output-head weights tug back into the recurrent loop, and the recurrent dynamics blow up. This is a more aggressive instance of the M3 "recurrent loop has no fixed point" failure mode that gradient clipping at norm 1.0 already fixed for the other variants.

Retrying with seed=1; if that also fails, fall back options are (in order):
1. Lower lr to 5e-4
2. Lower grad clip norm to 0.5
3. Add weight decay 1e-4
4. Reduce max_steps to 20K (might be overshooting convergence)

For the M5 sweep, this implies **Intercept training will need its own hyperparameter pass** before launching all 40 runs. Worth flagging in the M5 task description.

## Milestone 4 progress 3 — Fig. 4 building blocks + Fig. 5B (2026-04-27)

Closed out the four remaining M4 modules in one pass: `rdm.py`, `neural_consistency.py`, `simulation_index.py`, `reproduce_fig5b.py`. Test count 47 → 87 (+40); ruff + mypy clean. Plan file: `~/.claude/plans/depending-on-the-difficulty-declarative-rivest.md`.

### Headline result: extended Fig. 5B reproduces the rapid-vs-slow rise

`figures/fig5b_pilot.png` shows three groups on a shared 50 ms time axis (linear-interp from RNN's 41 ms native):

- **DMFC** (orange): rapid rise from ~0 to ~0.85 by t≈500 ms, sustained until t≈2400 ms, then noisy decline as fewer conditions remain valid past the longest-trial duration. Matches the published Fig. 5B shape qualitatively.
- **Four RNN class curves** (greys, mov / vis-mov / vis-sim-mov / sim-mov, all gabor_pca subset): slow gradual rise reaching ~0.7 by t≈1700 ms. Curves cluster tight (band overlap); little class differentiation visible at the published-iteration aggregate level.
- **IN** (blue, sim-mov pilot): r ≈ 1.0 from t=0 onward, with a drop near the end of the trial as valid windows shrink. **This is exactly what the M3 closeout predicted** for sim-mov: deterministic kinematics + occluded-ball + intercept supervision lets the IN embed full initial-condition information into `effect_receivers` from the very first step. The interesting comparison comes when the M5 sweep adds the `mov`/`vis-mov`/`vis-sim-mov` IN variants — those don't supervise occluded-ball position, so the IN should have less reason to encode endpoint-determining info from t=0.

The figure validates the **pipeline** more than it validates the **science**: the science question (does the IN match DMFC's rapid rise across loss variants?) is decided by the M5 sweep, not by the pilot.

### Implementation gotchas worth remembering

1. **Rajalingham's RDM cells are (cond, t) pairs, not conditions.** `get_state_pairwise_distances` (`code/utils/phys_utils.py:371`) takes states with axis order `(n_units, n_cond, T)` (line 248 of `RnnNeuralComparer.py` confirms the transpose), applies a 2-D mask, and pools all valid cells across both conditions and time into a `(n_cells, n_units)` sample matrix. The RDM is over those pooled cells. We adopted the same convention but with the more natural `(n_cond, T, n_features)` axis order so it composes with `flatten_receivers`. Both yield identical RDMs — only the wrapper transposes are different.
2. **`r_start1_all` is `(100 iter, 90 t)`, not `(100 iter, 79 cond, 90 t)`** as an earlier SCRATCHPAD line implied. The 79-condition aggregation is already baked into the per-iteration r — verified live. The full per-condition predictions are in `yp`/`yt` shape `(100, 79, 90)`, which is what would be needed for `--align onset`.
3. **`df['filename']` is the linker between `df` and `per_model`**, not `df['name']` (which is a higher-level grouping path with no checkpoint hash). The per_model keys are full checkpoint paths.
4. **Spearman-Brown can push noise-corrected r above 1.0** when `r_xx` is moderate and the underlying RDM correlation is high; this is mathematically valid but worth flagging in plots so a reader doesn't think the metric is broken. Both the `r_xy_n_sb` (full SB-corrected) and `r_xy_n` (no SB) variants are returned for transparency.
5. **The `--align onset` flag is wired up but raises NotImplementedError**. The honest implementation requires per-RNN per-condition decoding curves (because shifting curves by per-condition `t_occ` requires resolution at the condition level), and Rajalingham's release pre-aggregates across the 79 conditions in `r_start1_all`. The path forward is to recompute per-RNN per-condition r from `yp`/`yt` (`(100, 79, 90)`) — feasible but deserves its own session.

### Pinned design choices that matter for M5/M6

- **DMFC per-timestep curve must be re-computed by us.** The released `decode_*.pkl` has per-target r values across 23 behaviors but not per-timestep curves. We run `decode_endpoint` on `responses_reliable` `(1889, 79, 100)` (transposed to `(79, 100, 1889)`) with target `behavioral['ball_final_y'][:, 0]` and mask `start_end_pad0`. The DMFC curve is therefore *our* product, not Rajalingham's; if the writeup needs an apples-to-apples comparison with their published DMFC line, we'd have to reach out for their per-timestep numbers.
- **Neural Consistency uses occlusion-only mask `occ_end_pad0`** (per Rajalingham `RnnNeuralComparer.compare_representational_geometry_rnn_to_neural_data`, line 220 onward). Implemented in `neural_consistency_from_states(..., mask=...)` so the caller picks; our convention will be `dmfc.masks['occ_end_pad0']` for Fig. 4D.
- **Simulation Index uses train-mask = visible+occluded valid bins (`output_vis-sim`), test-mask = occluded-only (`output_sim`)**. For our IN runs that's `train_mask = valid_mask`, `test_mask = valid_mask & ~visible_mask`. The Zenodo k=2 default is what Fig. 4E plots; we expose `k` as a parameter but default to 2.
- **No pipeline validation in this plan.** Per the plan file, validating our `neural_consistency` and `simulation_index` against Rajalingham's published values (`df['pdist_similarity_*_r_xy_n_sb']` and `df['decode_vis-sim_to_sim_index_mae_k2']`) is its own follow-up. Until that happens, our Fig. 4 numbers should be considered *internally consistent* but not yet *externally validated*.

### Carry-forward to next M4 task (`reproduce_fig4.py` + pipeline validation)

- The Fig. 4 reproduction is now mostly assembly: load all 192 RNN models' precomputed metrics, load DMFC, compute IN's NC and SI, plot a 2-panel scatter (NC × SI) with per-class swarms.
- **Pipeline validation (PRD S1)** is the substantive remaining M4 step. For each gabor_pca RNN: load its `data_neur_nxcxt` from `per_model[fn]['data_neur_nxcxt']` (per the SCRATCHPAD note — verify this key exists; if not, re-extract from somewhere else in the pickle), then run our `neural_consistency_from_states` and `simulation_index` and compare to `df.loc[fn, 'pdist_similarity_occ_end_pad0_euclidean_r_xy_n_sb']` and `df.loc[fn, 'decode_vis-sim_to_sim_index_mae_k2']`. Match within ~1e-3 = pipeline correct.
- The M3 carry-forward note "looking at curves re-aligned to occlusion onset" still stands as the right move for the writeup's Fig. 5B comparison once the M5 sweep lands. With per-condition `t_occ` shifts and per-condition predictions for all three groups, the rapid-vs-slow rise structure becomes more interpretable than on a shared trial-start axis.

### Carry-forward to next M4 task (`rdm.py` → `neural_consistency.py` → `simulation_index.py` → `reproduce_fig5b.py`)

- **Re-aligning curves to `t_occ`**: per condition, `ConditionSpec.t_occ_steps × RNN_STEP_MS / DMFC_BIN_MS` gives the IN-grid bin where occlusion starts. For each cond, slice `effect_receivers[c, t_occ_bin:t_occ_bin+W, :]` then average across cond. That's the input to a Fig. 5B-style decoder if we want a clean post-occlusion curve.
- **Per-timestep DMFC curves**: not yet confirmed where they live. The `decode_*.pkl` we loaded has `(23,)` r_mu — that's per-target, not per-timestep. The per-timestep DMFC curve may have to be computed by us by running a `linear_regress_grouped`-style decoder on `DMFCData.responses` directly, mirroring what we did for the IN. If so, `endpoint_decoding.decode_endpoint` is already a drop-in (the loader returns the right `(n_neur, 79, 100)` shape; transpose to `(79, 100, n_neur)` and feed it).
- **Time alignment between IN, RNN, DMFC**: IN at 50 ms × 72 = 3600 ms; DMFC at 50 ms × 100 = 5000 ms; RNN at 41 ms × 90 = 3690 ms. For the Fig. 5B overlay, all three need to land on the same x-axis. Resampling the RNN's 41 ms grid to 50 ms is the cleanest move; alternatively, plot all three on time-relative-to-occlusion in their native bins and let the shared `t=0` carry the comparison.
