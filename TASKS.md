# Tasks — interaction-networks-dmfc

Generated from PRD.md + PLANNING.md. Work one milestone at a time. Mark `[x]` when complete.
`/summcommit` keeps this file in sync with real progress.

## Milestone 1 — Scaffolding and information gathering

Goal: everything needed to start real work exists — repo structure, pinned deps, Rajalingham data inspected, upstream code understood. No training yet.

- [x] Create the `dmfc/` package skeleton per PLANNING: `envs/`, `models/`, `training/`, `analysis/`, `rajalingham/`, each with a stub `__init__.py`. (2026-04-25)
- [x] Create `configs/`, `runs/`, `data/dmfc/`, `tests/`, `notebooks/` directories. Add `.gitignore` entries for `runs/`, `data/dmfc/`, `.venv/`, `__pycache__/`, `*.pt`. (2026-04-25; `data/dmfc` is a symlink to `~/Downloads/MentalPong/data`)
- [x] Set up `pyproject.toml` with uv + pinned deps (torch, numpy, scipy, scikit-learn, matplotlib, seaborn, pandas, pyyaml, pytest, ruff, mypy). Record exact versions in VERSIONS.md. (2026-04-25; 50-package tree resolved via `uv lock`)
- [x] Create `.claude/commands/freshstart.md` and `.claude/commands/summcommit.md` (already in `.claude/commands/` from the project-bootstrap run).
- [x] Verify upstream higgsfield IN code runs out of the box: identify entry points, run any existing demo, confirm what hidden-state extraction hooks exist. Log findings in SCRATCHPAD. **Done via read-through** of `Interaction Network.ipynb` (2026-04-24); `effect_receivers` identified as the per-object hidden state. Notebook will be exercised live in Milestone 3.
- [x] Clone jazlab/MentalPong locally (as a sibling directory or a read-only git reference — NOT inside this repo). **Satisfied (2026-04-25)** by the Zenodo release at `~/Downloads/MentalPong/`, which bundles both `code/` (jazlab/MentalPong source — `phys_utils.py`, `generic_plot_utils.py`, `rnn_comparisons_2021.py`, etc.) and `data/`. No separate `git clone` needed for a class project. Inspection subtasks:
  - [x] What hidden-unit values did Rajalingham actually sweep? **10, 20** (Zenodo release; 40 not present) — corrected in PLANNING.md (2026-04-25).
  - [x] What format is the Zenodo release in? **Python pickle (.pkl)** — full schema documented in SCRATCHPAD.md (2026-04-22).
  - [x] Do they release per-RNN Fig. 5B curves, or only aggregated traces? **Per-RNN individual curves released** in `offline_rnn_neural_responses_reliable_50.pkl` — file structure in SCRATCHPAD.md.
  - [x] What exactly do their RNNs receive as input? **100-dim Gabor + PCA features** (not raw pixels) — documented in SCRATCHPAD.md.
- [x] Download the Rajalingham Zenodo release into `data/dmfc/`. Write a one-paragraph README in that directory documenting what was downloaded and from where. (2026-04-25; release lives at `~/Downloads/MentalPong/`; `data/dmfc` symlinks to its `data/` subdir; provenance + recreate-on-other-machine instructions in `data/README.md`)
- [x] Resolve the four open architectural questions listed in PLANNING.md; update PLANNING or promote any remaining ambiguities into explicit tasks in later milestones. **All four resolved (2026-04-25)**:
  - [x] Zenodo release format → .pkl; schema documented in SCRATCHPAD.md.
  - [x] Per-RNN Fig. 5B curves available? → Yes; structure documented in SCRATCHPAD.md.
  - [x] Hidden-unit sweep values → 10, 20 (corrected from earlier wrong note of 40); updated in PLANNING.md experimental matrix (2026-04-25).
  - [x] Wall reflection — graph edge or env-level? **Resolved**: env-level; y = ±10° walls confirmed from Zenodo plotting code (2026-04-24).
- [x] Write a minimal `tests/test_smoke.py` that imports every `dmfc.*` subpackage and asserts they load. Run `pytest` to confirm the testing infrastructure works. (2026-04-25; 7 tests pass; `ruff check` + `ruff format --check` clean)

## Milestone 2 — Mental Pong environment

Goal: a deterministic, seed-controlled, unit-tested Mental Pong env that reproduces Rajalingham's 79 eval conditions and whose trajectories match their spec exactly.

- [x] Implement `dmfc/envs/mental_pong.py`: Gym-style env with `reset(seed, condition_id)`, `step(action)`, `render()`. Ball kinematics: rightward motion only, constant speed, reflections off horizontal walls, deterministic given seed. (2026-04-25)
- [x] Encode the visible/occluded epoch logic per Rajalingham's spec (visible/occluded epoch lengths vary by condition; ball masked when ball_x ≥ 5°; trajectory integrated at 41 ms native step, observations resampled to 50 ms bins to match neural data; bounces ∈ {0, 1}). (2026-04-25)
- [x] Load the 79 eval conditions from `valid_meta_sample_full.pkl` via `dmfc/envs/conditions.py` — preserves Rajalingham's `PONG_BASIC_META_IDX` order and reads MWK-frame columns directly (no re-derivation needed). (2026-04-25)
- [x] `tests/test_mental_pong.py`: 9 tests covering determinism, endpoint oracles for all 79 conditions (`y_occ_rnn_mwk` and `y_f_rnn_mwk` matched to ~1e-14), no-bounce, one-bounce, occluder timing, mask integrity, interception geometry, paddle invariants, and occluder-x consistency. All passing. (2026-04-25)
- [x] CLI: `python -m dmfc.envs.mental_pong --render --seed 0 --condition 0` renders a single condition; also `--animate` (GIF, real-time) and `--grid` (all 79 trajectories in one PNG) for visual inspection. Usage commands documented in SCRATCHPAD.md. (2026-04-25)
- [x] Write `dmfc/envs/conditions.py` with `load_conditions()` returning the canonical 79-condition `ConditionSpec` list (meta_index, x0, y0, dx, dy, t_f_steps, t_occ_steps, y_occ_oracle, y_f_oracle, n_bounce). Frozen by `PONG_BASIC_META_IDX` ordering. (2026-04-25)

## Milestone 3 — Models, training infrastructure, pilot runs

Goal: one IN and one flat-RNN pipeline baseline train end-to-end, produce valid run artifacts, and their hidden states can be extracted for analysis. No full sweep yet.

- [x] IN model (`dmfc/models/interaction_network.py`): vendored copy of upstream `RelationalModel`/`InteractionNetwork` at `dmfc/models/_upstream_in.py` (modernized for torch 2.4; returns `(predicted, effect_receivers)`). Wrapper `MentalPongIN` instantiates a local `_RelationalMLP` (the upstream final ReLU was dropped — see SCRATCHPAD M3 closeout for the dead-network finding). 2-object/2-directed-relation graph; recurrence via concat of prior `effect_receivers` into next-step object state; 7-d Rajalingham output head reads from concatenated effect_receivers. (Flat-RNN baseline cut.) (2026-04-26)
- [x] Object-graph construction: 2 objects (ball, paddle), 2 directed relations (ball→paddle, paddle→ball). Walls stay env-level per PLANNING decision 3. (2026-04-26)
- [x] Loss-variant system (`dmfc/training/losses.py`): all four Rajalingham variants as one `compute_loss(outputs, targets, variant, visible_mask, valid_mask)` — closed-form supervision masks per variant; tested per-index. (2026-04-26)
- [x] Training loop (`dmfc/training/train.py`): config-driven (`--config configs/in_*.yaml --seed N`), single-source seeding for `random/numpy/torch/torch.cuda`, gradient clipping at norm 1.0 (recurrent stability — see SCRATCHPAD), full run artifact to `runs/in_<variant>_h<eff>_s<seed>_<ts>_<git>/`. (2026-04-26)
- [x] Hidden-state serialization: post-train forward over the 79 → `hidden_states.npz` with `effect_receivers (79, T_max, 2, effect_dim)`, `outputs`, `targets`, `visible_mask`, `valid_mask`, `meta_index`. (2026-04-26)
- [x] Config files: `configs/in_intercept.yaml` (mov), `in_vis.yaml` (vis-mov), `in_vis_occ.yaml` (vis-sim-mov), `in_vis_and_occ.yaml` (sim-mov) — typed dataclass loader at `dmfc/training/config.py`. (2026-04-26)
- [x] Pilot training run: `Vis&Occ` (sim-mov), seed 0, effect_dim=10, 50K steps in ~13 min on CPU. Loss 36.8 → 0.05 EMA; on the 79 held-out: occluded ball_x R²=0.985, ball_y R²=0.989, intercept R²=0.997; effect_receivers std 2.15 globally / 1.31 mean across conditions. (2026-04-26)
- [x] `tests/test_training.py` + `tests/test_losses.py` + `tests/test_random_conditions.py`: artifact-writer coverage, seed determinism on CPU (sha256 match), shape sanity, supervision-mask per-index correctness, generator determinism + envelope match. 33/33 tests green; ruff + mypy clean. (2026-04-26)

## Milestone 4 — Analysis pipeline (Fig. 5B primary, Fig. 4 secondary)

Goal: all four analysis metrics from PRD A6 work end-to-end on the pilot IN runs, and the reproduce_fig5b / reproduce_fig4 scripts regenerate extended figures.

- [x] `dmfc/rajalingham/load.py`: adapter for loading the DMFC population data and the published RNN outputs from the Zenodo release. Three loaders (`load_dmfc_neural`, `load_rnn_metrics`, `load_decode_dmfc`), frozen dataclasses, `pd.read_pickle` shim for old-pandas pickles. (2026-04-26)
- [x] `dmfc/analysis/endpoint_decoding.py` (PRIMARY — Fig. 5B metric): time-resolved linear decoding of endpoint ball_y from hidden states via `GroupKFold` across the 79 conditions and per-timestep `LinearRegression`. Returns `r(T)`, `rmse(T)`, plus per-fold curves. Pilot smoke confirms M3-flagged finding: `effect_receivers` carry full endpoint info from t=0 (r ≈ 0.999 throughout). (2026-04-26)
- [x] `dmfc/analysis/rdm.py`: pairwise-distance matrix computation for hidden states (Euclidean or correlation distance, NaN-safe via sklearn). Mirrors Rajalingham's `get_state_pairwise_distances`. 11 tests covering identity, scipy-pdist parity, mask invariance, NaN handling. (2026-04-27)
- [x] `dmfc/analysis/neural_consistency.py`: noise-adjusted correlation per Rajalingham Eq. 4 (`r_xy_n_sb = r_xy / sqrt(SB(r_xx) · SB(r_yy))`, `SB(r) = 2r/(1+r)`). Uses `compute_rdm` for both model and neural splits; for deterministic IN, `r_yy = 1`. 10 tests covering perfect-consistency, pure-noise, SB-correction unit cases, noise-correction-lifts-score, end-to-end orthogonal-embedding smoke. Validation against published RNN class means deferred to a follow-up plan (PRD S1). (2026-04-27)
- [x] `dmfc/analysis/simulation_index.py`: k-fold OLS decoder of `ball_xy` from states; train on visible+occluded mask, test on occluded-only mask of held-out conditions. Reports MAE/RMSE/rho per coord; `si = mean(mae)` is the headline scalar. 11 tests; pilot smoke confirms the sim-mov pilot decodes occluded ball position with `si < 5°`, `rho > 0.5` per coord. Validation against published SI distribution deferred to S1 follow-up. (2026-04-27)
- [x] `dmfc/analysis/reproduce_fig5b.py` (PRD F3 — PRIMARY DELIVERABLE): glob `--in-runs`, `--rajalingham-data`, `--out`, `--align {start,onset}`, `--xlim-ms MIN MAX`. Composes DMFC per-timestep curve (decode_endpoint on responses_reliable), four RNN class curves (from `r_start1_all` aggregated within `loss_weight_type` on gabor_pca), and one IN curve per run. Default `start` alignment interpolates RNN's 41 ms grid onto the shared 50 ms axis; `onset` raises NotImplementedError (deferred to M5). **Display crops to 0–1200 ms by default** to match the paper's framing. **DMFC time-axis now correctly aligned to motion onset** via `t_from_start`-derived offset (bin 6 = motion start; previously bin 0 was treated as t=0, shifting the curve right by 300 ms — now fixed). Pilot figures: `figures/fig5b_pilot.png` (single sim-mov IN), `figures/fig5b_4variants_aligned.png` (all four IN variants, motion-onset aligned). DMFC rapid rise to ~0.78 by 200 ms ✓, RNN classes rise to ~0.5–0.65 by 1200 ms ✓, IN variants overlap at r ≈ 1.0 from t=0 (input-asymmetry; see SCRATCHPAD progress 4). 8 tests; full pipeline smoke green. (2026-04-27)
- [x] `dmfc/analysis/two_stage_endpoint.py` (Rajalingham Supplementary Fig. S8D analog): control for IN's input-asymmetry head start. Stage 1: linear `state(t) → (x, y, dx, dy)(t)`. Stage 2: linear `kinematics → endpoint`. Reports direct, kinematics-mediated, and kinematics-only curves plus per-axis state→kinematics r. Helper `kinematics_for_canonical_79()` integrates the 79 conditions on the 50 ms grid. Pilot results across all 4 IN variants: direct r ∈ [0.82, 0.98], kinmed r ∈ [0.67, 0.84], kinonly r = 0.79 (uniform across variants since it depends only on true kinematics). **Direct > kinonly in all variants → IN encodes endpoint info beyond instantaneous kinematics** (most plausibly bounce-aware future-trajectory inference). 7 tests; pilot smoke green. (2026-04-27)
- [x] **Train remaining 3 IN loss-variant pilots** (mov / vis-mov / vis-sim-mov) at effect_dim=10, seed=0, 50K steps each, ~13 min CPU. `vis-mov` and `vis-sim-mov` converged cleanly (final loss 0.014, 0.045). **`mov` (Intercept) diverged to NaN by step 800 with seed=0**; retrained with seed=1 and converged (final loss 0.028). Failure mode: with only one supervised output (a scalar per condition), gradient signal is too weak to constrain unsupervised channels → recurrent runaway despite existing grad-clip @ norm 1.0. M5 needs an Intercept-specific hyperparameter pass before the full sweep (lower lr, stricter clip, or weight decay). (2026-04-27)
- [x] `dmfc/analysis/reproduce_fig4.py` (PRD F4 — SECONDARY DELIVERABLE): two-panel figure with NC (Fig. 4D analog) and SI (Fig. 4E analog). Reads pre-computed RNN swarm values from the Zenodo `rnn_compare_*.pkl` summary (NC) and `offline_rnn_*.pkl` df (SI), filters to `gabor_pca` (96 of 192 models), computes per-IN-pilot points via `neural_consistency_from_states` and `simulation_index`. Auto-skips diverged runs. Pilot figure: `figures/fig4_pilot.png`. Pilot results: IN's SI is uniformly lower than the RNN swarm across all four loss variants (Intercept 1.79 deg vs RNN ~3, Vis 1.69 vs ~2.5, Vis+Occ 0.16 vs ~1.7, Vis&Occ 0.77 vs ~1.4); IN's NC is more variable — Vis&Occ (0.48) above the RNN swarm, Vis+Occ (-0.08) below. (2026-04-27)
- [x] Pipeline validation (PRD S1): rescoped to a math-identity check against Rajalingham's reference functions on synthetic input — `raw['state']` is **not** in the offline RNN pickle (verified live; SCRATCHPAD line 38 was wrong), so per-RNN comparison was infeasible. New `dmfc/analysis/validate_pipeline.py` imports `phys_utils.get_state_pairwise_distances` and `RnnNeuralComparer.get_noise_corrected_corr` from `~/Downloads/MentalPong/`, runs both pipelines on a fixed synthetic input, and confirms RDM, NC (`r_xy`, `r_xx`, `r_yy`, `r_xy_n`, `r_xy_n_sb`), and SI (per-coord MAE) match **bit-for-bit (max diff 0.0e+00)**. Caveat: cannot reproduce per-RNN published values, so the SI/NC RNN swarms in Fig. 4 use Rajalingham's pre-computed numbers, not our re-computation. M5 is unblocked. (2026-04-27)
- [ ] Two-stage panel for `reproduce_fig5b.py`: overlay direct vs. kinematics-mediated curves so the writeup figure makes the input-asymmetry control visible. Deferred per user direction; the analysis module already exists and runs end-to-end.

## Milestone 5 — Full IN sweep + primary results

Goal: all IN training runs complete, primary Fig. 5B result is known, secondary Fig. 4 result is known.

- [ ] Generate config files for the full experimental matrix: 4 loss variants × n_hidden {10, 20} × 5 seeds = 40 runs per PLANNING.
- [ ] **Intercept-specific hyperparameter pass before the sweep** (M4 carry-forward): try lr=5e-4, grad clip @ 0.5, or weight decay 1e-4. Seed=0 reliably diverges at default settings; need a configuration that converges across all 5 seeds.
- [ ] Execute the sweep. Log progress in SCRATCHPAD; flag any runs that don't converge or produce invalid artifacts.
- [ ] Regenerate Fig. 5B and Fig. 4 with the full IN sweep. Commit the figures + the list of consumed run directories.
- [ ] Statistical tests:
  - [ ] For Fig. 5B: IN vs. RNN-class-mean time-to-threshold-r, IN vs. RNN RMSE-AUC over 0–1200 ms. Wilcoxon rank-sum across IN seeds vs. RNN class members.
  - [ ] For Fig. 4: is the IN swarm's Neural Consistency distribution significantly different from (or within) the RNN class distributions? Report partial R² of Neural Consistency ~ Simulation Index with IN included vs. without.
- [ ] Answer PRD S2 (primary) in SCRATCHPAD: does IN match DMFC's rapid-rise, match RNNs' slow-gradual, or land intermediate? Quantified, not just eyeballed.
- [ ] Answer PRD S3 (secondary): does the IN swarm land within the RNN swarm range on Fig. 4? This is the no-regression check.

## Milestone 6 — Ablation and writeup

Goal: the ablation is run on both G1 and G2 metrics, the class writeup is drafted, all figures are regenerable from scripts.

- [ ] Implement ablated IN variants in `dmfc/models/`:
  - [ ] IN with pairwise interaction terms zeroed out (keeps object decomposition, removes relations).
  - [ ] IN with graph aggregation replaced by mean pooling (keeps objects + relations, removes structured aggregation).
- [ ] Run the ablation matrix: 2 ablated variants × 4 loss variants × 1 hidden-unit value (the best-performing from Milestone 5) × 5 seeds = 40 runs.
- [ ] Ablation analysis: Fig. 5B and Fig. 4 metrics for each ablated variant. Produce a 2-panel figure showing degradation on each axis.
- [ ] Answer PRD S4 (tertiary): which component of the IN drives G1? Does the same component drive G2? Report the dissociation (if any) honestly.
- [ ] Draft the writeup: intro → methods → three result sections (Fig. 5B replication + IN, Fig. 4 replication + IN, ablation) → discussion → limitations (including explicit note on Option 2 inputs and pixel-input future work).
- [ ] Verify every writeup figure is regenerated from a script in `dmfc/analysis/` pointing at a named run directory (PRD S5, CONSTITUTION writeup-discipline).
- [ ] Submit (PRD S6).

## Backlog

Ideas worth considering later; not committed to the milestones above.

- [ ] Pixel-input variant of the IN with a small perception front-end (PRD NG3 future-work note). Would be a natural follow-up project.
- [ ] Per-monkey analysis: do results hold if analysis is restricted to monkey P's 1552 units vs. monkey M's 337? Rajalingham show their findings hold across both; we should check too before final writeup.
- [ ] Behavioral comparison (Fig. 4B analog): does the IN produce paddle-ball endpoint errors with a similar pattern to monkeys? Rajalingham have a "Behavioral Consistency Score" we could compute.
- [ ] RNN retraining sanity check: retrain a small subset of Rajalingham's RNNs from their code and verify we reproduce their Fig. 5B curves. Only pursue if the adapter pipeline in Milestone 4 fails validation and we can't isolate the cause.

## Completed

Move tasks here as they're finished, with a date.
Example format: `- [x] Set up repo scaffolding (2026-04-21)`

- [x] **Milestone 1 — Scaffolding and information gathering** complete (2026-04-25). All sub-tasks `[x]` above. Repo ready for Milestone 2: `dmfc/` skeleton in place, deps installed via `uv` (`uv.lock` tracked, 50 packages, torch 2.4.1 / numpy 1.26.4 / Python 3.11.15), `data/dmfc` symlinked to the local Zenodo download, `tests/test_smoke.py` green (7 passed), `ruff` clean.
- [x] **Milestone 2 — Mental Pong environment** complete (2026-04-25). Deterministic env reproduces all 79 conditions to ~1e-14 against Zenodo endpoint oracles (`y_occ_rnn_mwk`, `y_f_rnn_mwk`). 16/16 tests green; `ruff` + `mypy` clean. Two material corrections to Milestone-1 documentation landed alongside: paddle is at x=+10° (right wall), occluder spans x ∈ [+5°, +10°]; ball reflects at y=±9.6° (not ±10°) — the latter discovered by integrator validation against the metadata. Repo ready for Milestone 3.
- [x] **Milestone 3 — Models, training infrastructure, pilot run** complete (2026-04-26). Vendored upstream IN (modernized for torch 2.4) at `dmfc/models/_upstream_in.py`; recurrent `MentalPongIN` wrapper at `dmfc/models/interaction_network.py`; random-condition generator; 4-variant loss-mask system; config-driven training loop with full run-artifact serialization; pilot trained sim-mov/seed=0/effect_dim=10 to loss 0.05 EMA in ~13 min CPU, R²>0.98 on supervised quantities for the 79 held-out, effect_receivers std 1.31 across conditions. 33/33 tests green; ruff + mypy clean. Two design deviations landed mid-milestone (final ReLU dropped from relational MLP; gradient clipping at norm 1.0) — full rationale in SCRATCHPAD M3 closeout. Repo ready for Milestone 4 (analysis pipeline) and gershman_lab clone for the M5 sweep.
- [x] **Milestone 4 substantially complete (2026-04-27)**: 4 of 6 sub-tasks landed plus a stretch deliverable. `rdm.py` + `neural_consistency.py` + `simulation_index.py` + `reproduce_fig5b.py` (PRD F3 PRIMARY) all green on 47→94 tests. Stretch: `two_stage_endpoint.py` (Rajalingham S8D analog) implemented and run on all 4 IN pilots — direct r ∈ [0.82, 0.98], gap to kinematics-mediated r is 0.12–0.16, and direct > kinonly in every case → IN encodes endpoint info beyond raw kinematics. All 4 loss-variant pilots trained (`mov` needed seed=1 retry after seed=0 NaN divergence). `figures/fig5b_4variants_aligned.png` is the current canonical figure: motion-onset aligned, 0–1200 ms cropped, all 4 IN variants overlaid on the 4 RNN classes and DMFC. Open for M4 final close: `reproduce_fig4.py` and pipeline validation (PRD S1). Two-stage figure panel deferred per user direction.
- [x] **Milestone 4 closed (2026-04-27)**: final two sub-tasks landed. `reproduce_fig4.py` (PRD F4) green — pilot figure at `figures/fig4_pilot.png` shows IN points overlaid on the 96-model gabor_pca RNN swarm; IN's SI uniformly better than RNNs across all 4 loss variants. Pipeline validation (PRD S1) rescoped: Zenodo doesn't ship per-RNN hidden states (`raw['state']` not in pickle), so we ran a math-identity check against Rajalingham's reference functions (`phys_utils.get_state_pairwise_distances`, `RnnNeuralComparer.get_noise_corrected_corr`) on synthetic input — RDM, NC (5 quantities), and SI all match at 0.0e+00 absolute diff. New `dmfc/analysis/validate_pipeline.py`. Suite total 94→104. M5 unblocked; only Intercept HP pass remains as M5 prereq (already noted in M5 task list).
