# Tasks — interaction-networks-dmfc

Generated from PRD.md + PLANNING.md. Work one milestone at a time. Mark `[x]` when complete.
`/summcommit` keeps this file in sync with real progress.

## Milestone 1 — Scaffolding and information gathering

Goal: everything needed to start real work exists — repo structure, pinned deps, Rajalingham data inspected, upstream code understood. No training yet.

- [ ] Create the `dmfc/` package skeleton per PLANNING: `envs/`, `models/`, `training/`, `analysis/`, `rajalingham/`, each with a stub `__init__.py`.
- [ ] Create `configs/`, `runs/`, `data/dmfc/`, `tests/`, `notebooks/` directories. Add `.gitignore` entries for `runs/`, `data/dmfc/`, `.venv/`, `__pycache__/`, `*.pt`.
- [ ] Set up `pyproject.toml` with uv + pinned deps (torch, numpy, scipy, scikit-learn, matplotlib, seaborn, pandas, pyyaml, pytest, ruff, mypy). Record exact versions in VERSIONS.md.
- [ ] Create `.claude/commands/freshstart.md` and `.claude/commands/summcommit.md` (copied verbatim from project-bootstrap skill — already done by the bootstrap run).
- [ ] Verify upstream higgsfield IN code runs out of the box: identify entry points, run any existing demo, confirm what hidden-state extraction hooks exist. Log findings in SCRATCHPAD.
- [ ] Clone jazlab/MentalPong locally (as a sibling directory or a read-only git reference — NOT inside this repo). Inspect:
  - [x] What hidden-unit values did Rajalingham actually sweep? **10, 20** (Zenodo release; 40 not present) — corrected in PLANNING.md (2026-04-25).
  - [x] What format is the Zenodo release in? **Python pickle (.pkl)** — full schema documented in SCRATCHPAD.md (2026-04-22).
  - [x] Do they release per-RNN Fig. 5B curves, or only aggregated traces? **Per-RNN individual curves released** in `offline_rnn_neural_responses_reliable_50.pkl` — file structure in SCRATCHPAD.md.
  - [x] What exactly do their RNNs receive as input? **100-dim Gabor + PCA features** (not raw pixels) — documented in SCRATCHPAD.md.
- [ ] Download the Rajalingham Zenodo release into `data/dmfc/`. Write a one-paragraph README in that directory documenting what was downloaded and from where.
- [ ] Resolve the four open architectural questions listed in PLANNING.md; update PLANNING or promote any remaining ambiguities into explicit tasks in later milestones.
  - [x] Zenodo release format → .pkl; schema documented in SCRATCHPAD.md.
  - [x] Per-RNN Fig. 5B curves available? → Yes; structure documented in SCRATCHPAD.md.
  - [x] Hidden-unit sweep values → 10, 20 (corrected from earlier wrong note of 40); updated in PLANNING.md experimental matrix (2026-04-25).
  - [x] Wall reflection — graph edge or env-level? **Resolved**: env-level; y = ±10° walls confirmed from Zenodo plotting code (2026-04-24).
- [ ] Write a minimal `tests/test_smoke.py` that imports every `dmfc.*` subpackage and asserts they load. Run `pytest` to confirm the testing infrastructure works.

## Milestone 2 — Mental Pong environment

Goal: a deterministic, seed-controlled, unit-tested Mental Pong env that reproduces Rajalingham's 79 eval conditions and whose trajectories match their spec exactly.

- [ ] Implement `dmfc/envs/mental_pong.py`: Gym-style env with `reset(seed, condition_id)`, `step(action)`, `render()`. Ball kinematics: rightward motion only, constant speed, reflections off horizontal walls, deterministic given seed.
- [ ] Encode the visible/occluded epoch logic per Rajalingham's spec (visible/occluded epoch lengths vary by condition; epoch masks stored in `valid_meta_sample_full.pkl`; use **50 ms bins** matching the neural data, not 16.7 ms; bounces ∈ {0, 1}).
- [ ] Load or construct the 79 eval conditions. If the Zenodo release contains them: parse and cache. If not: construct by sampling the parameter ranges Rajalingham described (`x0 ∈ [-8, 0]°`, `y0 ∈ [-10, 10]°`, `dx0 ∈ [6.25, 18.75]°/s`, `dy0 ∈ [-18.75, 18.75]°/s`) subject to the constraints.
- [ ] `tests/test_mental_pong.py`: unit tests against hand-computed expected trajectories for at least 3 fixed seeds. Cover: (a) no-bounce condition, (b) one-bounce condition, (c) occluder timing, (d) interception geometry at trial end.
- [ ] CLI: `python -m dmfc.envs.mental_pong --render --seed 0 --condition 0` renders a single condition for visual inspection (PRD F1).
- [ ] Write `dmfc/envs/conditions.py` with a function returning the canonical 79-condition evaluation set, frozen by a seed+index mapping.

## Milestone 3 — Models, training infrastructure, pilot runs

Goal: one IN and one flat-RNN pipeline baseline train end-to-end, produce valid run artifacts, and their hidden states can be extracted for analysis. No full sweep yet.

- [ ] IN model (`dmfc/models/interaction_network.py`): subclass the upstream higgsfield `InteractionNetwork` and override `forward()` to return `(predicted, effect_receivers)` where `effect_receivers` shape is `[batch, n_objects, effect_dim]`. This is the per-object relational state used for DMFC comparison. Do NOT edit the upstream file. (Flat-RNN baseline cut — timeline.)
- [ ] Object-graph construction for Mental Pong IN: define the object set (ball, paddle) and the relations (ball↔paddle). Walls are handled inside the env per PLANNING decision 3; revisit only if needed for the ablation.
- [ ] Loss-variant system (`dmfc/training/losses.py`): the four Rajalingham variants as loss-mask configs over model outputs. One implementation, four configs.
- [ ] Training loop (`dmfc/training/train.py`): config-driven, seed-driven, writes the full run artifact (config, git hash, seed, checkpoint, curves, stdout/stderr) to `runs/<ts>-<hash>/` per CONSTITUTION.
- [ ] Hidden-state serialization: at end of training, run one forward pass over all 79 eval conditions and save hidden states to `runs/<...>/hidden_states.npz` for later CPU-only analysis.
- [ ] Config files: one YAML per loss variant × a baseline hidden-unit count in `configs/`. Full sweep comes in Milestone 5.
- [ ] Pilot training run: train one IN on the `Vis&Occ` loss variant (`sim-mov`), seed 0, n_hidden=10. Confirm it converges, produces valid artifacts, and hidden states can be loaded.
- [ ] `tests/test_training.py`: verify the run-artifact writer produces all required files and that seed determinism actually works (same seed + config → same final checkpoint hash).

## Milestone 4 — Analysis pipeline (Fig. 5B primary, Fig. 4 secondary)

Goal: all four analysis metrics from PRD A6 work end-to-end on the pilot IN runs, and the reproduce_fig5b / reproduce_fig4 scripts regenerate extended figures.

- [ ] `dmfc/rajalingham/load.py`: adapter for loading the DMFC population data and the published RNN outputs from the Zenodo release. Decoupled from everything else; if the format changes, only this file breaks.
- [ ] `dmfc/analysis/endpoint_decoding.py` (PRIMARY — Fig. 5B metric): time-resolved linear decoding of endpoint ball_y from hidden states. Cross-validated across conditions. Returns Pearson r and RMSE vectors over time matching the 0–1200 ms axis of Fig. 5B.
- [ ] `dmfc/analysis/rdm.py`: pairwise-distance matrix computation for hidden states, restricted to occluded-epoch states per Rajalingham Fig. 4C.
- [ ] `dmfc/analysis/neural_consistency.py`: noise-adjusted correlation per Rajalingham Eq. 4, with split-half reliability estimation. Validate by reproducing the published RNN class means from Fig. 4D within noise.
- [ ] `dmfc/analysis/simulation_index.py`: linear decoder for ball position during occlusion, mean absolute error metric. Validate against Rajalingham's published SI distribution.
- [ ] `dmfc/analysis/reproduce_fig5b.py` (PRD F3 — PRIMARY DELIVERABLE): takes `--in-runs runs/in_*`, `--rajalingham-data data/dmfc/`, produces an extended Fig. 5B PNG with IN curve overlaid on DMFC + four RNN class curves.
- [ ] `dmfc/analysis/reproduce_fig4.py` (PRD F4 — SECONDARY DELIVERABLE): takes the same inputs, produces extended Fig. 4D/E with IN swarm added.
- [ ] Pipeline validation (PRD S1): run the full analysis against Rajalingham's released data + their published RNN outputs; confirm DMFC Fig. 5B curve matches published shape and that reproduced Neural Consistency means per RNN class fall within published error bars. If not within tolerance, debug the pipeline — do not proceed to Milestone 5 until this passes.

## Milestone 5 — Full IN sweep + primary results

Goal: all IN training runs complete, primary Fig. 5B result is known, secondary Fig. 4 result is known.

- [ ] Generate config files for the full experimental matrix: 4 loss variants × n_hidden {10, 20} × 5 seeds = 40 runs per PLANNING.
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
