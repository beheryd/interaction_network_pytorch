# Project Requirements Document — interaction-networks-dmfc

**Class:** Foundations of Computational Cognitive Neuroscience

## Overview

Rajalingham et al. 2025 identify a specific discrepancy between macaque DMFC activity and their task-optimized RNNs on the Mental Pong task: DMFC exhibits **rapid early prediction of the ball's endpoint** (accurate linear decoding from ~250 ms after trial onset, well before the ball reaches the occluder), while **none** of their four RNN classes show this behavior — all RNNs instead show a slow, gradual rise in endpoint-decoding accuracy (their Fig. 5B). They flag this as an open problem: "the discrepancy between DMFC and RNNs highlights the need for additional models that would manifest dynamics commensurate with both early prediction and online simulation."

**This project tests whether an interaction network (IN; Battaglia et al. 2016), by virtue of its explicit object and pairwise-relation structure, closes that gap** — i.e., whether it reproduces DMFC's rapid early-prediction curve on Fig. 5B where Rajalingham's flat RNNs fail to.

The IN is also placed on the Fig. 4 axes (Neural Consistency vs. Simulation Index) to verify it matches RNN-level performance on the online-simulation axis that Rajalingham's paper was primarily about. The logic: if the IN merely closes Fig. 5B but underperforms on Fig. 4, that's not a general architectural advance — it would just be a model that does one thing better and another worse. The interesting claim requires both.

## Goals

- **G1 (PRIMARY) — Reproduce Rajalingham et al. 2025 Fig. 5B with the IN added as a new curve.** Compute time-resolved linear decoding of ball endpoint (Pearson r and RMSE vs. time) from IN hidden states, alongside the DMFC curve and the four RNN class curves. Test whether the IN's curve matches DMFC's shape (rapid rise ~250 ms, sustained plateau) rather than the RNNs' slow gradual rise.
- **G2 (SECONDARY) — Reproduce Rajalingham et al. 2025 Fig. 4D–G with the IN swarm added.** Verifies the IN is at least as good as RNNs on the online-simulation axis Rajalingham evaluated; establishes that any G1 advantage is not paid for by a G2 loss.
- **G3 (TERTIARY) — Relational ablation.** Compare IN vs. (a) IN with pairwise interaction terms removed, (b) IN with graph aggregation replaced with mean pooling on both the G1 (Fig. 5B) and G2 (Fig. 4) metrics. Isolates whether any IN advantage comes from object decomposition, from relational processing, or from both.

## Non-goals

- **NG1** — Not retraining any of Rajalingham's RNNs. Their 192 published RNN points and their published Fig. 5B RNN curves are used as-is.
- **NG2** — Not collecting new neural data. Only the Zenodo release (DOI 10.5281/zenodo.13952210) is used.
- **NG3** — Not using pixel inputs to the IN. The IN receives ball (x, y) during the visible epoch and a "ball invisible" flag during occlusion — matching the *information content* of Rajalingham's RNN inputs without replicating their pixel-to-position perception step. Paddle position is available throughout; wall positions are encoded as static graph edges. Rationale: this isolates the architectural hypothesis (object/relation structure) without conflating it with a perception-module hypothesis. A learned perception front-end is explicitly noted as future work in the writeup.
- **NG4** — Not modifying upstream `higgsfield/interaction_network_pytorch` code in-place (per CONSTITUTION).
- **NG5** — Not a PLATO-style (Piloto et al. 2022) implementation. Object states are provided; the project does not learn object discovery.
- **NG6** — Not claiming causal evidence for mental simulation or for any specific DMFC mechanism. Contribution is a model-comparison result: architecture X reproduces a specific DMFC signature that architecture Y failed to.
- **NG7** — Not producing a polished paper. Final deliverable is a class writeup + reproducible code.

## Requirements

### Architectural

- **A1** — Repository layout per CONSTITUTION: upstream IN code at repo root (untouched); new project code under `dmfc/`; configs under `configs/`; run artifacts under `runs/` (gitignored); neural data under `data/dmfc/` (gitignored).
- **A2** — Mental Pong environment implemented as a Gym-style env with deterministic seeding. Ball kinematics match Rajalingham's spec: rightward motion only, visible+occluded epochs each 15–45 timesteps, bounces ∈ {0, 1}, 79 evaluation conditions matching their published set.
- **A3** — IN input spec (Option 2, "visible-ball-position-only"): at each timestep the IN receives (ball_x, ball_y, visible_flag, paddle_y, paddle_vy). Walls are static graph edges with fixed parameters (positions, reflection coefficients). No pixel input; no perception front-end.
- **A4** — Models: (a) vanilla flat RNNs as internal pipeline sanity-check against Rajalingham's published numbers; (b) interaction network adapted from the forked `higgsfield/interaction_network_pytorch` for the Mental Pong domain.
- **A5** — Training loop supports all 4 Rajalingham loss variants (Intercept, Vis, Vis+Occ, Vis&Occ) via a loss-mask config, so the same code produces all IN classes.
- **A6** — Analysis pipeline: (i) time-resolved endpoint-decoding from hidden states (Fig. 5B metric), (ii) pairwise-distance RDM extraction, (iii) noise-adjusted Neural Consistency Score per Rajalingham Eq. 4, (iv) Simulation Index via linear ball-position decoder during occlusion. (i) is computed on the visible-epoch states; (ii)–(iv) are computed on occluded-epoch states matching Rajalingham's Fig. 4 convention.

### Functional

- **F1** — `python -m dmfc.envs.mental_pong --render` runs a visual check of the task environment against Rajalingham's published condition list.
- **F2** — `python -m dmfc.training.train --config configs/in_vis_occ.yaml --seed 0` trains one IN, writes full run artifact (config, git hash, seed, checkpoint, curves, logs) to `runs/<timestamp>-<hash>/`.
- **F3 (PRIMARY DELIVERABLE)** — `python -m dmfc.analysis.reproduce_fig5b --in-runs runs/in_* --rajalingham-data data/dmfc/` regenerates an extended Fig. 5B with the IN curve added alongside DMFC and the four RNN class curves.
- **F4 (SECONDARY DELIVERABLE)** — `python -m dmfc.analysis.reproduce_fig4 --in-runs runs/in_* --rajalingham-data data/dmfc/` regenerates an extended Fig. 4D/E with the IN swarm added.
- **F5 (TERTIARY DELIVERABLE)** — Ablation script runs IN with (a) full relational structure, (b) no pairwise interaction terms, (c) graph aggregation replaced with mean pooling; produces comparison on both F3 and F4 metrics.

### Non-functional

- **NF1** — Every reported number is mean ± s.e.m. over ≥5 seeds (per CONSTITUTION).
- **NF2** — All training runs write deterministic-seed-bearing run directories; any figure in the writeup names the exact run dir(s) it consumed.
- **NF3** — Training a single IN completes in under ~4 hours on a single GPU (so the full 4-variant × hidden-unit sweep × 5 seeds matrix fits in a few days of wall-clock).
- **NF4** — Analysis scripts are CPU-runnable; no GPU required to regenerate figures from cached run artifacts.

## Success criteria

- **S1 (pipeline validation)** — Applied to Rajalingham's released DMFC data, the reproduced Fig. 5B DMFC curve matches their published curve shape (rapid rise ~250 ms, r > 0.85 plateau by ~500 ms). Reproduced Neural Consistency Scores for each RNN class (G2) match their Fig. 4D swarm distributions within published error bars.
- **S2 (primary outcome)** — A clear answer to G1: does the IN's Fig. 5B endpoint-decoding curve match DMFC's rapid-rise pattern, or does it match the RNNs' slow-gradual-rise pattern, or something in between? Quantified by (i) time-to-threshold-r comparison between IN and DMFC vs. IN and RNN-class-mean, (ii) RMSE-AUC over the 0–1200 ms window. "IN matches DMFC", "IN matches RNNs", and "IN intermediate" are all acceptable findings; reported honestly per CONSTITUTION.
- **S3 (secondary outcome)** — IN swarm on extended Fig. 4D/E lands *at least within* the RNN swarm range on Neural Consistency. This is a no-regression check — the interesting claim requires the IN to not be worse on online simulation than Rajalingham's RNNs.
- **S4 (tertiary outcome)** — Ablation answers: which component of the IN (object decomposition, pairwise relations) drives the G1 result, and does the same component drive the G2 result? A dissociation (e.g., relations matter for G1 but not G2) would be more informative than a uniform degradation.
- **S5** — All figures in the final writeup are regenerable from scripts in `dmfc/analysis/` pointing at named run directories.
- **S6** — Final deliverable: writeup + git repo meeting CONSTITUTION's writeup-discipline rules, submitted by the class deadline.
