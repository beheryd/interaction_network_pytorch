# Project Constitution — interaction-networks-dmfc

Project-specific rules. General coding preferences live in `~/.claude/CLAUDE.md`.
Rules here are specific to this project's stack (PyTorch + neural-data modeling)
and its purpose (a class project for *Foundations of Computational Cognitive
Neuroscience* whose core contribution is a head-to-head model comparison against
DMFC data).

## Upstream / fork discipline (NON-NEGOTIABLE)

- This repo is a fork of `higgsfield/interaction_network_pytorch`. Upstream files
  (anything present at the initial fork commit) are treated as vendored
  reference code.
- FORBIDDEN: modifying upstream files in place. If upstream behavior needs to
  change, subclass, wrap, or copy the relevant code into `dmfc/` and edit the
  copy — never the original.
- New project code lives under `dmfc/`. Upstream code stays at the repo root
  where it was.
- If upstream code is copied into `dmfc/`, retain the original header/license
  and add a comment noting which upstream file it derived from.

## Reproducibility (NON-NEGOTIABLE)

- Every training run sets seeds for `random`, `numpy`, `torch`, and
  `torch.cuda` from a single `seed` argument.
- Every training run writes a run directory containing: the exact config used,
  git commit hash, seed, final model checkpoint, training curves, and the
  stdout/stderr log. No run is valid without this artifact.
- FORBIDDEN: hardcoded hyperparameters buried in scripts. All hyperparameters
  go through a config file (YAML or dataclass) that gets saved into the run
  directory.
- FORBIDDEN: `torch.load(..., weights_only=False)` on checkpoints whose
  provenance isn't this repo.
- Analysis notebooks that report numbers for the writeup must name the exact
  run directory they loaded results from.

## Model comparison discipline (NON-NEGOTIABLE)

- Baselines (flat RNNs) and the interaction-network model are trained on
  *identical* task data, identical train/val/test splits, and identical
  optimizer/training-budget settings. Any deviation is a bug.
- Primary capacity-matching criterion is **hidden-unit count**, swept across
  the same values used by Rajalingham et al. 2025 (exact values to be
  extracted from https://github.com/jazlab/MentalPong during Milestone 1 and
  recorded in PLANNING.md before any IN training begins).
- All IN loss variants (IN_Intercept, IN_Vis, IN_Vis+Occ, IN_Vis&Occ) are
  trained at this full hidden-unit sweep so the IN swarm on the replicated
  Fig. 4 has the same shape as the published RNN swarm.
- Parameter counts are reported alongside hidden-unit counts for every
  trained model, so readers can assess capacity matching independently.
- FORBIDDEN: placing a single IN point against the full RNN swarm and
  claiming a comparison. Swarm-vs-swarm only.
- Any deviation from Rajalingham's hidden-unit sweep is justified in
  PLANNING.md with an explicit rationale.
- FORBIDDEN: cherry-picking seeds. All reported numbers are mean ± s.e.m. over
  a pre-declared number of seeds (minimum 5).
- Representational-alignment metrics (RSA, linear decoding) are computed with
  held-out data the model never saw during training.

## Neural data handling (NON-NEGOTIABLE)

- DMFC descriptors or raw data obtained from Rajalingham et al. 2025 (or
  requested from the authors) are read-only. They live in `data/dmfc/` which
  is `.gitignore`'d.
- FORBIDDEN: committing neural data to the repo. A README in `data/dmfc/`
  documents what files are expected and where they came from.
- If processed data isn't obtainable, the project falls back to comparing
  against re-trained open-source RNN baselines matched to Rajalingham's
  architecture specs — this fallback is documented in PLANNING.md and any
  writeup.

## Testing (RECOMMENDED, not NON-NEGOTIABLE)

- Test coverage priority: task-environment code > data-loading code > analysis
  code > model code (model correctness is mostly validated by training
  dynamics, not unit tests).
- The Mental Pong environment has unit tests verifying ball trajectories,
  occluder timing, and reward/termination logic against hand-computed
  expected values for at least 3 fixed seeds.
- FORBIDDEN: tests that depend on GPU availability. CPU-only tests.

## Project structure (RECOMMENDED)

```
interaction_network_pytorch/       # repo root (fork)
├── CLAUDE.md, PRD.md, PLANNING.md, TASKS.md, SCRATCHPAD.md,
│   CONSTITUTION.md, VERSIONS.md   # context files
├── <upstream files>                # UNTOUCHED (Battaglia 2016 IN impl)
├── dmfc/                           # all new project code
│   ├── envs/                       # Mental Pong task env
│   ├── models/                     # IN variants + RNN baselines
│   ├── training/                   # train loops, configs, seed mgmt
│   ├── analysis/                   # RSA, linear decoding, ablations
│   └── __init__.py
├── configs/                        # YAML configs per experiment
├── runs/                           # training run artifacts (.gitignore'd)
├── data/dmfc/                      # neural data (.gitignore'd)
├── notebooks/                      # exploratory analysis
└── tests/
```

## Writeup discipline (NON-NEGOTIABLE for final submission)

- Every figure in the writeup is regenerable from a script in `dmfc/analysis/`
  pointed at a named run directory. No hand-edited plots.
- Any negative result is reported, not buried. If the IN does not beat the
  RNN baseline, that is the finding — document it and investigate why rather
  than tuning until the preferred conclusion appears.
