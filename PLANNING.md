# Planning — interaction-networks-dmfc

Generated from PRD.md + CONSTITUTION.md. Update whenever architecture or tooling changes.

## System architecture

Five components, each isolated in its own subpackage so they can be developed and tested independently:

```
dmfc/
├── envs/       — Mental Pong environment (stateful task simulator)
│                 Produces per-timestep observations + ground-truth ball/paddle state
│                 for training and evaluation. Matches Rajalingham's 79-condition eval set.
│
├── models/     — IN variants (from forked upstream code) + flat-RNN pipeline baseline
│                 All models share a common forward(obs_seq) → (hidden_states, outputs)
│                 interface so downstream analysis is model-agnostic.
│
├── training/   — Training loop, loss-variant configs (Intercept/Vis/Vis+Occ/Vis&Occ),
│                 seeding, checkpointing, run-artifact serialization. One training run
│                 = one config + one seed → one run directory.
│
├── analysis/   — The four metrics from PRD A6:
│                 (i) endpoint-decoding over time (Fig. 5B)       — PRIMARY
│                 (ii) pairwise-distance RDMs
│                 (iii) noise-adjusted Neural Consistency (Fig. 4D)
│                 (iv) Simulation Index (Fig. 4E)
│                 Plus the two figure-reproduction scripts (reproduce_fig5b, reproduce_fig4).
│
└── rajalingham/ — Thin adapter layer for loading their Zenodo data release and any
                   cached RNN outputs we need to overlay on our figures. Isolated here
                   because their data format may change and we want the mess contained.
```

Data flow for a single IN training run:

```
configs/in_vis_occ.yaml
         │
         ▼
dmfc.training.train  ──►  dmfc.envs.mental_pong  (generates trajectories)
         │                         │
         │                         ▼
         │                 dmfc.models.interaction_network (forward pass)
         │                         │
         │                         ▼
         │                 loss (per loss-variant mask)  ──►  backprop
         │
         ▼
runs/<ts>-<hash>/{config.yaml, checkpoint.pt, curves.jsonl, git.txt, seed.txt, log.txt}
```

Data flow for the primary Fig. 5B deliverable:

```
runs/in_*/                          data/dmfc/ (Zenodo release)
     │                                     │
     ▼                                     ▼
dmfc.analysis.endpoint_decoding     dmfc.rajalingham.load
     │                                     │
     └──────────────┬──────────────────────┘
                    ▼
        dmfc.analysis.reproduce_fig5b
                    ▼
              figures/fig5b_extended.png
```

## Technology stack

- **Language**: Python 3.11
- **Core numerics**: PyTorch (torch + torchvision), NumPy, SciPy
- **Analysis**: scikit-learn (linear decoders), statsmodels (stats), matplotlib + seaborn (plots). Pandas for tabular results wrangling.
- **Config**: PyYAML + a small dataclass layer for type-checked config loading.
- **Package manager**: uv (fast, reproducible, pyproject.toml-based). Fallback: pip + requirements.txt if uv unavailable.
- **Exact pinned versions**: see VERSIONS.md.

## Development tools

- **Editor/IDE**: user's choice (agent-facing work through Claude Code)
- **Testing framework**: pytest. CPU-only tests per CONSTITUTION.
- **Linter/formatter**: ruff (replaces black + flake8 + isort).
- **Type checking**: mypy on `dmfc/` only (upstream code exempt).
- **Version control**: git, hosted on GitHub (fork of higgsfield/interaction_network_pytorch).
- **CI**: none for now. A class project doesn't justify the setup overhead. Local `pytest` + `ruff` gates commits instead.

## Development workflow

1. Write a failing test for any new `dmfc/` code (CONSTITUTION testing priority: env > data > analysis > model).
2. Implement the minimum to make it pass.
3. `ruff check . && ruff format . && pytest` before committing.
4. Commits use conventional-commit-ish prefixes: `feat:`, `fix:`, `test:`, `chore:`, `analysis:`.
5. When work stops, `/summcommit` updates TASKS.md + SCRATCHPAD.md and commits.
6. Training runs are kicked off manually; each writes a self-describing run directory. No run, no result.

## Experimental matrix

The full IN training sweep, to be run in Milestone 3:

| Axis | Values | Count |
|---|---|---|
| Loss variant | Intercept, Vis, Vis+Occ, Vis&Occ | 4 |
| Hidden-unit count | 10, 20, 40 (exact values from Rajalingham's sweep — confirmed Milestone 1) | 3 |
| Seed | 0, 1, 2, 3, 4 (min 5 per CONSTITUTION) | 5 |

Estimated total: **60 training runs** (4 × 3 × 5). At ~4 hr/run on a single GPU (NF3 ceiling), serial wall-clock would be ~10 days. With cluster access and parallel job submission (SLURM array jobs), the full sweep can complete in a few hours of wall-clock if 60 GPUs are available simultaneously, or ~1–2 days with a smaller allocation.

## Environment / setup

```bash
# one-time setup
git clone <fork-url> interaction_network_pytorch
cd interaction_network_pytorch
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"    # installs pinned versions from pyproject.toml / VERSIONS.md

# get the Rajalingham data
mkdir -p data/dmfc
# download from https://doi.org/10.5281/zenodo.13952210 into data/dmfc/
# (manual step — do not commit data per CONSTITUTION)

# verify install
pytest -q
python -m dmfc.envs.mental_pong --render --seed 0
```

## Key design decisions and rationale

- **Gym-style env interface** for Mental Pong: matches the PyTorch/RL ecosystem's default abstraction, makes it trivial to swap in other envs if the project pivots, and keeps the env state explicit and loggable. Alternative (inline tensors in the training loop) was rejected because it couples env logic to training logic and makes the env harder to test.
- **One training run = one config + one seed**: no multi-seed runs in a single directory. Each seed gets its own artifact dir. Keeps seed accounting honest and allows dropping bad runs cleanly.
- **Upstream code untouched**: forced by CONSTITUTION. Implementation: if the upstream IN class needs a different signature for Mental Pong, subclass it in `dmfc/models/`; do not edit the original. If a given upstream change is too invasive to subclass, copy the file into `dmfc/models/` with a comment naming its upstream origin.
- **Analysis scripts CPU-only**: allows regenerating figures without GPU access, which matters for iteration on the writeup. Hidden states are serialized to `runs/<...>/hidden_states.npz` during training so analysis never needs to re-run inference.
- **uv over pip/conda**: reproducibility + speed. `uv.lock` pins the exact dependency tree. Conda was rejected because it adds a dependency on channel availability that could break reproducibility a year from now.

## Open architectural questions

- [ ] **What format is Rajalingham's Zenodo release actually in?** (Source Data files per paper; need to inspect — Milestone 1 task.) Determines the adapter implementation in `dmfc/rajalingham/`.
- [ ] **Can we extract their per-RNN Fig. 5B curves from their repo/data release, or do we only have the mean±SEM traces from the published figure?** If only the latter, the IN vs. RNN comparison on Fig. 5B uses their aggregated traces rather than individual-RNN curves — statistically weaker but still interpretable.
- [ ] **Does the upstream `higgsfield/interaction_network_pytorch` expose hidden states cleanly?** If not, the subclass in `dmfc/models/` needs a hook for extracting them at each timestep. Check in Milestone 2.
- [ ] **Wall reflection — a graph edge or an env-level state update?** If wall-ball collisions are handled inside the env (ball's velocity is flipped on collision), the IN doesn't need to learn reflection physics — just track the resulting trajectory. If walls are graph edges the IN must learn to use, it's closer to the "full relational physics" story but harder to train. Default: env handles reflection; walls included as graph context only for the ablation. Revisit if this turns out to matter scientifically.
