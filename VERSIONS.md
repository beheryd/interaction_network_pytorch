# VERSIONS — Pinned Dependencies

**Last updated**: 2026-04-25 (Milestone 1 scaffolding)

All dependencies installed at pinned exact versions per CONSTITUTION reproducibility rules. The canonical source is `pyproject.toml`; `uv.lock` records the full resolved tree (50 packages); this file documents intent and compatibility notes.

## Core — runtime

| Package | Version | Source | Notes |
|---|---|---|---|
| python | 3.11.15 | uv-managed CPython build | 3.11 chosen for broad library compatibility; `<3.12` upper bound to avoid surprises until torch / numpy / scipy lockstep is verified |
| torch | 2.4.1 | PyPI | Modern stable; upstream higgsfield code (Battaglia 2016 reimplementation) is old enough that we are intentionally using a current torch — copy-and-modify the upstream file in `dmfc/models/` if any API drift bites |
| numpy | 1.26.4 | PyPI | Held to `1.x` deliberately — `numpy>=2` has caused pickling regressions in scientific stacks (see compatibility notes) |
| scipy | 1.14.1 | PyPI | `scipy.stats` for permutation / sign-rank tests in pipeline validation |
| scikit-learn | 1.5.2 | PyPI | Linear decoders for endpoint decoding, Simulation Index, Neural Consistency split-halves |
| pandas | 2.2.3 | PyPI | Tabular results wrangling for the swarm plots |

## Core — analysis & plotting

| Package | Version | Source | Notes |
|---|---|---|---|
| matplotlib | 3.9.2 | PyPI | Figures for Fig. 5B and Fig. 4 reproductions |
| seaborn | 0.13.2 | PyPI | Swarm plots (Rajalingham Fig. 4D uses these) |
| pyyaml | 6.0.2 | PyPI | Config loading |

## Core — dev tooling

| Package | Version | Source | Notes |
|---|---|---|---|
| pytest | 8.3.3 | PyPI | CPU-only tests per CONSTITUTION |
| ruff | 0.6.9 | PyPI | Replaces black + flake8 + isort |
| mypy | 1.11.2 | PyPI | Type checking on `dmfc/` only |
| uv | 0.11.7 | Homebrew | Package manager; `uv.lock` is the real source of truth |

## Upstream fork — higgsfield/interaction_network_pytorch

The fork's only "dependency" is whatever torch the original code expects (the upstream repo predates pinned `requirements.txt`). Per CONSTITUTION (upstream / fork discipline), we do **not** edit upstream files in place; if torch 2.4 breaks the upstream `InteractionNetwork`, we copy the file into `dmfc/models/` and patch the copy.

| Upstream pin | Our pin | Resolution |
|---|---|---|
| (untracked / pre-PEP 518) | torch 2.4.1 | Subclass or copy upstream code into `dmfc/models/` if API drift surfaces during Milestone 3 |

## Rajalingham data

Not a package, but versioning-relevant:

| Asset | Version | Source |
|---|---|---|
| DMFC Zenodo release | 1 (DOI 10.5281/zenodo.13952210) | https://doi.org/10.5281/zenodo.13952210 |
| jazlab/MentalPong code | bundled inside the Zenodo release at `code/` | https://github.com/jazlab/MentalPong (no separate clone needed) |

## Compatibility notes

- We deliberately stay on numpy `1.26.x`. The `numpy>=2` cutover broke pickle round-trips for some SciPy / sklearn objects, and the Rajalingham Zenodo `.pkl` files were produced under numpy 1; loading them under numpy 2 can produce silent dtype shifts.
- torch 2.4 → 2.5 is a known smooth bump if we ever need to upgrade; bumping past 2.5 would cross a Python-min-version line.
- The upstream `Interaction Network.ipynb` has not been re-executed under torch 2.4. It will be exercised live in Milestone 3 when we subclass `InteractionNetwork`; any breakage gets a `dmfc/models/` copy rather than an in-place upstream edit.

## Update procedure

1. Update this file first (the `2026-04-25` entries reflect Milestone 1 install).
2. Update pinned versions in `pyproject.toml`.
3. Re-run `uv lock` to refresh `uv.lock`.
4. Re-run the full test suite (`pytest`).
5. Re-run pipeline validation against Rajalingham's published numbers (Milestone 4 gate) — any dep change that breaks pipeline validation is a regression and must be reverted or investigated.
6. Commit with message: `chore(deps): bump <pkg> to <version>`.
