# VERSIONS — Pinned Dependencies

**Last updated**: 2026-04-21

All dependencies loaded/installed at pinned exact versions per CONSTITUTION reproducibility rules. The canonical source is `pyproject.toml`; this file documents intent and compatibility notes.

## Core — runtime

| Package | Version | Source | Notes |
|---|---|---|---|
| python | 3.11.x | system / pyenv | 3.11 chosen for broad library compatibility as of 2026-04 |
| torch | TBD | PyPI | Pin to the latest stable at Milestone 1; match upstream higgsfield/interaction_network_pytorch's torch requirement if it's stricter |
| numpy | TBD | PyPI | Pin whatever torch pulls in; avoid `numpy>=2` if torch pins `<2` |
| scipy | TBD | PyPI | Needed for scipy.stats (permutation tests, sign-rank tests in pipeline validation) |
| scikit-learn | TBD | PyPI | Linear decoders (endpoint decoding, Simulation Index, Neural Consistency split-halves) |
| pandas | TBD | PyPI | Tabular results wrangling for the swarm plots |

## Core — analysis & plotting

| Package | Version | Source | Notes |
|---|---|---|---|
| matplotlib | TBD | PyPI | Figures for Fig. 5B and Fig. 4 reproductions |
| seaborn | TBD | PyPI | Swarm plots (Rajalingham Fig. 4D uses these) |
| pyyaml | TBD | PyPI | Config loading |

## Core — dev tooling

| Package | Version | Source | Notes |
|---|---|---|---|
| pytest | TBD | PyPI | CPU-only tests per CONSTITUTION |
| ruff | TBD | PyPI | Replaces black + flake8 + isort |
| mypy | TBD | PyPI | Type checking on `dmfc/` only |
| uv | TBD | PyPI (or standalone installer) | Package manager; `uv.lock` is the real source of truth |

## Upstream fork — higgsfield/interaction_network_pytorch

The fork's own dependencies take precedence where they conflict with the above. Milestone 1 task: read upstream's requirements, reconcile conflicts, document resolutions here.

| Upstream pin | Our pin | Resolution |
|---|---|---|
| TBD | TBD | TBD |

## Rajalingham data

Not a package, but versioning-relevant:

| Asset | Version | Source |
|---|---|---|
| DMFC Zenodo release | 1 (the accession at DOI 10.5281/zenodo.13952210) | https://doi.org/10.5281/zenodo.13952210 |
| jazlab/MentalPong code | TBD commit hash (pinned at Milestone 1 inspection) | https://github.com/jazlab/MentalPong |

## Compatibility notes

- If upstream higgsfield code pins an old torch (pre-2.0), we must either upgrade it in our fork copy (per CONSTITUTION: copy-and-modify, never edit-in-place) or accept the old torch. Revisit in Milestone 1 once upstream's `requirements.txt` is inspected.
- `numpy>=2` broke SciPy's pickling behavior for certain objects. If our environment ends up on numpy 2, any cached analysis pickles from numpy 1 runs need to be regenerated. Note in SCRATCHPAD if encountered.

## Update procedure

1. Update this file first.
2. Update pinned versions in `pyproject.toml`.
3. Re-run `uv lock` to refresh `uv.lock`.
4. Re-run the full test suite (`pytest`).
5. Re-run pipeline validation against Rajalingham's published numbers (Milestone 4 gate) — any dep change that breaks pipeline validation is a regression and must be reverted or investigated.
6. Commit with message: `chore(deps): bump <pkg> to <version>`.
